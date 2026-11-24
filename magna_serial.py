"""
Controlador Serial Marlin Seguro para Hellbot Magna SE (COM4 @ 115200 baud)
Diseñado con control de flujo Ping-Pong estricto ('ok' por línea) para parada rápida
(importante porque M115 reportó Cap:EMERGENCY_PARSER:0).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import threading
import time
from typing import Callable
import serial
import serial.tools.list_ports


@dataclass
class PrinterTelemetry:
    connected: bool = False
    port: str = ""
    baud: int = 115200
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    hotend_temp: float = 0.0
    bed_temp: float = 0.0
    firmware: str = ""


def find_printer_ports() -> list[tuple[str, str]]:
    out = []
    for p in serial.tools.list_ports.comports():
        desc = f"{p.device} - {p.description}"
        out.append((p.device, desc))
    return out


class MagnaController:
    def __init__(self, log_cb: Callable[[str], None] | None = None):
        self._ser: serial.Serial | None = None
        self._lock = threading.RLock()
        self._log_cb = log_cb or (lambda msg: None)
        self.telemetry = PrinterTelemetry()

        self._job_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self.job_running = False
        self.job_progress = (0, 0)

    def log(self, msg: str) -> None:
        self._log_cb(msg)

    def connect(self, port: str = "COM4", baud: int = 115200, reset_board: bool = False) -> PrinterTelemetry:
        with self._lock:
            self.disconnect()
            ser = serial.Serial()
            ser.port = port
            ser.baudrate = baud
            ser.timeout = 1.5
            ser.dtr = bool(reset_board)
            ser.rts = bool(reset_board)
            ser.open()
            time.sleep(2.2 if reset_board else 0.4)
            ser.reset_input_buffer()
            self._ser = ser
            self.telemetry.connected = True
            self.telemetry.port = port
            self.telemetry.baud = baud

        # Confirmación inicial + Laser/Fan OFF por seguridad
        resp = self.send_and_wait("M107", timeout=2.0)
        m115 = self.send_and_wait("M115", timeout=2.5)
        for line in m115:
            if "FIRMWARE_NAME:" in line:
                self.telemetry.firmware = line
        self.send_and_wait("M117 XILOGRAFIA STUDIO OK", timeout=1.5)
        self.refresh_status()
        self.log(f"[CONECTADO] {port} @ {baud} bps | ok={bool(resp)}")
        return self.telemetry

    def disconnect(self) -> None:
        self._stop_event.set()
        with self._lock:
            if self._ser and self._ser.is_open:
                try:
                    self._ser.close()
                except Exception:
                    pass
            self._ser = None
            self.telemetry.connected = False

    def send_and_wait(self, gcode: str, timeout: float = 10.0) -> list[str]:
        clean = gcode.split(";")[0].strip()
        if not clean:
            return ["ok"]
        with self._lock:
            if not self._ser or not self._ser.is_open:
                raise RuntimeError("Impresora no conectada.")
            self._ser.reset_input_buffer()
            self._ser.write((clean + "\n").encode("ascii", errors="ignore"))
            self._ser.flush()

            lines: list[str] = []
            t0 = time.time()
            while time.time() - t0 < timeout:
                raw = self._ser.readline()
                if not raw:
                    continue
                txt = raw.decode("utf-8", errors="replace").strip()
                if not txt:
                    continue
                lines.append(txt)
                # Marlin responde 'ok' (o 'ok T:...') cuando acepta/termina según el comando
                low = txt.lower()
                if low == "ok" or low.startswith("ok "):
                    return lines
                if "error:" in low:
                    self.log(f"[MARLIN ERR] {txt}")
                    return lines
            self.log(f"[TIMEOUT] Sin 'ok' tras {timeout:.1f}s para: {clean}")
            return lines

    def refresh_status(self) -> PrinterTelemetry:
        if not self.telemetry.connected:
            return self.telemetry
        pos_lines = self.send_and_wait("M114", timeout=2.0)
        for l in pos_lines:
            m = re.search(r"X:([-\d.]+)\s+Y:([-\d.]+)\s+Z:([-\d.]+)", l)
            if m:
                self.telemetry.x = float(m.group(1))
                self.telemetry.y = float(m.group(2))
                self.telemetry.z = float(m.group(3))
        t_lines = self.send_and_wait("M105", timeout=2.0)
        for l in t_lines:
            mt = re.search(r"T:([-\d.]+)", l)
            mb = re.search(r"B:([-\d.]+)", l)
            if mt:
                self.telemetry.hotend_temp = float(mt.group(1))
            if mb:
                self.telemetry.bed_temp = float(mb.group(1))
        return self.telemetry

    def jog(self, axis: str, delta_mm: float) -> PrinterTelemetry:
        axis = axis.upper()
        if axis not in ("X", "Y", "Z"):
            raise ValueError(f"Eje inválido: {axis}")
        feed = 180 if axis == "Z" else 1500
        self.send_and_wait("G91")
        self.send_and_wait(f"G1 {axis}{delta_mm:.3f} F{feed}", timeout=15.0)
        self.send_and_wait("M400", timeout=15.0)
        self.send_and_wait("G90")
        return self.refresh_status()

    def home_xy_safe(self) -> PrinterTelemetry:
        """Hace Home SOLAMENTE de X e Y tras levantar Z +10mm (nunca baja Z contra la madera)."""
        self.log("[SAFE HOME] Subiendo Z +10mm y haciendo Home X/Y...")
        self.send_and_wait("G91")
        self.send_and_wait("G0 Z10.0 F180", timeout=15.0)
        self.send_and_wait("M400", timeout=15.0)
        self.send_and_wait("G90")
        self.send_and_wait("G28 X Y", timeout=45.0)
        return self.refresh_status()

    def set_work_zero(self) -> PrinterTelemetry:
        self.send_and_wait("G92 X0 Y0 Z0")
        self.send_and_wait("M117 CERO FIJADO (0,0,0)")
        self.log("[CERO] Posicion actual fijada como X=0 Y=0 Z=0 sobre el taco.")
        return self.refresh_status()

    def trace_air_box(self, width_mm: float, height_mm: float, off_x: float = 0.0, off_y: float = 0.0, z_air: float = 8.0) -> None:
        """Recorre los 4 bordes del área de talla en el aire para validar encuadre sobre la madera."""
        self.log(f"[TEST ENCUADRE] Trazando rectangulo {width_mm:.1f}x{height_mm:.1f} mm a Z+{z_air:.1f}mm...")
        cmds = [
            "M107",
            "G90",
            f"G0 Z{z_air:.2f} F180",
            f"G0 X{off_x:.2f} Y{off_y:.2f} F1800",
            f"G1 X{off_x + width_mm:.2f} Y{off_y:.2f} F1500",
            f"G1 X{off_x + width_mm:.2f} Y{off_y + height_mm:.2f} F1500",
            f"G1 X{off_x:.2f} Y{off_y + height_mm:.2f} F1500",
            f"G1 X{off_x:.2f} Y{off_y:.2f} F1500",
            "M400",
        ]
        for c in cmds:
            self.send_and_wait(c, timeout=20.0)
        self.refresh_status()
        self.log("[TEST ENCUADRE] Contorno completado en el aire.")

    def emergency_stop(self) -> None:
        self._stop_event.set()
        self._pause_event.clear()
        with self._lock:
            if self._ser and self._ser.is_open:
                try:
                    self._ser.write(b"\nM410\nM107\nM18\n")
                    self._ser.flush()
                except Exception:
                    pass
        self.log("[EMERGENCIA] M410 (Quickstop) + M107 (OFF) + M18 (Motores libres) enviados.")

    def toggle_pause(self) -> bool:
        if self._pause_event.is_set():
            self._pause_event.clear()
            self.log("[REANUDADO] Continuando trayectoria...")
            return False
        self._pause_event.set()
        self.log("[PAUSA] Pausando tras el segmento actual y subiendo Z +10mm...")
        return True

    def start_gcode_file(
        self,
        gcode_path: str | Path,
        progress_cb: Callable[[int, int, str], None] | None = None,
        done_cb: Callable[[bool], None] | None = None,
    ) -> None:
        if self.job_running:
            raise RuntimeError("Ya hay un trabajo ejecutándose.")
        raw_lines = Path(gcode_path).read_text(encoding="utf-8").splitlines()
        cmds = [l.split(";")[0].strip() for l in raw_lines if l.split(";")[0].strip()]
        if not cmds:
            raise ValueError("El archivo G-Code está vacío.")

        self._stop_event.clear()
        self._pause_event.clear()
        self.job_running = True
        self.job_progress = (0, len(cmds))

        def worker():
            ok_finished = False
            paused_retracted = False
            try:
                for idx, cmd in enumerate(cmds, start=1):
                    if self._stop_event.is_set():
                        self.log("[DETENIDO] Trabajo abortado por operador.")
                        break

                    while self._pause_event.is_set() and not self._stop_event.is_set():
                        if not paused_retracted:
                            self.send_and_wait("M400", timeout=20.0)
                            self.send_and_wait("G91")
                            self.send_and_wait("G0 Z10.0 F180", timeout=15.0)
                            self.send_and_wait("G90")
                            paused_retracted = True
                        time.sleep(0.15)

                    if self._stop_event.is_set():
                        break

                    if paused_retracted:
                        self.send_and_wait("G91")
                        self.send_and_wait("G0 Z-10.0 F150", timeout=15.0)
                        self.send_and_wait("G90")
                        paused_retracted = False

                    self.send_and_wait(cmd, timeout=30.0)
                    self.job_progress = (idx, len(cmds))
                    if progress_cb and (idx % 10 == 0 or idx == len(cmds)):
                        progress_cb(idx, len(cmds), cmd)
                else:
                    ok_finished = True
            except Exception as exc:
                self.log(f"[ERROR STREAM] {exc}")
            finally:
                try:
                    if not ok_finished and self.telemetry.connected:
                        self.send_and_wait("M107", timeout=2.0)
                        self.send_and_wait("G91", timeout=2.0)
                        self.send_and_wait("G0 Z10.0 F180", timeout=15.0)
                        self.send_and_wait("G90", timeout=2.0)
                except Exception:
                    pass
                self.job_running = False
                if done_cb:
                    done_cb(ok_finished)

        self._job_thread = threading.Thread(target=worker, daemon=True)
        self._job_thread.start()
