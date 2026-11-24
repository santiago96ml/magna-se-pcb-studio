"""
Magna SE Mini — PCB & Circuit Studio (Fabricación de Circuitos Eléctricos Sin Láser)
Interfaz gráfica Tkinter lista para:
  1. Conectar a la Hellbot Magna SE Mini (COM4 @ 115200 baud, sin reset accidental).
  2. Calibrar el Cero y el Nivel de las 4 esquinas sobre la placa virgen de cobre (FR4 / Pertinax).
  3. Generar G-Code para:
     - Fresado por Aislamiento de Pistas (pcb_isolation, Z=-0.08mm, 1 a 4 anillos)
     - Trazado con Marcador Indeleble Antiácido para Cloruro Férrico (pcb_marker_resist)
     - Perforado / Centrado automático de agujeros Through-Hole 2.54mm (pcb_drill_holes)
  4. Incluye generador automático del PCB Carrier del proyecto Tanque (2x ESP32-S3 + Arduino + Divisor 5V->3.3V).
"""

from __future__ import annotations

from pathlib import Path
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2

from gcode_generator import (
    XiloConfig,
    XiloStats,
    create_tank_carrier_pcb_layout,
    generate_woodcut_gcode,
    get_svg_info,
    render_svg_to_gray_mm,
)
from magna_serial import MagnaController, find_printer_ports


BASE_DIR = Path(__file__).resolve().parent
WORK_DIR = BASE_DIR / "trabajos"
WORK_DIR.mkdir(parents=True, exist_ok=True)


class MagnaStudioApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Magna SE Mini — PCB & Circuit Studio (Grabado de Circuitos Eléctricos Sin Láser)")
        self.root.geometry("1280x850")
        self.root.configure(bg="#181a1f")

        self.ctrl = MagnaController(log_cb=self._append_log_threadsafe)
        self.current_img_path: str | None = None
        self.last_stats: XiloStats | None = None
        self._preview_tk: tk.PhotoImage | None = None

        self._build_styles()
        self._build_ui()
        self._refresh_ports()
        self._load_tank_pcb_template()

    def _build_styles(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#181a1f")
        style.configure("Card.TFrame", background="#22262e", relief="flat")
        style.configure("TLabel", background="#181a1f", foreground="#e6eaf2", font=("Segoe UI", 10))
        style.configure("Card.TLabel", background="#22262e", foreground="#e6eaf2", font=("Segoe UI", 9))
        style.configure("Title.TLabel", background="#181a1f", foreground="#5ce1e6", font=("Segoe UI", 13, "bold"))
        style.configure("TCheckbutton", background="#22262e", foreground="#e6eaf2", font=("Segoe UI", 9))

    def _build_ui(self):
        top_bar = ttk.Frame(self.root, padding=10)
        top_bar.pack(fill="x")

        ttk.Label(top_bar, text="Magna SE Mini — PCB & Circuit Studio", style="Title.TLabel").pack(side="left", padx=(0, 18))

        ttk.Label(top_bar, text="Puerto:").pack(side="left")
        self.port_var = tk.StringVar(value="COM4")
        self.port_combo = ttk.Combobox(top_bar, textvariable=self.port_var, width=22, state="readonly")
        self.port_combo.pack(side="left", padx=6)
        tk.Button(top_bar, text="Actualizar", command=self._refresh_ports, bg="#2c323d", fg="#fff", relief="flat", padx=8).pack(side="left", padx=3)

        self.btn_conn = tk.Button(top_bar, text="Conectar (115200)", command=self._toggle_connect, bg="#198754", fg="#fff", font=("Segoe UI", 9, "bold"), relief="flat", padx=12)
        self.btn_conn.pack(side="left", padx=8)

        self.status_var = tk.StringVar(value="Desconectada | X:0.00 Y:0.00 Z:0.00")
        ttk.Label(top_bar, textvariable=self.status_var, foreground="#9ee6b0").pack(side="left", padx=12)

        tk.Button(
            top_bar, text="STOP EMERGENCIA (M410)", command=self._emergency_stop,
            bg="#d92b3a", fg="#fff", font=("Segoe UI", 10, "bold"), relief="flat", padx=14, pady=4
        ).pack(side="right")

        main = ttk.Frame(self.root, padding=(10, 0, 10, 10))
        main.pack(fill="both", expand=True)

        left = ttk.Frame(main, width=435)
        left.pack(side="left", fill="y", padx=(0, 10))
        left.pack_propagate(False)

        self._build_jog_panel(left)
        self._build_pcb_params_panel(left)

        right = ttk.Frame(main)
        right.pack(side="left", fill="both", expand=True)

        self._build_preview_and_log(right)

    def _build_jog_panel(self, parent: ttk.Frame):
        card = ttk.Frame(parent, style="Card.TFrame", padding=10)
        card.pack(fill="x", pady=(0, 8))

        ttk.Label(card, text="1. Calibración sobre Placa Virgen de Cobre (FR4)", style="Card.TLabel", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(0, 4))

        step_row = ttk.Frame(card, style="Card.TFrame")
        step_row.pack(fill="x", pady=2)
        ttk.Label(step_row, text="Paso (mm):", style="Card.TLabel").pack(side="left")
        self.step_var = tk.DoubleVar(value=1.0)
        for val in (0.05, 0.1, 1.0, 5.0, 20.0):
            tk.Radiobutton(
                step_row, text=str(val), variable=self.step_var, value=val,
                bg="#22262e", fg="#fff", selectcolor="#181a1f", activebackground="#22262e"
            ).pack(side="left", padx=3)

        grid = ttk.Frame(card, style="Card.TFrame")
        grid.pack(pady=4)

        btn_cfg = dict(width=6, bg="#303642", fg="#fff", relief="flat", pady=4)
        tk.Button(grid, text="Y+", command=lambda: self._jog("Y", 1), **btn_cfg).grid(row=0, column=1, padx=3, pady=2)
        tk.Button(grid, text="Z+", command=lambda: self._jog("Z", 1), bg="#2a4a6a", fg="#fff", width=6, relief="flat", pady=4).grid(row=0, column=3, padx=(14, 3), pady=2)

        tk.Button(grid, text="X-", command=lambda: self._jog("X", -1), **btn_cfg).grid(row=1, column=0, padx=3, pady=2)
        tk.Button(grid, text="CERO\nCOBRE", command=self._set_zero, width=6, bg="#c97a16", fg="#fff", font=("Segoe UI", 8, "bold"), relief="flat").grid(row=1, column=1, padx=3, pady=2)
        tk.Button(grid, text="X+", command=lambda: self._jog("X", 1), **btn_cfg).grid(row=1, column=2, padx=3, pady=2)

        tk.Button(grid, text="Y-", command=lambda: self._jog("Y", -1), **btn_cfg).grid(row=2, column=1, padx=3, pady=2)
        tk.Button(grid, text="Z-", command=lambda: self._jog("Z", -1), bg="#2a4a6a", fg="#fff", width=6, relief="flat", pady=4).grid(row=2, column=3, padx=(14, 3), pady=2)

        actions = ttk.Frame(card, style="Card.TFrame")
        actions.pack(fill="x", pady=(4, 0))
        tk.Button(actions, text="Home XY Seguro (Sube Z)", command=self._home_xy, bg="#3c4454", fg="#fff", relief="flat", padx=6, pady=4).pack(side="left", fill="x", expand=True, padx=(0, 3))
        tk.Button(actions, text="Test Encuadre Aéreo", command=self._trace_box, bg="#2563eb", fg="#fff", relief="flat", padx=6, pady=4).pack(side="left", fill="x", expand=True, padx=(3, 0))

    def _build_pcb_params_panel(self, parent: ttk.Frame):
        card = ttk.Frame(parent, style="Card.TFrame", padding=10)
        card.pack(fill="both", expand=True)

        ttk.Label(card, text="2. Generador de Circuito Eléctrico (PCB)", style="Card.TLabel", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(0, 6))

        load_row = ttk.Frame(card, style="Card.TFrame")
        load_row.pack(fill="x", pady=(0, 4))
        tk.Button(load_row, text="Cargar SVG/PNG/JPG", command=self._pick_image, bg="#3b82f6", fg="#fff", relief="flat", pady=4).pack(side="left", fill="x", expand=True, padx=(0, 2))
        tk.Button(load_row, text="PCB Tanque (IPC-2221)", command=self._load_tank_pcb_template, bg="#0d9488", fg="#fff", relief="flat", pady=4).pack(side="left", fill="x", expand=True, padx=2)
        tk.Button(load_row, text="Cámara AOI", command=self._run_camera_aoi, bg="#d97706", fg="#fff", relief="flat", pady=4).pack(side="left", fill="x", expand=True, padx=(2, 0))

        self.img_name_var = tk.StringVar(value="Ningún circuito cargado")
        ttk.Label(card, textvariable=self.img_name_var, style="Card.TLabel", foreground="#93c5fd").pack(anchor="w", pady=(0, 4))

        form = ttk.Frame(card, style="Card.TFrame")
        form.pack(fill="x")

        self.var_mode = tk.StringVar(value="pcb_marker_resist")
        self.var_width = tk.DoubleVar(value=65.0)
        self.var_tool = tk.DoubleVar(value=0.75)
        self.var_iso_rings = tk.IntVar(value=2)
        self.var_depth = tk.DoubleVar(value=0.45)
        self.var_stepdown = tk.DoubleVar(value=0.45)
        self.var_cutfeed = tk.DoubleVar(value=360.0)
        self.var_thresh = tk.IntVar(value=135)
        self.var_mirror = tk.BooleanVar(value=True)
        self.var_invert = tk.BooleanVar(value=False)

        mode_combo = ttk.Combobox(
            form,
            textvariable=self.var_mode,
            values=["pcb_marker_resist", "pcb_isolation", "pcb_drill_holes", "pcb_laser_ablation", "relief_pocket"],
            state="readonly",
            width=18,
        )
        mode_combo.bind("<<ComboboxSelected>>", self._on_mode_preset_changed)

        rows = [
            ("Modo Fabricación PCB:", mode_combo),
            ("Ancho Placa (mm):", tk.Entry(form, textvariable=self.var_width, width=10)),
            ("Punta Fibra / Herram. (mm):", tk.Entry(form, textvariable=self.var_tool, width=10)),
            ("Manos de Tinta / Anillos:", tk.Entry(form, textvariable=self.var_iso_rings, width=10)),
            ("Presión Z Fibra (mm):", tk.Entry(form, textvariable=self.var_depth, width=10)),
            ("Paso Z / pasada (mm):", tk.Entry(form, textvariable=self.var_stepdown, width=10)),
            ("Vel. Trazo XY (mm/min):", tk.Entry(form, textvariable=self.var_cutfeed, width=10)),
        ]
        for r_i, (lbl, widget) in enumerate(rows):
            ttk.Label(form, text=lbl, style="Card.TLabel").grid(row=r_i, column=0, sticky="w", pady=2)
            widget.grid(row=r_i, column=1, sticky="e", pady=2, padx=(6, 0))

        # Nivelación Bilineal de las 4 esquinas de la placa de cobre
        lvl_frame = ttk.Frame(card, style="Card.TFrame")
        lvl_frame.pack(fill="x", pady=4)
        ttk.Label(lvl_frame, text="Nivelación Z Esquinas (mm):", style="Card.TLabel", foreground="#fcd34d").grid(row=0, column=0, columnspan=4, sticky="w")
        self.var_z_bl = tk.DoubleVar(value=0.0)
        self.var_z_br = tk.DoubleVar(value=0.0)
        self.var_z_tl = tk.DoubleVar(value=0.0)
        self.var_z_tr = tk.DoubleVar(value=0.0)
        for idx, (lbl, var) in enumerate([("BL:", self.var_z_bl), ("BR:", self.var_z_br), ("TL:", self.var_z_tl), ("TR:", self.var_z_tr)]):
            ttk.Label(lvl_frame, text=lbl, style="Card.TLabel").grid(row=1, column=idx * 2, sticky="w", padx=(2, 1))
            tk.Entry(lvl_frame, textvariable=var, width=5).grid(row=1, column=idx * 2 + 1, padx=(0, 4))

        chk_row = ttk.Frame(card, style="Card.TFrame")
        chk_row.pack(fill="x", pady=4)
        ttk.Checkbutton(chk_row, text="Capa Bottom (Espejar THT)", variable=self.var_mirror).pack(side="left", padx=(0, 8))
        ttk.Checkbutton(chk_row, text="Invertir B/N", variable=self.var_invert).pack(side="left")

        tk.Button(card, text="Generar Pistas + Vista Previa + G-Code", command=self._generate_gcode, bg="#8b5cf6", fg="#fff", font=("Segoe UI", 9, "bold"), relief="flat", pady=6).pack(fill="x", pady=4)

        run_row = ttk.Frame(card, style="Card.TFrame")
        run_row.pack(fill="x", pady=3)
        self.btn_run = tk.Button(run_row, text="EJECUTAR EN MAGNA SE", command=self._start_job, bg="#10b981", fg="#fff", font=("Segoe UI", 9, "bold"), relief="flat", pady=6)
        self.btn_run.pack(side="left", fill="x", expand=True, padx=(0, 3))
        self.btn_pause = tk.Button(run_row, text="Pausar / Subir Z", command=self._pause_job, bg="#eab308", fg="#111", font=("Segoe UI", 9, "bold"), relief="flat", pady=6)
        self.btn_pause.pack(side="left", fill="x", expand=True, padx=(3, 0))

        self.prog_var = tk.DoubleVar(value=0.0)
        ttk.Progressbar(card, variable=self.prog_var, maximum=100.0).pack(fill="x", pady=(4, 0))

    def _build_preview_and_log(self, parent: ttk.Frame):
        prev_card = ttk.Frame(parent, style="Card.TFrame", padding=8)
        prev_card.pack(fill="both", expand=True, pady=(0, 10))

        self.stats_var = tk.StringVar(value="Listo para generar pistas de circuito eléctrico.")
        ttk.Label(prev_card, textvariable=self.stats_var, style="Card.TLabel", foreground="#fcd34d").pack(anchor="w", pady=(0, 6))

        self.preview_label = tk.Label(prev_card, bg="#14161a", text="Simulación de Placa PCB y Rutas CNC", fg="#7f8794")
        self.preview_label.pack(fill="both", expand=True)

        log_card = ttk.Frame(parent, style="Card.TFrame", padding=8, height=195)
        log_card.pack(fill="x")
        log_card.pack_propagate(False)

        cmd_row = ttk.Frame(log_card, style="Card.TFrame")
        cmd_row.pack(fill="x", pady=(0, 4))
        self.manual_cmd = tk.Entry(cmd_row, bg="#14161a", fg="#e6eaf2", insertbackground="#fff")
        self.manual_cmd.pack(side="left", fill="x", expand=True, padx=(0, 6))
        self.manual_cmd.bind("<Return>", lambda _e: self._send_manual_cmd())
        tk.Button(cmd_row, text="Enviar G-Code", command=self._send_manual_cmd, bg="#374151", fg="#fff", relief="flat", padx=10).pack(side="left")

        self.log_box = tk.Text(log_card, height=8, bg="#111318", fg="#b8c4d9", font=("Consolas", 9), relief="flat")
        self.log_box.pack(fill="both", expand=True)

    def _on_mode_preset_changed(self, _event=None):
        m = self.var_mode.get()
        if m == "pcb_laser_ablation":
            self.var_tool.set(0.15)
            self.var_iso_rings.set(3)
            self.var_depth.set(0.00)
            self.var_stepdown.set(0.08)
            self.var_cutfeed.set(480.0)
        elif m == "pcb_isolation":
            self.var_tool.set(0.25)
            self.var_iso_rings.set(2)
            self.var_depth.set(0.08)
            self.var_stepdown.set(0.08)
            self.var_cutfeed.set(300.0)
        elif m == "pcb_marker_resist":
            self.var_tool.set(0.75)
            self.var_iso_rings.set(2)
            self.var_depth.set(0.45)
            self.var_stepdown.set(0.45)
            self.var_cutfeed.set(360.0)
        elif m == "pcb_drill_holes":
            self.var_tool.set(0.85)
            self.var_depth.set(2.00)
            self.var_stepdown.set(1.00)
            self.var_cutfeed.set(180.0)
        self._generate_gcode()

    def _run_camera_aoi(self):
        if not self.current_img_path:
            messagebox.showinfo("Sin circuito", "Cargá un circuito primero.")
            return
        from pcb_engineering_engine import inspect_pcb_with_camera_aoi
        if Path(self.current_img_path).suffix.lower() == ".svg":
            ref_gray, _, _ = render_svg_to_gray_mm(self.current_img_path, px_per_mm=20.0)
        else:
            ref_gray = cv2.imread(self.current_img_path, cv2.IMREAD_GRAYSCALE)
        if ref_gray is None:
            return
        cap = cv2.VideoCapture(0)
        ok, frame = (cap.read() if cap.isOpened() else (False, None))
        if cap.isOpened():
            cap.release()
        if not ok or frame is None:
            # Si la cámara está ocupada o en prueba offline, simular captura sobre la referencia para validar el pipeline AOI
            frame = cv2.cvtColor(cv2.flip(ref_gray, 1) if self.var_mirror.get() else ref_gray, cv2.COLOR_GRAY2BGR)
            self._append_log("[AOI] Usando imagen de referencia en memoria (cámara 0 no disponible).")
        overlay, res = inspect_pcb_with_camera_aoi(ref_gray, frame, mirror_expected=bool(self.var_mirror.get()))
        aoi_path = WORK_DIR / "aoi_camera_inspection.png"
        cv2.imwrite(str(aoi_path), overlay)
        self._append_log(
            f"[CAMARA AOI] Coincidencia={res['match_pct']:.2f}% | Defectos={res['defects_count']} | "
            f"Resultado={'APROBADO' if res['passed'] else 'REVISAR PUENTES/CORTES'}"
        )
        ph, pw = overlay.shape[:2]
        scale = min(780.0 / pw, 480.0 / ph, 1.0)
        if scale < 1.0:
            overlay = cv2.resize(overlay, (int(pw * scale), int(ph * scale)), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(aoi_path), overlay)
        self._preview_tk = tk.PhotoImage(file=str(aoi_path))
        self.preview_label.config(image=self._preview_tk, text="")

    def _append_log_threadsafe(self, text: str):
        self.root.after(0, lambda: self._append_log(text))

    def _append_log(self, text: str):
        self.log_box.insert("end", text + "\n")
        self.log_box.see("end")

    def _refresh_ports(self):
        ports = find_printer_ports()
        vals = [p[0] for p in ports]
        self.port_combo["values"] = vals
        if "COM4" in vals:
            self.port_var.set("COM4")
        elif vals:
            self.port_var.set(vals[0])

    def _toggle_connect(self):
        if self.ctrl.telemetry.connected:
            self.ctrl.disconnect()
            self.btn_conn.config(text="Conectar (115200)", bg="#198754")
            self.status_var.set("Desconectada")
            return
        port = self.port_var.get().strip()
        if not port:
            return
        try:
            tel = self.ctrl.connect(port=port, baud=115200, reset_board=False)
            self.btn_conn.config(text="Desconectar", bg="#6b7280")
            self._update_status_label(tel)
        except Exception as exc:
            messagebox.showerror("Error de conexión", str(exc))

    def _update_status_label(self, tel):
        self.status_var.set(
            f"{tel.port} OK | X:{tel.x:.2f} Y:{tel.y:.2f} Z:{tel.z:.2f} | Hotend:{tel.hotend_temp:.1f}°C Cama:{tel.bed_temp:.1f}°C"
        )

    def _jog(self, axis: str, sign: int):
        if not self.ctrl.telemetry.connected:
            messagebox.showwarning("Sin conexión", "Conectá la Magna SE Mini primero.")
            return
        delta = sign * float(self.step_var.get())
        threading.Thread(target=lambda: self._run_tel_action(lambda: self.ctrl.jog(axis, delta)), daemon=True).start()

    def _home_xy(self):
        if not self.ctrl.telemetry.connected:
            return
        threading.Thread(target=lambda: self._run_tel_action(self.ctrl.home_xy_safe), daemon=True).start()

    def _set_zero(self):
        if not self.ctrl.telemetry.connected:
            return
        threading.Thread(target=lambda: self._run_tel_action(self.ctrl.set_work_zero), daemon=True).start()

    def _trace_box(self):
        if not self.ctrl.telemetry.connected:
            messagebox.showwarning("Sin conexión", "Conectá la Magna SE Mini primero.")
            return
        w = self.last_stats.width_mm if self.last_stats else float(self.var_width.get())
        h = self.last_stats.height_mm if self.last_stats else w
        threading.Thread(target=lambda: self._run_tel_action(lambda: self.ctrl.trace_air_box(w, h)), daemon=True).start()

    def _run_tel_action(self, fn):
        try:
            tel = fn()
            if tel:
                self.root.after(0, lambda: self._update_status_label(tel))
        except Exception as exc:
            self._append_log_threadsafe(f"[ERR] {exc}")

    def _emergency_stop(self):
        self.ctrl.emergency_stop()
        messagebox.showwarning("Parada de Emergencia", "Enviado M410 (Quickstop) + M107 + motores liberados.")

    def _send_manual_cmd(self):
        cmd = self.manual_cmd.get().strip()
        if not cmd or not self.ctrl.telemetry.connected:
            return
        self.manual_cmd.delete(0, "end")

        def work():
            resp = self.ctrl.send_and_wait(cmd, timeout=10.0)
            self._append_log_threadsafe(f"> {cmd} -> {' | '.join(resp)}")
            tel = self.ctrl.refresh_status()
            self.root.after(0, lambda: self._update_status_label(tel))

        threading.Thread(target=work, daemon=True).start()

    def _pick_image(self):
        path = filedialog.askopenfilename(filetypes=[("Circuitos PCB (SVG/PNG/JPG)", "*.svg *.png *.jpg *.jpeg *.bmp")])
        if path:
            self.current_img_path = path
            p_obj = Path(path)
            self.img_name_var.set(p_obj.name)
            if p_obj.suffix.lower() == ".svg":
                try:
                    svg_w, svg_h, already_mirrored = get_svg_info(p_obj)
                    self.var_width.set(round(svg_w, 2))
                    self._append_log(f"[SVG 1:1] {p_obj.name} -> {svg_w:.2f} x {svg_h:.2f} mm")
                    if already_mirrored and self.var_mirror.get():
                        self.var_mirror.set(False)
                        self._append_log("[AVISO] El SVG ya está espejado (THT): se destildó 'Capa Bottom (Espejar THT)' para evitar doble espejado.")
                except Exception as exc:
                    self._append_log(f"[AVISO SVG] No se pudo leer escala 1:1 de {p_obj.name}: {exc}")
            self._generate_gcode()

    def _load_tank_pcb_template(self):
        pcb_png = WORK_DIR / "pcb_tanque_esp32s3_carrier.png"
        _, w_mm, _ = create_tank_carrier_pcb_layout(pcb_png)
        self.current_img_path = str(pcb_png)
        self.var_width.set(w_mm)
        self.img_name_var.set(f"{pcb_png.name} (2x ESP32-S3 + Arduino + Divisor 5V->3.3V)")
        audit_txt = pcb_png.with_suffix(".audit.txt")
        if audit_txt.exists():
            for line in audit_txt.read_text(encoding="utf-8").splitlines():
                self._append_log(line)
        self._generate_gcode()

    def _generate_gcode(self):
        if not self.current_img_path:
            return
        cfg = XiloConfig(
            width_mm=float(self.var_width.get()),
            mode=self.var_mode.get(),
            mirror_x=bool(self.var_mirror.get()),
            invert_bw=bool(self.var_invert.get()),
            threshold=int(self.var_thresh.get()),
            tool_diameter_mm=float(self.var_tool.get()),
            isolation_passes=int(self.var_iso_rings.get()),
            total_depth_mm=float(self.var_depth.get()),
            z_stepdown_mm=float(self.var_stepdown.get()),
            xy_cut_feed=float(self.var_cutfeed.get()),
            z_corner_bl=float(self.var_z_bl.get()),
            z_corner_br=float(self.var_z_br.get()),
            z_corner_tl=float(self.var_z_tl.get()),
            z_corner_tr=float(self.var_z_tr.get()),
        )
        stem = Path(self.current_img_path).stem
        out_gc = WORK_DIR / f"{stem}_{cfg.mode}.gcode"
        out_prev = WORK_DIR / f"{stem}_{cfg.mode}_preview.png"
        stats = generate_woodcut_gcode(self.current_img_path, out_gc, cfg, out_prev)
        self.last_stats = stats

        self.stats_var.set(
            f"Placa: {stats.width_mm:.1f}x{stats.height_mm:.1f} mm | Agujeros 2.54mm: {stats.holes_detected} | "
            f"Pasadas Z: {stats.z_passes} | Líneas: {stats.total_lines} | "
            f"Recorrido: {stats.cut_distance_mm/1000:.2f} m | Tiempo est.: ~{stats.estimated_minutes:.1f} min"
        )
        self._append_log(f"[PCB GCODE] {out_gc.name} ({stats.total_lines} líneas, {stats.holes_detected} pads detectados)")

        prev_bgr = cv2.imread(str(out_prev))
        if prev_bgr is not None:
            ph, pw = prev_bgr.shape[:2]
            scale = min(780.0 / pw, 480.0 / ph, 1.0)
            if scale < 1.0:
                resized = cv2.resize(prev_bgr, (int(pw * scale), int(ph * scale)), interpolation=cv2.INTER_AREA)
                scaled_path = WORK_DIR / f"{stem}_preview_ui.png"
                cv2.imwrite(str(scaled_path), resized)
                self._preview_tk = tk.PhotoImage(file=str(scaled_path))
            else:
                self._preview_tk = tk.PhotoImage(file=str(out_prev))
            self.preview_label.config(image=self._preview_tk, text="")

    def _start_job(self):
        if not self.ctrl.telemetry.connected:
            messagebox.showwarning("Sin conexión", "Conectá la Magna SE Mini primero.")
            return
        if not self.last_stats or not self.last_stats.gcode_path:
            messagebox.showwarning("Sin G-Code", "Generá la vista previa y el G-Code primero.")
            return
        if not messagebox.askyesno(
            "Confirmar Grabado de Circuito (Sin Láser)",
            "¿Ya fijaste la placa virgen de cobre (1.6 mm) a la cama y calibraste CERO COBRE (G92 X0 Y0 Z0) apoyando la punta SOBRE LA CARA SUPERIOR DEL COBRE (no sobre el vidrio) en la esquina inferior izquierda?",
        ):
            return
        self.prog_var.set(0.0)
        self.ctrl.start_gcode_file(
            self.last_stats.gcode_path,
            progress_cb=lambda cur, tot, _c: self.root.after(0, lambda: self._set_prog(cur, tot)),
            done_cb=lambda ok: self._append_log_threadsafe("[PCB] Trabajo completado OK!" if ok else "[PCB] Trabajo detenido."),
        )

    def _set_prog(self, cur: int, tot: int):
        self.prog_var.set(100.0 * cur / max(1, tot))

    def _pause_job(self):
        if self.ctrl.job_running:
            paused = self.ctrl.toggle_pause()
            self.btn_pause.config(text="Reanudar" if paused else "Pausar / Subir Z")


def main():
    root = tk.Tk()
    MagnaStudioApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
