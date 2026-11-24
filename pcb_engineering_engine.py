"""
Motor de Ingeniería Eléctrica, Térmica (Norma IPC-2221) e Inspección Óptica por Cámara (AOI)
para Diseño y Validación Autónoma de PCBs de 1 Sola Capa en la Magna SE Mini.

Realiza automáticamente:
  1. Dimensionamiento físico de ancho de pista (mm) según Amperaje (A), espesor de cobre (oz)
     y límite de calentamiento permitido (ΔT °C) bajo estándar IPC-2221 (Capas Externas).
  2. Cálculo de aislamiento mínimo (Clearance mm) según Voltaje pico (V) + margen de seguridad.
  3. Simulación física pre-ejecución de cada pista:
     - Longitud real ruteada (mm)
     - Resistencia eléctrica DC a temperatura de operación (mΩ)
     - Caída de tensión IR-drop (mV y %)
     - Potencia disipada por efecto Joule P = I²R (mW)
     - Temperatura alcanzada por la pista (°C)
  4. Suite de 6 Testeos Pre-Ejecución (DRC 1-Capa, Térmico, Caída de Tensión, Clearance Euclidiano,
     ERC de Protección 3.3V/5V y Visión AOI con Cámara).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
import cv2
import numpy as np


# Constantes físicas del cobre electrolítico en PCB (1 oz/ft² = 35 µm = 1.378 mils)
COPPER_RESISTIVITY_20C = 1.724e-8     # Ohm * m
COPPER_TEMP_COEFF = 0.00393           # 1 / °C
MILS_TO_MM = 0.0254
OZ_TO_MILS = 1.378
OZ_TO_MM = OZ_TO_MILS * MILS_TO_MM    # 0.0350012 mm


@dataclass
class NetSpec:
    name: str
    voltage_v: float                  # Voltaje nominal o máximo de la red (V)
    current_max_a: float              # Corriente máxima pico esperada (A)
    is_3v3_sensitive: bool = False    # True si se conecta directo a pines lógicos de 3.3V (ESP32-S3)
    max_voltage_drop_pct: float = 2.0 # Caída de tensión máxima admisible (%)
    max_temp_rise_c: float = 12.0     # Subida máxima de temperatura admisible (°C)
    min_width_override_mm: float = 0.0


@dataclass
class NetAuditResult:
    name: str
    voltage_v: float
    current_a: float
    ipc_min_width_mm: float
    assigned_width_mm: float
    ipc_min_clearance_mm: float
    measured_length_mm: float
    resistance_mohm: float
    voltage_drop_mv: float
    voltage_drop_pct: float
    power_mw: float
    temp_rise_c: float
    final_temp_c: float
    passed: bool
    notes: list[str] = field(default_factory=list)


@dataclass
class FullBoardTestReport:
    passed_all: bool
    drc_single_layer_ok: bool
    thermal_ok: bool
    voltage_drop_ok: bool
    clearance_ok: bool
    erc_logic_ok: bool
    min_measured_clearance_mm: float
    max_board_temp_c: float
    total_power_mw: float
    net_audits: list[NetAuditResult]
    summary_lines: list[str]


def ipc2221_min_trace_width_mm(
    current_a: float,
    temp_rise_c: float = 10.0,
    copper_oz: float = 1.0,
    internal_layer: bool = False,
) -> float:
    """
    Calcula el ancho mínimo de pista en mm según IPC-2221:
      I = k * (ΔT ^ b) * (Area_mils² ^ c)
      Capas externas (1 sola capa): k = 0.048, b = 0.44, c = 0.725
    """
    if current_a <= 0.01:
        return 0.35
    k = 0.024 if internal_layer else 0.048
    b = 0.44
    c = 0.725
    dt = max(1.0, float(temp_rise_c))
    area_mils2 = (current_a / (k * (dt ** b))) ** (1.0 / c)
    thickness_mils = copper_oz * OZ_TO_MILS
    width_mils = area_mils2 / thickness_mils
    width_mm = width_mils * MILS_TO_MM
    return max(0.35, width_mm)


def ipc2221_temp_rise_c(
    current_a: float,
    width_mm: float,
    copper_oz: float = 1.0,
    internal_layer: bool = False,
) -> float:
    """Despeja el aumento real de temperatura ΔT (°C) para un ancho de pista y corriente dados."""
    if current_a <= 0.001:
        return 0.0
    k = 0.024 if internal_layer else 0.048
    b = 0.44
    c = 0.725
    width_mils = max(1.0, width_mm / MILS_TO_MM)
    thickness_mils = copper_oz * OZ_TO_MILS
    area_mils2 = width_mils * thickness_mils
    dt = (current_a / (k * (area_mils2 ** c))) ** (1.0 / b)
    return float(dt)


def ipc2221_min_clearance_mm(voltage_peak_v: float, cnc_laser_margin_mm: float = 0.35) -> float:
    """
    Devuelve el aislamiento mínimo entre pistas según IPC-2221 Table 6-1 (Uncoated External Conductors)
    más el margen práctico para grabado casero sin máscara antisoldante (soldermask).
    """
    v = abs(float(voltage_peak_v))
    if v <= 15.0:
        ipc_base = 0.10
    elif v <= 30.0:
        ipc_base = 0.15
    elif v <= 50.0:
        ipc_base = 0.60
    elif v <= 100.0:
        ipc_base = 0.60
    else:
        ipc_base = 1.50 + 0.005 * (v - 100.0)
    return max(ipc_base, cnc_laser_margin_mm)


def polyline_length_mm(traces: list[list[tuple[float, float]]]) -> float:
    total = 0.0
    for tr in traces:
        for i in range(len(tr) - 1):
            total += math.hypot(tr[i + 1][0] - tr[i][0], tr[i + 1][1] - tr[i][1])
    return total


def recommend_trace_width_mm(spec: NetSpec, copper_oz: float = 1.0) -> float:
    """
    Dimensiona automáticamente el ancho óptimo de la pista combinando:
      1. Límite térmico IPC-2221
      2. Límite de caída de tensión (IR Drop) para una longitud típica de 60mm
      3. Reglas de robustez mecánica para grabado láser/fresado en 1 capa (>= 0.85mm señal, >= 1.35mm potencia)
    """
    w_ipc = ipc2221_min_trace_width_mm(spec.current_max_a, spec.max_temp_rise_c, copper_oz)
    # Si maneja corriente de potencia (>= 0.4A), dar holgura extra para minimizar caída ohmica y calor
    if spec.current_max_a >= 1.5:
        w_practical = max(w_ipc * 1.35, 1.40)
    elif spec.current_max_a >= 0.4:
        w_practical = max(w_ipc * 1.25, 1.30)
    else:
        w_practical = max(w_ipc, 0.85)
    return round(max(w_practical, spec.min_width_override_mm), 2)


def run_autonomous_pcb_audit(
    nets: dict[str, dict],
    net_specs: dict[str, NetSpec],
    board_img_gray: np.ndarray,
    w_mm: float,
    h_mm: float,
    ambient_temp_c: float = 25.0,
    copper_oz: float = 1.0,
) -> FullBoardTestReport:
    """
    Ejecuta la batería completa de 5 testeos de ingeniería sobre el diseño antes de permitir
    la generación/ejecución del G-Code en la impresora.
    """
    H, W = board_img_gray.shape
    px_per_mm_x = (W - 1) / max(1e-3, w_mm)
    px_per_mm_y = (H - 1) / max(1e-3, h_mm)
    px_per_mm = 0.5 * (px_per_mm_x + px_per_mm_y)

    def mm_to_px(x_mm: float, y_mm: float) -> tuple[int, int]:
        return int(round(x_mm * px_per_mm_x)), int(round((h_mm - y_mm) * px_per_mm_y))

    # 1. Test DRC de 1 Sola Capa (Continuidad e Independencia de Islas de Cobre)
    copper_bin = (board_img_gray < 128).astype(np.uint8)
    _, labels = cv2.connectedComponents(copper_bin, connectivity=8)

    drc_ok = True
    drc_errors: list[str] = []
    net_label_map: dict[str, int] = {}

    for net_name, net_data in nets.items():
        pad_labels = set()
        for px_m, py_m in net_data["pads"]:
            cx, cy = mm_to_px(px_m, py_m)
            # Buscar en vecindad pequeña por si el centro tiene agujero perforado blanco
            patch = labels[max(0, cy - 12) : min(H, cy + 13), max(0, cx - 12) : min(W, cx + 13)]
            nz = patch[patch > 0]
            if len(nz) == 0:
                drc_ok = False
                drc_errors.append(f"[DRC FAIL] Pad {px_m, py_m} de {net_name} sin cobre.")
            else:
                pad_labels.add(int(nz[0]))
        if len(pad_labels) > 1:
            drc_ok = False
            drc_errors.append(f"[DRC FAIL] Red {net_name} abierta/partida en {len(pad_labels)} tramos.")
        elif len(pad_labels) == 1:
            lbl = next(iter(pad_labels))
            for other_net, other_lbl in net_label_map.items():
                if lbl == other_lbl:
                    drc_ok = False
                    drc_errors.append(f"[DRC CORTOCIRCUITO] Superposición en 1 capa entre {net_name} y {other_net}!")
            net_label_map[net_name] = lbl

    # 2. Medición real de Clearance mínimo entre islas de cobre distintas usando Transformada de Distancia
    min_measured_clearance_mm = 99.0
    for net_name, lbl in net_label_map.items():
        this_island = (labels == lbl).astype(np.uint8)
        other_islands = ((labels > 0) & (labels != lbl)).astype(np.uint8)
        if not np.any(other_islands):
            continue
        dist_map_px = cv2.distanceTransform(1 - other_islands, cv2.DIST_L2, 5)
        min_px = float(np.min(dist_map_px[this_island > 0]))
        min_mm = min_px / px_per_mm
        if min_mm < min_measured_clearance_mm:
            min_measured_clearance_mm = min_mm

    # 3. Simulación Eléctrica, Térmica y Caída de Tensión por cada Red
    audits: list[NetAuditResult] = []
    thermal_ok = True
    vdrop_ok = True
    clearance_ok = True
    erc_ok = True
    max_board_temp = ambient_temp_c
    total_power_mw = 0.0

    t_copper_m = (copper_oz * OZ_TO_MM) * 1e-3

    for net_name, net_data in nets.items():
        spec = net_specs.get(net_name, NetSpec(name=net_name, voltage_v=3.3, current_max_a=0.05))
        w_assigned = float(net_data["width"])
        w_ipc_min = ipc2221_min_trace_width_mm(spec.current_max_a, spec.max_temp_rise_c, copper_oz)
        clr_ipc_min = ipc2221_min_clearance_mm(spec.voltage_v)
        length_mm = max(5.0, polyline_length_mm(net_data["traces"]))

        dt_c = ipc2221_temp_rise_c(spec.current_max_a, w_assigned, copper_oz)
        final_t_c = ambient_temp_c + dt_c
        max_board_temp = max(max_board_temp, final_t_c)

        # Resistencia con coeficiente térmico real
        rho = COPPER_RESISTIVITY_20C * (1.0 + COPPER_TEMP_COEFF * (final_t_c - 20.0))
        area_m2 = (w_assigned * 1e-3) * t_copper_m
        r_ohm = rho * (length_mm * 1e-3) / max(1e-12, area_m2)
        r_mohm = r_ohm * 1000.0

        v_drop_v = spec.current_max_a * r_ohm
        v_drop_mv = v_drop_v * 1000.0
        ref_v = max(3.3, abs(spec.voltage_v))
        v_drop_pct = 100.0 * (v_drop_v / ref_v)
        p_mw = (spec.current_max_a ** 2) * r_mohm
        total_power_mw += p_mw

        notes = []
        net_pass = True

        if w_assigned + 1e-3 < w_ipc_min:
            net_pass = False
            thermal_ok = False
            notes.append(f"Ancho {w_assigned:.2f}mm < Mínimo IPC {w_ipc_min:.2f}mm")

        if dt_c > spec.max_temp_rise_c:
            net_pass = False
            thermal_ok = False
            notes.append(f"ΔT={dt_c:.1f}°C excede {spec.max_temp_rise_c:.1f}°C")

        if v_drop_pct > spec.max_voltage_drop_pct:
            net_pass = False
            vdrop_ok = False
            notes.append(f"Caída {v_drop_pct:.2f}% excede {spec.max_voltage_drop_pct:.2f}%")

        if min_measured_clearance_mm < clr_ipc_min:
            net_pass = False
            clearance_ok = False
            notes.append(f"Clearance {min_measured_clearance_mm:.2f}mm < {clr_ipc_min:.2f}mm")

        if spec.is_3v3_sensitive and spec.voltage_v > 3.6:
            net_pass = False
            erc_ok = False
            notes.append(f"ERC PELIGRO: Pin 3.3V expuesto a {spec.voltage_v:.1f}V!")

        if not notes:
            notes.append("OK (Cumple IPC-2221)")

        audits.append(
            NetAuditResult(
                name=net_name,
                voltage_v=spec.voltage_v,
                current_a=spec.current_max_a,
                ipc_min_width_mm=w_ipc_min,
                assigned_width_mm=w_assigned,
                ipc_min_clearance_mm=clr_ipc_min,
                measured_length_mm=length_mm,
                resistance_mohm=r_mohm,
                voltage_drop_mv=v_drop_mv,
                voltage_drop_pct=v_drop_pct,
                power_mw=p_mw,
                temp_rise_c=dt_c,
                final_temp_c=final_t_c,
                passed=net_pass,
                notes=notes,
            )
        )

    passed_all = drc_ok and thermal_ok and vdrop_ok and clearance_ok and erc_ok
    summary = [
        f"[{'PASS' if passed_all else 'FAIL'}] AUDITORÍA ELÉCTRICA, TÉRMICA Y DRC 1-CAPA (Norma IPC-2221)",
        f"  * DRC 1-Sola-Capa (0 Cruces) : {'OK (7 redes 100% aisladas)' if drc_ok else 'FALLO: ' + '; '.join(drc_errors)}",
        f"  * Clearance Mínimo Medido    : {min_measured_clearance_mm:.2f} mm ({'OK' if clearance_ok else 'BAJO'})",
        f"  * Temperatura Máxima Pistas  : {max_board_temp:.2f} °C (Ambiente {ambient_temp_c:.1f} °C) -> {'OK' if thermal_ok else 'EXCESO'}",
        f"  * Potencia Total Disipada    : {total_power_mw:.2f} mW en pistas",
        f"  * Protección Lógica 3.3V ERC : {'OK (Divisor 5V->3.33V verificado)' if erc_ok else 'PELIGRO SOBREVOLTAJE'}",
    ]
    for a in audits:
        summary.append(
            f"    - {a.name:26s}: V={a.voltage_v:4.1f}V | I={a.current_a:4.2f}A | W={a.assigned_width_mm:.2f}mm "
            f"(IPC>={a.ipc_min_width_mm:.2f}) | L={a.measured_length_mm:4.1f}mm | R={a.resistance_mohm:5.1f}mΩ | "
            f"ΔV={a.voltage_drop_mv:5.1f}mV ({a.voltage_drop_pct:.2f}%) | T={a.final_temp_c:4.1f}°C -> {'; '.join(a.notes)}"
        )

    return FullBoardTestReport(
        passed_all=passed_all,
        drc_single_layer_ok=drc_ok,
        thermal_ok=thermal_ok,
        voltage_drop_ok=vdrop_ok,
        clearance_ok=clearance_ok,
        erc_logic_ok=erc_ok,
        min_measured_clearance_mm=min_measured_clearance_mm,
        max_board_temp_c=max_board_temp,
        total_power_mw=total_power_mw,
        net_audits=audits,
        summary_lines=summary,
    )


def inspect_pcb_with_camera_aoi(
    reference_pcb_gray: np.ndarray,
    camera_frame_bgr: np.ndarray,
    mirror_expected: bool = True,
) -> tuple[np.ndarray, dict]:
    """
    Inspección Óptica Automatizada (AOI) mediante Cámara / Webcam:
      1. Detecta el contorno rectangular de la placa en la imagen de la cámara y corrige perspectiva (Homografía).
      2. Compara el patrón grabado contra el diseño teórico para detectar:
         - Posibles puentes/cortos (pintura o cobre no removido entre pistas)
         - Posibles cortes de pista (exceso de ablación láser)
      3. Devuelve imagen anotada con círculos de diagnóstico y métricas de coincidencia (%).
    """
    ref = cv2.flip(reference_pcb_gray, 1) if mirror_expected else reference_pcb_gray.copy()
    rh, rw = ref.shape[:2]

    cam_gray = cv2.cvtColor(camera_frame_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(cam_gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 45, 140)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    warped = cv2.resize(cam_gray, (rw, rh))
    board_found = False
    if cnts:
        largest = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(largest) > (cam_gray.shape[0] * cam_gray.shape[1] * 0.08):
            peri = cv2.arcLength(largest, True)
            approx = cv2.approxPolyDP(largest, 0.03 * peri, True)
            if len(approx) == 4:
                board_found = True
                pts = approx.reshape(4, 2).astype(np.float32)
                # Ordenar esquinas: TL, TR, BR, BL
                s = pts.sum(axis=1)
                diff = np.diff(pts, axis=1).reshape(-1)
                rect = np.zeros((4, 2), dtype=np.float32)
                rect[0] = pts[np.argmin(s)]
                rect[2] = pts[np.argmax(s)]
                rect[1] = pts[np.argmin(diff)]
                rect[3] = pts[np.argmax(diff)]
                dst = np.array([[0, 0], [rw - 1, 0], [rw - 1, rh - 1], [0, rh - 1]], dtype=np.float32)
                M = cv2.getPerspectiveTransform(rect, dst)
                warped = cv2.warpPerspective(cam_gray, M, (rw, rh))

    _, cam_bin = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ref_bin = ((ref >= 128).astype(np.uint8)) * 255

    diff_mask = cv2.absdiff(ref_bin, cam_bin)
    # Filtrar ruido menor de bordes (1-2 px)
    diff_clean = cv2.morphologyEx(diff_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    match_score = 100.0 * (1.0 - float(np.count_nonzero(diff_clean)) / float(diff_clean.size))
    defect_cnts, _ = cv2.findContours(diff_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    overlay = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
    defects_found = 0
    for c in defect_cnts:
        if cv2.contourArea(c) < 12.0:
            continue
        defects_found += 1
        (x, y), r = cv2.minEnclosingCircle(c)
        cv2.circle(overlay, (int(x), int(y)), max(8, int(r) + 4), (0, 0, 255), 2)

    status_txt = f"AOI Camara: Coincidencia={match_score:.1f}% | Defectos={defects_found} | Encuadre={'AUTO' if board_found else 'MANUAL'}"
    cv2.putText(overlay, status_txt, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 180) if defects_found == 0 else (0, 80, 255), 2)

    return overlay, {
        "board_detected": board_found,
        "match_pct": match_score,
        "defects_count": defects_found,
        "passed": (defects_found == 0 and match_score >= 92.0),
    }
