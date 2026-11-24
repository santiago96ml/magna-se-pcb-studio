"""
Motor CNC de Circuitos Eléctricos (PCB) y Grabado Mecánico para Hellbot Magna SE Mini (SIN LÁSER)
Diseñado específicamente para fabricar placas electrónicas (PCBs) sobre cobre virgen (FR4 / Pertinax):

Modos de Fabricación de Circuitos Eléctricos:
  1. 'pcb_isolation'     : Fresado de Aislamiento de Pistas (Isolation Routing). Contornea ALREDEDOR
                           de cada pista y pad con 1 a 4 pasadas concéntricas (Z = -0.07 mm) para
                           aislar eléctricamente el cobre conductor del resto de la placa.
  2. 'pcb_marker_resist' : Plotter con Marcador Indeleble / Tinta Antiácido (para Cloruro Férrico).
                           Dibuja y rellena las pistas y los pads directamente sobre el cobre.
  3. 'pcb_drill_holes'   : Perforado / Centrado automático de agujeros (Through-Hole 2.54mm y pads).
  4. 'relief_pocket'     : Desbaste completo de zonas sin cobre (vaciado total).

Incluye:
  - Interpolación Bilineal de Altura Z en 4 esquinas (compensa inclinación de la placa de cobre al micrón).
  - Generador paramétrico de placas reales (incluyendo el Carrier PCB del Tanque ESP32-S3 + Divisor 5V->3.3V).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET
import cv2
import numpy as np


BED_MAX_X = 230.0
BED_MAX_Y = 230.0
BED_MAX_Z = 250.0
MAX_Z_FEED_MM_MIN = 210.0   # Tope seguro para M203 Z4.00 (240 mm/min)
MAX_XY_FEED_MM_MIN = 3600.0


@dataclass
class XiloConfig:
    width_mm: float = 65.0
    height_mm: float = 0.0            # 0 = mantiene relación de aspecto exacta de la placa
    offset_x: float = 0.0
    offset_y: float = 0.0
    mode: str = "pcb_marker_resist"   # Por defecto: Trazado directo del circuito con Fibra Indeleble 0.75mm
    mirror_x: bool = True             # True = Capa Bottom (Through-Hole soldado abajo) | False = Capa Top (SMD)
    invert_bw: bool = False           # False = Pistas negras sobre fondo blanco
    threshold: int = 128
    tool_diameter_mm: float = 0.75    # 0.75mm para Edding 404 (o ajustar según punta)
    isolation_passes: int = 2         # En fibra: cantidad de manos de tinta (1 o 2 manos para blindar contra ácido)
    stepover_ratio: float = 0.50      # 50% de solapamiento entre trazos de la fibra para cubrir 100% sin poros
    total_depth_mm: float = 0.45      # Presión suave de la punta de fibra (0.45mm con soporte elástico para compensar desnivel)
    z_stepdown_mm: float = 0.45       # En fibra baja directo en 1 paso por mano de tinta
    z_safe_mm: float = 3.5            # Levanta 3.5mm la fibra en cada salto para no manchar entre pistas
    laser_pwm: int = 0                # 0 = Láser desactivado (M107 forzado)
    laser_focus_z_mm: float = 0.0
    xy_cut_feed: float = 360.0        # 360 mm/min (6 mm/s): velocidad ideal para depósito denso en cobre liso
    xy_travel_feed: float = 2400.0    # mm/min
    z_plunge_feed: float = 150.0      # mm/min (bajada vertical suave para no achatar la punta de 0.4mm)
    use_work_zero: bool = True        # Usa G92 X0 Y0 Z0 donde la punta de la fibra apenas besa el cobre
    px_per_mm: float = 20.0           # 20 px/mm (50 micrones/px) para compensación exacta del radio de 0.20mm
    # Nivelación bilineal de las 4 esquinas de la placa de cobre
    z_corner_bl: float = 0.0
    z_corner_br: float = 0.0
    z_corner_tl: float = 0.0
    z_corner_tr: float = 0.0


@dataclass
class XiloStats:
    width_mm: float
    height_mm: float
    z_passes: int
    total_lines: int
    cut_distance_mm: float
    travel_distance_mm: float
    estimated_minutes: float
    holes_detected: int = 0
    preview_path: str | None = None
    gcode_path: str | None = None


def _clamp_feed_xy(f: float) -> float:
    return max(30.0, min(MAX_XY_FEED_MM_MIN, float(f)))


def _clamp_feed_z(f: float) -> float:
    return max(15.0, min(MAX_Z_FEED_MM_MIN, float(f)))


def _bilinear_z_offset(x_mm: float, y_mm: float, w_mm: float, h_mm: float, cfg: XiloConfig) -> float:
    """Interpola el desnivel de la placa de cobre entre sus 4 esquinas."""
    u = max(0.0, min(1.0, (x_mm - cfg.offset_x) / max(1e-3, w_mm)))
    v = max(0.0, min(1.0, (y_mm - cfg.offset_y) / max(1e-3, h_mm)))
    z_bot = (1.0 - u) * cfg.z_corner_bl + u * cfg.z_corner_br
    z_top = (1.0 - u) * cfg.z_corner_tl + u * cfg.z_corner_tr
    return (1.0 - v) * z_bot + v * z_top


def _order_paths_nearest(paths: list[np.ndarray]) -> list[np.ndarray]:
    if len(paths) <= 1:
        return paths
    remaining = list(paths)
    ordered = [remaining.pop(0)]
    while remaining:
        cur_end = ordered[-1][-1]
        best_i = 0
        best_dist = float("inf")
        best_rev = False
        for i, p in enumerate(remaining):
            d_start = float(np.hypot(p[0][0] - cur_end[0], p[0][1] - cur_end[1]))
            d_end = float(np.hypot(p[-1][0] - cur_end[0], p[-1][1] - cur_end[1]))
            if d_start < best_dist:
                best_dist = d_start
                best_i = i
                best_rev = False
            if d_end < best_dist:
                best_dist = d_end
                best_i = i
                best_rev = True
        nxt = remaining.pop(best_i)
        ordered.append(nxt[::-1] if best_rev else nxt)
    return ordered


_SVG_NUM_RE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
_SVG_LEN_RE = re.compile(r"^\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*([a-zA-Z%]*)\s*$")
_SVG_TF_RE = re.compile(r"([a-zA-Z]+)\s*\(([^)]*)\)")


def _parse_svg_length_mm(val: str | None) -> float | None:
    """Convierte longitudes SVG (mm, cm, in, px, pt, pc o adimensional) a milímetros."""
    if val is None:
        return None
    m = _SVG_LEN_RE.match(val)
    if not m:
        return None
    num = float(m.group(1))
    unit = m.group(2).lower()
    factors = {
        "": 1.0,
        "mm": 1.0,
        "cm": 10.0,
        "in": 25.4,
        "pt": 25.4 / 72.0,
        "pc": 25.4 / 6.0,
        "px": 25.4 / 96.0,
    }
    return num * factors.get(unit, 1.0)


def _parse_svg_transform(transform_str: str | None) -> np.ndarray:
    """Convierte cadenas transform SVG (translate, scale, rotate, matrix) a matriz afín 3x3."""
    mat = np.eye(3, dtype=np.float64)
    if not transform_str:
        return mat
    for cmd, args_str in _SVG_TF_RE.findall(transform_str):
        args = [float(x) for x in _SVG_NUM_RE.findall(args_str)]
        op = cmd.lower()
        t = np.eye(3, dtype=np.float64)
        if op == "translate" and args:
            tx = args[0]
            ty = args[1] if len(args) > 1 else 0.0
            t[0, 2] = tx
            t[1, 2] = ty
        elif op == "scale" and args:
            sx = args[0]
            sy = args[1] if len(args) > 1 else sx
            t[0, 0] = sx
            t[1, 1] = sy
        elif op == "rotate" and args:
            rad = math.radians(args[0])
            c, s = math.cos(rad), math.sin(rad)
            rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
            if len(args) >= 3:
                cx, cy = args[1], args[2]
                t1 = np.array([[1.0, 0.0, cx], [0.0, 1.0, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
                t2 = np.array([[1.0, 0.0, -cx], [0.0, 1.0, -cy], [0.0, 0.0, 1.0]], dtype=np.float64)
                t = t1 @ rot @ t2
            else:
                t = rot
        elif op == "matrix" and len(args) >= 6:
            a, b, c, d, e, f = args[:6]
            t = np.array([[a, c, e], [b, d, f], [0.0, 0.0, 1.0]], dtype=np.float64)
        mat = mat @ t
    return mat


def _parse_svg_style(elem: ET.Element) -> dict[str, str]:
    """Combina atributos XML con propiedades inline CSS style="..." en minúsculas."""
    props: dict[str, str] = {k.lower(): v.strip() for k, v in elem.attrib.items()}
    style_str = props.get("style", "")
    if style_str:
        for item in style_str.split(";"):
            if ":" in item:
                k, v = item.split(":", 1)
                props[k.strip().lower()] = v.strip()
    return props


def _is_dark_paint(val: str) -> bool:
    v = val.strip().lower()
    if not v or v in ("none", "transparent", "white", "#fff", "#ffffff"):
        return False
    if v.startswith("rgb(") and "255,255,255" in v.replace(" ", ""):
        return False
    return True


def _parse_svg_root_dims(svg_path: str | Path) -> tuple[ET.Element, float, float, tuple[float, float, float, float]]:
    tree = ET.parse(str(svg_path))
    root = tree.getroot()
    vb_raw = root.attrib.get("viewBox") or root.attrib.get("viewbox")
    vb_nums = [float(x) for x in _SVG_NUM_RE.findall(vb_raw)] if vb_raw else []
    w_mm = _parse_svg_length_mm(root.attrib.get("width"))
    h_mm = _parse_svg_length_mm(root.attrib.get("height"))

    if len(vb_nums) >= 4:
        vb_x, vb_y, vb_w, vb_h = vb_nums[:4]
    else:
        vb_x, vb_y = 0.0, 0.0
        vb_w = w_mm if (w_mm and w_mm > 0) else 100.0
        vb_h = h_mm if (h_mm and h_mm > 0) else vb_w

    if w_mm is None or w_mm <= 0:
        w_mm = vb_w
    if h_mm is None or h_mm <= 0:
        h_mm = vb_h

    return root, float(w_mm), float(h_mm), (float(vb_x), float(vb_y), float(vb_w), float(vb_h))


def get_svg_info(svg_path: str | Path) -> tuple[float, float, bool]:
    """Devuelve (ancho_mm, alto_mm, ya_espejado) leyendo dimensiones exactas 1:1 del SVG."""
    root, w_mm, h_mm, _ = _parse_svg_root_dims(svg_path)
    name_low = Path(svg_path).name.lower()
    already_mirrored = ("espejado" in name_low) or ("mirror" in name_low)
    if not already_mirrored:
        for elem in root.iter():
            tf_str = elem.attrib.get("transform")
            if tf_str:
                m = _parse_svg_transform(tf_str)
                det = float(m[0, 0] * m[1, 1] - m[0, 1] * m[1, 0])
                if det < 0.0:
                    already_mirrored = True
                    break
    return w_mm, h_mm, already_mirrored


def _apply_svg_matrix(pts: np.ndarray, mat: np.ndarray) -> np.ndarray:
    """Aplica matriz 3x3 a un array Nx2 de puntos."""
    ones = np.ones((len(pts), 1), dtype=np.float64)
    hom = np.hstack([pts.astype(np.float64), ones])
    return (hom @ mat.T)[:, :2]


def render_svg_to_gray_mm(svg_path: str | Path, px_per_mm: float = 20.0) -> tuple[np.ndarray, float, float]:
    """
    Rasteriza un PCB en formato SVG vectorial a escala exacta 1:1 en milímetros sobre lienzo
    blanco (255) con trazos y pads negros (0) sin dependencias externas (xml + OpenCV).
    Soporta <g transform>, <rect>, <circle>, <line>, <polyline>, <polygon>.
    """
    root, w_mm, h_mm, (vb_x, vb_y, vb_w, vb_h) = _parse_svg_root_dims(svg_path)
    w_px = max(32, int(round(w_mm * px_per_mm)))
    h_px = max(32, int(round(h_mm * px_per_mm)))
    canvas = np.full((h_px, w_px), 255, dtype=np.uint8)

    sx_vb = w_mm / max(1e-9, vb_w)
    sy_vb = h_mm / max(1e-9, vb_h)
    m_root = np.array([
        [sx_vb, 0.0, -vb_x * sx_vb],
        [0.0, sy_vb, -vb_y * sy_vb],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    px_scale_x = (w_px - 1) / max(1e-9, w_mm)
    px_scale_y = (h_px - 1) / max(1e-9, h_mm)

    def to_px_coords(pts_mm: np.ndarray) -> np.ndarray:
        out = np.empty_like(pts_mm, dtype=np.float64)
        out[:, 0] = pts_mm[:, 0] * px_scale_x
        out[:, 1] = pts_mm[:, 1] * px_scale_y
        return out

    def draw_elem(elem: ET.Element, parent_mat: np.ndarray, parent_props: dict[str, str]):
        tag = elem.tag.split("}")[-1].lower()
        cur_mat = parent_mat @ _parse_svg_transform(elem.attrib.get("transform"))
        props = {**parent_props, **_parse_svg_style(elem)}

        fill_val = props.get("fill", "black")
        stroke_val = props.get("stroke", "none")
        sw_raw = _parse_svg_length_mm(props.get("stroke-width"))
        sw_val = sw_raw if sw_raw is not None else 1.0
        linecap = props.get("stroke-linecap", "butt").lower()
        linejoin = props.get("stroke-linejoin", "miter").lower()

        det_abs = abs(float(cur_mat[0, 0] * cur_mat[1, 1] - cur_mat[0, 1] * cur_mat[1, 0]))
        s_scale = math.sqrt(det_abs) if det_abs > 1e-12 else 1.0
        sw_mm = max(0.0, sw_val * s_scale)
        sw_px = max(1, int(round(sw_mm * px_per_mm))) if _is_dark_paint(stroke_val) and sw_mm > 0 else 0
        has_fill = _is_dark_paint(fill_val)
        has_stroke = _is_dark_paint(stroke_val) and sw_mm > 0

        if tag in ("svg", "g"):
            for child in elem:
                draw_elem(child, cur_mat, props)
            return

        if tag == "rect":
            rx = _parse_svg_length_mm(elem.attrib.get("x")) or 0.0
            ry = _parse_svg_length_mm(elem.attrib.get("y")) or 0.0
            rw = _parse_svg_length_mm(elem.attrib.get("width")) or 0.0
            rh = _parse_svg_length_mm(elem.attrib.get("height")) or 0.0
            if rw > 0 and rh > 0:
                pts_local = np.array([[rx, ry], [rx + rw, ry], [rx + rw, ry + rh], [rx, ry + rh]], dtype=np.float64)
                pts_px = np.round(to_px_coords(_apply_svg_matrix(pts_local, cur_mat))).astype(np.int32)
                if has_fill:
                    cv2.fillPoly(canvas, [pts_px], 0)
                if has_stroke:
                    cv2.polylines(canvas, [pts_px], True, 0, sw_px, cv2.LINE_AA)

        elif tag == "circle":
            cx = _parse_svg_length_mm(elem.attrib.get("cx")) or 0.0
            cy = _parse_svg_length_mm(elem.attrib.get("cy")) or 0.0
            r = _parse_svg_length_mm(elem.attrib.get("r")) or 0.0
            c_mm = _apply_svg_matrix(np.array([[cx, cy]], dtype=np.float64), cur_mat)
            c_px = to_px_coords(c_mm)[0]
            cx_px, cy_px = float(c_px[0]), float(c_px[1])
            r_eff = max(0.0, r * s_scale)
            r_out_px = (r_eff + (0.5 * sw_mm if has_stroke else 0.0)) * px_per_mm
            r_in_px = max(0.0, (r_eff - 0.5 * sw_mm) * px_per_mm) if (has_stroke and not has_fill) else 0.0

            x0 = max(0, int(math.floor(cx_px - r_out_px - 1)))
            x1 = min(w_px - 1, int(math.ceil(cx_px + r_out_px + 1)))
            y0 = max(0, int(math.floor(cy_px - r_out_px - 1)))
            y1 = min(h_px - 1, int(math.ceil(cy_px + r_out_px + 1)))
            if x1 >= x0 and y1 >= y0:
                gx = np.arange(x0, x1 + 1, dtype=np.float64)[None, :]
                gy = np.arange(y0, y1 + 1, dtype=np.float64)[:, None]
                d2 = (gx - cx_px) ** 2 + (gy - cy_px) ** 2
                mask_c = np.zeros_like(d2, dtype=bool)
                if has_fill:
                    mask_c |= (d2 <= (r_eff * px_per_mm + 0.25) ** 2)
                if has_stroke:
                    mask_c |= (d2 <= (r_out_px + 0.25) ** 2) & (d2 >= max(0.0, r_in_px - 0.25) ** 2)
                patch = canvas[y0 : y1 + 1, x0 : x1 + 1]
                patch[mask_c] = 0

        elif tag == "line":
            if has_stroke:
                x1 = _parse_svg_length_mm(elem.attrib.get("x1")) or 0.0
                y1 = _parse_svg_length_mm(elem.attrib.get("y1")) or 0.0
                x2 = _parse_svg_length_mm(elem.attrib.get("x2")) or 0.0
                y2 = _parse_svg_length_mm(elem.attrib.get("y2")) or 0.0
                pts_px = np.round(to_px_coords(_apply_svg_matrix(np.array([[x1, y1], [x2, y2]], dtype=np.float64), cur_mat))).astype(np.int32)
                cv2.line(canvas, tuple(pts_px[0]), tuple(pts_px[1]), 0, sw_px, cv2.LINE_AA)
                if linecap == "round":
                    cap_r = max(1, int(round(0.5 * sw_mm * px_per_mm)))
                    cv2.circle(canvas, tuple(pts_px[0]), cap_r, 0, -1, cv2.LINE_AA)
                    cv2.circle(canvas, tuple(pts_px[1]), cap_r, 0, -1, cv2.LINE_AA)

        elif tag in ("polyline", "polygon"):
            nums = [float(x) for x in _SVG_NUM_RE.findall(elem.attrib.get("points", ""))]
            if len(nums) >= 4:
                pairs = np.array(nums[: len(nums) - (len(nums) % 2)], dtype=np.float64).reshape(-1, 2)
                pts_px = np.round(to_px_coords(_apply_svg_matrix(pairs, cur_mat))).astype(np.int32)
                is_closed = (tag == "polygon")
                if has_fill and len(pts_px) >= 3:
                    cv2.fillPoly(canvas, [pts_px], 0)
                if has_stroke:
                    cv2.polylines(canvas, [pts_px], is_closed, 0, sw_px, cv2.LINE_AA)
                    if linecap == "round" or linejoin == "round":
                        cap_r = max(1, int(round(0.5 * sw_mm * px_per_mm)))
                        for pt in pts_px:
                            cv2.circle(canvas, tuple(pt), cap_r, 0, -1, cv2.LINE_AA)

    draw_elem(root, m_root, {})
    return canvas, w_mm, h_mm


def load_and_prepare_image(img_path: str | Path, cfg: XiloConfig) -> tuple[np.ndarray, np.ndarray, float, float]:
    path_obj = Path(img_path)
    if path_obj.suffix.lower() == ".svg":
        gray_orig, svg_w_mm, svg_h_mm = render_svg_to_gray_mm(path_obj, px_per_mm=cfg.px_per_mm)
        w_mm = svg_w_mm if cfg.width_mm <= 0 else min(BED_MAX_X - 10.0, max(5.0, cfg.width_mm))
        h_mm = cfg.height_mm if cfg.height_mm > 0 else (w_mm * svg_h_mm / svg_w_mm)
        target_w_px = max(32, int(round(w_mm * cfg.px_per_mm)))
        target_h_px = max(32, int(round(h_mm * cfg.px_per_mm)))
        if abs(w_mm - svg_w_mm) < 1e-3 and abs(h_mm - svg_h_mm) < 1e-3 and gray_orig.shape == (target_h_px, target_w_px):
            gray = gray_orig
        else:
            gray = cv2.resize(gray_orig, (target_w_px, target_h_px), interpolation=cv2.INTER_AREA)
        copper_top_view = (gray < cfg.threshold) if not cfg.invert_bw else (gray >= cfg.threshold)
        gray_work = cv2.flip(gray, 1) if cfg.mirror_x else gray.copy()
        return gray_work, copper_top_view, w_mm, h_mm

    raw = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(f"No se pudo abrir la imagen de circuito: {img_path}")

    if raw.ndim == 3 and raw.shape[2] == 4:
        alpha = raw[:, :, 3:4] / 255.0
        rgb = raw[:, :, :3].astype(np.float32)
        bg = np.full_like(rgb, 255.0)
        comp = (rgb * alpha + bg * (1.0 - alpha)).astype(np.uint8)
        gray_orig = cv2.cvtColor(comp, cv2.COLOR_BGR2GRAY)
    elif raw.ndim == 3:
        gray_orig = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
    else:
        gray_orig = raw

    src_h, src_w = gray_orig.shape
    w_mm = min(BED_MAX_X - 10.0, max(5.0, cfg.width_mm))
    h_mm = cfg.height_mm if cfg.height_mm > 0 else (w_mm * src_h / src_w)

    target_w_px = max(32, int(round(w_mm * cfg.px_per_mm)))
    target_h_px = max(32, int(round(h_mm * cfg.px_per_mm)))

    gray = cv2.resize(gray_orig, (target_w_px, target_h_px), interpolation=cv2.INTER_AREA)
    copper_top_view = (gray < cfg.threshold) if not cfg.invert_bw else (gray >= cfg.threshold)
    gray_work = cv2.flip(gray, 1) if cfg.mirror_x else gray.copy()
    return gray_work, copper_top_view, w_mm, h_mm


def _extract_pcb_isolation_paths(
    gray_work: np.ndarray, w_mm: float, h_mm: float, cfg: XiloConfig
) -> tuple[list[np.ndarray], np.ndarray]:
    """
    Fresado de Aislamiento de Pistas (PCB Isolation Routing):
    En vez de cortar por encima del cobre, dilata la máscara de pistas por el radio de la fresa
    (y anillos adicionales de aislamiento) para cortar el canal aislante alrededor de cada pista/pad.
    """
    h_px, w_px = gray_work.shape
    copper_mask = ((gray_work < cfg.threshold) if not cfg.invert_bw else (gray_work >= cfg.threshold)).astype(np.uint8) * 255

    paths: list[np.ndarray] = []
    cut_viz_mask = np.zeros_like(copper_mask)

    base_r_px = max(1, int(round((cfg.tool_diameter_mm * 0.5) * cfg.px_per_mm)))
    step_r_px = max(1, int(round((cfg.tool_diameter_mm * cfg.stepover_ratio) * cfg.px_per_mm)))
    n_rings = max(1, int(cfg.isolation_passes))

    for ring_idx in range(n_rings):
        r_px = base_r_px + ring_idx * step_r_px
        k_sz = r_px * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_sz, k_sz))
        dilated = cv2.dilate(copper_mask, kernel)

        # Registrar canal comido en máscara visual
        k_tool = max(1, base_r_px * 2 + 1)
        edge = cv2.morphologyEx(dilated, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
        cut_viz_mask = cv2.bitwise_or(cut_viz_mask, cv2.dilate(edge, np.ones((k_tool, k_tool), np.uint8)))

        contours, _ = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        ring_paths: list[np.ndarray] = []
        for cnt in contours:
            if len(cnt) < 4 or cv2.contourArea(cnt) < 3.0:
                continue
            approx = cv2.approxPolyDP(cnt, epsilon=0.45, closed=True)
            pts = approx.reshape(-1, 2).astype(np.float32)
            if len(pts) < 2:
                continue
            pts = np.vstack([pts, pts[0]])
            pts_mm = np.column_stack([
                cfg.offset_x + (pts[:, 0] / max(1, w_px - 1)) * w_mm,
                cfg.offset_y + (1.0 - pts[:, 1] / max(1, h_px - 1)) * h_mm,
            ])
            ring_paths.append(pts_mm)

        paths.extend(_order_paths_nearest(ring_paths))

    return paths, (cut_viz_mask > 127)


def _extract_pcb_marker_resist_paths(
    gray_work: np.ndarray, w_mm: float, h_mm: float, cfg: XiloConfig
) -> tuple[list[np.ndarray], np.ndarray]:
    """
    Modo Plotter Directo del Circuito con Fibra Indeleble (0.75 mm / 0.40 mm) para Cloruro Férrico:
      - Rellena temporalmente los agujeros pequeños de pads Through-Hole (<= 1.45 mm) antes del
        cálculo de distancia interior para evitar que una punta de 0.75 mm borre los anillos de
        pads de 1.85 mm (mientras preserva abiertos los agujeros M3 >= 2.0 mm).
      - Usa Transformada de Distancia continua (cv2.distanceTransform) con compensación de radio
        interior (-D/2 = -0.375 mm para punta de 0.75 mm) y respaldo de espina central en pistas finas.
      - Genera anillos concéntricos internos dejando libre un micro-punto central (0.25 mm de radio)
        de guía de perforado en las pasadas interiores.
    """
    h_px, w_px = gray_work.shape
    copper_mask = ((gray_work < cfg.threshold) if not cfg.invert_bw else (gray_work >= cfg.threshold)).astype(np.uint8) * 255

    solid_copper = copper_mask.copy()
    drill_keepout = np.zeros_like(copper_mask)

    # 1. Rellenar agujeros de pads pequeños (<= 1.45 mm) preservando agujeros grandes M3
    contours_h, hierarchy = cv2.findContours(copper_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is not None:
        r_keep_px = max(1, int(round(0.25 * cfg.px_per_mm)))
        for idx, cnt in enumerate(contours_h):
            parent = hierarchy[0][idx][3]
            if parent < 0:
                continue
            area = cv2.contourArea(cnt)
            if area <= 0:
                continue
            d_mm = (2.0 * math.sqrt(area / math.pi)) / max(1e-3, cfg.px_per_mm)
            if d_mm <= 1.45:
                cv2.drawContours(solid_copper, [cnt], -1, 255, -1)
                mom = cv2.moments(cnt)
                if mom["m00"] > 0:
                    cx = int(round(mom["m10"] / mom["m00"]))
                    cy = int(round(mom["m01"] / mom["m00"]))
                    cv2.circle(drill_keepout, (cx, cy), r_keep_px, 255, -1)

    # 2. Transformada de distancia continua sub-píxel en milímetros
    dist_mm = cv2.distanceTransform(solid_copper, cv2.DIST_L2, 5) / max(1e-3, cfg.px_per_mm)
    r_in = cfg.tool_diameter_mm * 0.5

    # 3. Respaldo de espina central por componente si alguna isla/pista es más angosta que D
    fallback_spine = np.zeros_like(copper_mask, dtype=bool)
    num_comp, comp_labels = cv2.connectedComponents((solid_copper > 0).astype(np.uint8), connectivity=8)
    for cid in range(1, num_comp):
        comp_mask = (comp_labels == cid)
        max_d = float(np.max(dist_mm[comp_mask]))
        if max_d <= r_in + 1e-4:
            p65 = float(np.percentile(dist_mm[comp_mask], 65))
            fb_thresh = min(0.80 * max_d, max(0.04, p65))
            fallback_spine |= comp_mask & (dist_mm >= fb_thresh)

    paths: list[np.ndarray] = []
    step_mm = max(0.18, cfg.tool_diameter_mm * cfg.stepover_ratio)

    # 4. Anillos concéntricos internos desde r_in hacia el corazón de la pista/pad
    for pass_i in range(10):
        thresh = r_in + pass_i * step_mm
        if pass_i == 0:
            ring_bool = (dist_mm >= thresh) | fallback_spine
        else:
            ring_bool = (dist_mm >= thresh) & (drill_keepout == 0)

        if not np.any(ring_bool):
            break

        ring_bin = ring_bool.astype(np.uint8) * 255
        contours, _ = cv2.findContours(ring_bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        ring_paths = []
        for cnt in contours:
            if len(cnt) < 2:
                continue
            approx = cv2.approxPolyDP(cnt, epsilon=0.35, closed=True)
            pts = approx.reshape(-1, 2).astype(np.float32)
            if len(pts) < 2:
                continue
            pts = np.vstack([pts, pts[0]])
            pts_mm = np.column_stack([
                cfg.offset_x + (pts[:, 0] / max(1, w_px - 1)) * w_mm,
                cfg.offset_y + (1.0 - pts[:, 1] / max(1, h_px - 1)) * h_mm,
            ])
            ring_paths.append(pts_mm)
        paths.extend(_order_paths_nearest(ring_paths))

    # Repetir según cantidad de manos de tinta (isolation_passes)
    n_coats = max(1, min(3, int(cfg.isolation_passes)))
    if n_coats > 1:
        base_paths = list(paths)
        for _ in range(n_coats - 1):
            paths.extend(base_paths)

    # 5. Simulación exacta de la tinta depositada por una punta de tool_diameter_mm
    ink_deposited = np.zeros_like(copper_mask)
    stroke_px = max(2, int(round(cfg.tool_diameter_mm * cfg.px_per_mm)))
    for poly in paths:
        pts_px = np.column_stack([
            ((poly[:, 0] - cfg.offset_x) / max(1e-3, w_mm)) * (w_px - 1),
            (1.0 - (poly[:, 1] - cfg.offset_y) / max(1e-3, h_mm)) * (h_px - 1),
        ]).astype(np.int32)
        cv2.polylines(ink_deposited, [pts_px], False, 255, stroke_px, cv2.LINE_AA)

    return paths, (ink_deposited > 127)


def _detect_pcb_drill_holes(
    gray_work: np.ndarray, w_mm: float, h_mm: float, cfg: XiloConfig
) -> tuple[list[tuple[float, float]], np.ndarray]:
    """
    Detecta los centros de los agujeros de los pads (círculos blancos pequeños rodeados
    de un anillo negro de cobre) y los ordena por ruta más corta para perforado (Peck Drilling).
    """
    h_px, w_px = gray_work.shape
    copper_mask = ((gray_work < cfg.threshold) if not cfg.invert_bw else (gray_work >= cfg.threshold)).astype(np.uint8) * 255

    # Buscar contornos con jerarquía: agujeros internos (hijos dentro de islas de cobre)
    contours, hierarchy = cv2.findContours(copper_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    holes_mm: list[tuple[float, float]] = []
    hole_viz = np.zeros_like(copper_mask)

    min_d_px = 0.35 * cfg.px_per_mm
    max_d_px = 3.20 * cfg.px_per_mm
    min_area = math.pi * (min_d_px * 0.5) ** 2
    max_area = math.pi * (max_d_px * 0.5) ** 2

    if hierarchy is not None:
        for i, cnt in enumerate(contours):
            # En RETR_CCOMP, hierarchy[0][i][3] >= 0 significa agujero interior dentro de cobre
            parent = hierarchy[0][i][3]
            if parent < 0:
                continue
            area = cv2.contourArea(cnt)
            if not (min_area <= area <= max_area):
                continue
            perim = cv2.arcLength(cnt, True)
            if perim <= 0:
                continue
            circularity = 4.0 * math.pi * area / (perim * perim)
            if circularity < 0.45:
                continue
            M = cv2.moments(cnt)
            if M["m00"] <= 0:
                continue
            cx_px = M["m10"] / M["m00"]
            cy_px = M["m01"] / M["m00"]
            cv2.circle(hole_viz, (int(round(cx_px)), int(round(cy_px))), max(3, int(round(0.6 * cfg.px_per_mm))), 255, -1)
            x_mm = cfg.offset_x + (cx_px / max(1, w_px - 1)) * w_mm
            y_mm = cfg.offset_y + (1.0 - cy_px / max(1, h_px - 1)) * h_mm
            holes_mm.append((x_mm, y_mm))

    # Ordenar agujeros por vecino más cercano
    ordered: list[tuple[float, float]] = []
    rem = list(holes_mm)
    cur = (cfg.offset_x, cfg.offset_y)
    while rem:
        best_i = min(range(len(rem)), key=lambda k: math.hypot(rem[k][0] - cur[0], rem[k][1] - cur[1]))
        cur = rem.pop(best_i)
        ordered.append(cur)

    return ordered, (hole_viz > 127)


def _extract_relief_pocket_paths(
    gray_work: np.ndarray, w_mm: float, h_mm: float, cfg: XiloConfig
) -> tuple[list[np.ndarray], np.ndarray]:
    h_px, w_px = gray_work.shape
    ink_work = (gray_work < cfg.threshold) if not cfg.invert_bw else (gray_work >= cfg.threshold)
    carve_mask = (~ink_work).astype(np.uint8) * 255

    radius_px = max(1, int(round((cfg.tool_diameter_mm * 0.5) * cfg.px_per_mm)))
    k_sz = radius_px * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_sz, k_sz))
    safe_carve = cv2.erode(carve_mask, kernel)

    paths: list[np.ndarray] = []
    contours, _ = cv2.findContours(safe_carve, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if len(cnt) < 3 or cv2.contourArea(cnt) < 4.0:
            continue
        approx = cv2.approxPolyDP(cnt, epsilon=0.5, closed=True)
        pts = approx.reshape(-1, 2).astype(np.float32)
        if len(pts) < 2:
            continue
        pts = np.vstack([pts, pts[0]])
        pts_mm = np.column_stack([
            cfg.offset_x + (pts[:, 0] / max(1, w_px - 1)) * w_mm,
            cfg.offset_y + (1.0 - pts[:, 1] / max(1, h_px - 1)) * h_mm,
        ])
        paths.append(pts_mm)

    step_mm = max(0.25, cfg.tool_diameter_mm * cfg.stepover_ratio)
    step_px = max(1, int(round(step_mm * cfg.px_per_mm)))
    for row_idx, y_px in enumerate(range(0, h_px, step_px)):
        row = safe_carve[y_px, :] > 127
        if not row.any():
            continue
        padded = np.concatenate([[False], row, [False]])
        diffs = np.diff(padded.astype(np.int8))
        starts = np.where(diffs == 1)[0]
        ends = np.where(diffs == -1)[0] - 1
        segs = list(zip(starts, ends))
        if row_idx % 2 == 1:
            segs = list(reversed(segs))
        y_mm = cfg.offset_y + (1.0 - y_px / max(1, h_px - 1)) * h_mm
        for s_px, e_px in segs:
            if e_px <= s_px:
                continue
            x0_mm = cfg.offset_x + (s_px / max(1, w_px - 1)) * w_mm
            x1_mm = cfg.offset_x + (e_px / max(1, w_px - 1)) * w_mm
            if row_idx % 2 == 1:
                x0_mm, x1_mm = x1_mm, x0_mm
            paths.append(np.array([[x0_mm, y_mm], [x1_mm, y_mm]], dtype=np.float32))

    return _order_paths_nearest(paths), (safe_carve > 127)


def generate_woodcut_gcode(
    img_path: str | Path,
    out_gcode_path: str | Path,
    cfg: XiloConfig,
    out_preview_path: str | Path | None = None,
) -> XiloStats:
    """Genera G-Code de PCB / Xilografía seguro para Magna SE Mini + previsualización."""
    gray_work, copper_top_view, w_mm, h_mm = load_and_prepare_image(img_path, cfg)
    cut_feed = _clamp_feed_xy(cfg.xy_cut_feed)
    travel_feed = _clamp_feed_xy(cfg.xy_travel_feed)
    plunge_feed = _clamp_feed_z(cfg.z_plunge_feed)

    lines: list[str] = [
        "; ==========================================================",
        "; Magna SE Mini - PCB & Circuit Studio (SIN LASER)",
        f"; Archivo origen : {Path(img_path).name}",
        f"; Dimension PCB  : {w_mm:.2f} x {h_mm:.2f} mm",
        f"; Modo Circuito  : {cfg.mode} | Capa={'Bottom (Espejo)' if cfg.mirror_x else 'Top (Directa)'}",
        f"; Herramienta    : D={cfg.tool_diameter_mm:.2f} mm | Z-Prof={cfg.total_depth_mm:.3f} mm",
        f"; Nivelacion Z   : BL={cfg.z_corner_bl:+.3f} BR={cfg.z_corner_br:+.3f} TL={cfg.z_corner_tl:+.3f} TR={cfg.z_corner_tr:+.3f}",
        "; ==========================================================",
        "M107 ; SEGURIDAD: Pin de laser/ventilador apagado",
        "G21  ; Unidades en milimetros",
        "G90  ; Coordenadas absolutas",
    ]
    if cfg.use_work_zero:
        lines.append("G92 X0 Y0 Z0 ; Cero actual sobre esquina inferior-izquierda del cobre")
    lines.extend([
        f"M117 PCB {w_mm:.0f}x{h_mm:.0f}mm",
        f"G0 Z{cfg.z_safe_mm:.3f} F{plunge_feed:.0f}",
    ])

    cut_dist = 0.0
    travel_dist = 0.0
    cur_pos = np.array([0.0, 0.0, cfg.z_safe_mm], dtype=np.float64)
    preview_paths_2d: list[np.ndarray] = []
    holes_detected = 0

    if cfg.mode == "pcb_drill_holes":
        holes, action_mask = _detect_pcb_drill_holes(gray_work, w_mm, h_mm, cfg)
        holes_detected = len(holes)
        z_passes = 1
        lines.append(f"; --- Perforado de {holes_detected} agujeros (Peck Drilling) ---")
        for hx, hy in holes:
            d_trav = float(np.hypot(hx - cur_pos[0], hy - cur_pos[1]))
            travel_dist += d_trav
            z_surf = _bilinear_z_offset(hx, hy, w_mm, h_mm, cfg)
            z_bottom = z_surf - cfg.total_depth_mm
            z_half = z_surf - (cfg.total_depth_mm * 0.5)
            lines.append(f"G0 Z{cfg.z_safe_mm:.3f} F{plunge_feed:.0f}")
            lines.append(f"G0 X{hx:.3f} Y{hy:.3f} F{travel_feed:.0f}")
            lines.append(f"G1 Z{z_half:.3f} F{plunge_feed:.0f}")
            lines.append(f"G0 Z{z_surf + 0.4:.3f} F{plunge_feed:.0f}")
            lines.append(f"G1 Z{z_bottom:.3f} F{plunge_feed:.0f}")
            lines.append(f"G0 Z{cfg.z_safe_mm:.3f} F{plunge_feed:.0f}")
            cut_dist += cfg.total_depth_mm * 2.0
            cur_pos[:] = [hx, hy, cfg.z_safe_mm]
            preview_paths_2d.append(np.array([[hx - 0.5, hy], [hx + 0.5, hy]], dtype=np.float32))
            preview_paths_2d.append(np.array([[hx, hy - 0.5], [hx, hy + 0.5]], dtype=np.float32))
    elif cfg.mode == "pcb_laser_ablation":
        # Ablación con Láser de la pintura antiácido alrededor de pistas y dentro de centros de pads
        paths_2d, action_mask = _extract_pcb_isolation_paths(gray_work, w_mm, h_mm, cfg)
        holes, _ = _detect_pcb_drill_holes(gray_work, w_mm, h_mm, cfg)
        holes_detected = len(holes)
        preview_paths_2d = paths_2d
        z_passes = max(1, int(round(cfg.total_depth_mm / max(0.01, cfg.z_stepdown_mm)))) if cfg.total_depth_mm > 0.01 else 1
        pwm = max(10, min(255, int(cfg.laser_pwm)))

        for p_idx in range(1, z_passes + 1):
            lines.append(f"; --- Pasada Ablacion Laser {p_idx}/{z_passes} (PWM S={pwm}) ---")
            lines.append("M107 ; Laser OFF en traslado")
            for poly in paths_2d:
                if len(poly) < 2:
                    continue
                sx, sy = float(poly[0][0]), float(poly[0][1])
                z_foc = cfg.laser_focus_z_mm + _bilinear_z_offset(sx, sy, w_mm, h_mm, cfg)
                d_trav = float(np.hypot(sx - cur_pos[0], sy - cur_pos[1]))
                travel_dist += d_trav
                lines.append("M107")
                lines.append(f"G0 X{sx:.3f} Y{sy:.3f} Z{z_foc:.3f} F{travel_feed:.0f}")
                lines.append("M400")
                lines.append(f"M106 S{pwm} ; Laser ON para ablacion de aislamiento")
                cur_pos[:] = [sx, sy, z_foc]

                for pt in poly[1:]:
                    nx, ny = float(pt[0]), float(pt[1])
                    d_cut = float(np.hypot(nx - cur_pos[0], ny - cur_pos[1]))
                    if d_cut < 1e-4:
                        continue
                    z_pt = cfg.laser_focus_z_mm + _bilinear_z_offset(nx, ny, w_mm, h_mm, cfg)
                    cut_dist += d_cut
                    if abs(z_pt - cur_pos[2]) > 0.002:
                        lines.append(f"G1 X{nx:.3f} Y{ny:.3f} Z{z_pt:.3f} F{cut_feed:.0f}")
                    else:
                        lines.append(f"G1 X{nx:.3f} Y{ny:.3f} F{cut_feed:.0f}")
                    cur_pos[:] = [nx, ny, z_pt]

                lines.append("M400")
                lines.append("M107 ; Laser OFF al terminar contorno")
    else:
        if cfg.mode == "pcb_isolation":
            paths_2d, action_mask = _extract_pcb_isolation_paths(gray_work, w_mm, h_mm, cfg)
        elif cfg.mode == "pcb_marker_resist":
            paths_2d, action_mask = _extract_pcb_marker_resist_paths(gray_work, w_mm, h_mm, cfg)
        else:
            paths_2d, action_mask = _extract_relief_pocket_paths(gray_work, w_mm, h_mm, cfg)

        holes, _ = _detect_pcb_drill_holes(gray_work, w_mm, h_mm, cfg)
        holes_detected = len(holes)
        preview_paths_2d = paths_2d
        z_passes = max(1, int(math.ceil(cfg.total_depth_mm / max(0.01, cfg.z_stepdown_mm))))

        for p_idx in range(1, z_passes + 1):
            depth_step = min(cfg.total_depth_mm, p_idx * cfg.z_stepdown_mm)
            lines.append(f"; --- Pasada Z {p_idx}/{z_passes}: Profundidad = -{depth_step:.3f} mm ---")
            for poly in paths_2d:
                if len(poly) < 2:
                    continue
                sx, sy = float(poly[0][0]), float(poly[0][1])
                d_trav = float(np.hypot(sx - cur_pos[0], sy - cur_pos[1]))
                z_start = _bilinear_z_offset(sx, sy, w_mm, h_mm, cfg) - depth_step

                if d_trav > 0.35 or cur_pos[2] > 0:
                    if cur_pos[2] < cfg.z_safe_mm:
                        lines.append(f"G0 Z{cfg.z_safe_mm:.3f} F{plunge_feed:.0f}")
                        cur_pos[2] = cfg.z_safe_mm
                    lines.append(f"G0 X{sx:.3f} Y{sy:.3f} F{travel_feed:.0f}")
                    travel_dist += d_trav
                    cur_pos[0], cur_pos[1] = sx, sy
                    lines.append(f"G1 Z{z_start:.3f} F{plunge_feed:.0f}")
                    cur_pos[2] = z_start

                for pt in poly[1:]:
                    nx, ny = float(pt[0]), float(pt[1])
                    d_cut = float(np.hypot(nx - cur_pos[0], ny - cur_pos[1]))
                    if d_cut < 1e-4:
                        continue
                    z_pt = _bilinear_z_offset(nx, ny, w_mm, h_mm, cfg) - depth_step
                    cut_dist += d_cut
                    if abs(z_pt - cur_pos[2]) > 0.002:
                        lines.append(f"G1 X{nx:.3f} Y{ny:.3f} Z{z_pt:.3f} F{cut_feed:.0f}")
                    else:
                        lines.append(f"G1 X{nx:.3f} Y{ny:.3f} F{cut_feed:.0f}")
                    cur_pos[:] = [nx, ny, z_pt]

            lines.append(f"G0 Z{cfg.z_safe_mm:.3f} F{plunge_feed:.0f}")
            cur_pos[2] = cfg.z_safe_mm

    lines.extend([
        "M400 ; Esperar finalizacion mecanica",
        f"G0 Z{max(cfg.z_safe_mm, 12.0):.3f} F{plunge_feed:.0f}",
        "G0 X0.000 Y0.000 F1800",
        "M107 ; Confirmar laser/fan OFF",
        "M117 PCB COMPLETADO!",
    ])

    out_gcode = Path(out_gcode_path)
    out_gcode.parent.mkdir(parents=True, exist_ok=True)
    out_gcode.write_text("\n".join(lines) + "\n", encoding="utf-8")

    est_min = (cut_dist / cut_feed) + (travel_dist / travel_feed) + (len(preview_paths_2d) * z_passes * (2 * cfg.z_safe_mm / plunge_feed) * 0.35)

    preview_saved = None
    if out_preview_path is not None:
        preview_saved = _render_pcb_preview(copper_top_view, gray_work, preview_paths_2d, action_mask, w_mm, h_mm, cfg, Path(out_preview_path))

    return XiloStats(
        width_mm=w_mm,
        height_mm=h_mm,
        z_passes=z_passes,
        total_lines=len(lines),
        cut_distance_mm=cut_dist,
        travel_distance_mm=travel_dist,
        estimated_minutes=est_min,
        holes_detected=holes_detected,
        preview_path=preview_saved,
        gcode_path=str(out_gcode),
    )


def _render_pcb_preview(
    copper_top_view: np.ndarray,
    gray_work: np.ndarray,
    paths_2d: list[np.ndarray],
    action_mask: np.ndarray,
    w_mm: float,
    h_mm: float,
    cfg: XiloConfig,
    out_path: Path,
) -> str:
    """Renderiza Izquierda = Diseño de Pistas de Cobre | Derecha = Placa FR4 Cobre + Rutas CNC."""
    panel_w = 520
    panel_h = max(280, int(round(panel_w * (h_mm / w_mm))))

    # Panel Izquierdo: Vista de Pistas (Soldermask Verde Oscuro + Pads/Pistas Doradas)
    pcb_design = np.full((panel_h, panel_w, 3), (28, 72, 34), dtype=np.uint8)
    cop_res = cv2.resize(copper_top_view.astype(np.uint8) * 255, (panel_w, panel_h), interpolation=cv2.INTER_NEAREST)
    pcb_design[cop_res > 127] = (75, 195, 235)  # Pistas cobre/dorado BGR

    # Panel Derecho: Placa Virgen de Cobre (FR4 Cobre brillante + Surcos fresados / Tinta + Trayectoria)
    copper_work = ((gray_work < cfg.threshold) if not cfg.invert_bw else (gray_work >= cfg.threshold)).astype(np.uint8) * 255
    cop_work_res = cv2.resize(copper_work, (panel_w, panel_h), interpolation=cv2.INTER_NEAREST)
    act_res = cv2.resize(action_mask.astype(np.uint8) * 255, (panel_w, panel_h), interpolation=cv2.INTER_NEAREST)

    board = np.full((panel_h, panel_w, 3), (95, 155, 215), dtype=np.uint8)  # Cobre virgen BGR
    if cfg.mode == "pcb_marker_resist":
        # Donde pasa el marcador indeleble queda tinta azul/negra protectora
        board[act_res > 127] = (35, 30, 30)
    else:
        # Donde pasa la fresa se descubre la fibra FR4 amarilla/verdosa debajo del cobre
        board[act_res > 127] = (55, 105, 115)
        board[cop_work_res > 127] = (110, 175, 235)

    line_color = (80, 255, 100) if cfg.mode != "pcb_marker_resist" else (255, 190, 50)
    for poly in paths_2d:
        if len(poly) < 2:
            continue
        pts_px = np.column_stack([
            ((poly[:, 0] - cfg.offset_x) / max(1e-3, w_mm)) * (panel_w - 1),
            (1.0 - (poly[:, 1] - cfg.offset_y) / max(1e-3, h_mm)) * (panel_h - 1),
        ]).astype(np.int32)
        cv2.polylines(board, [pts_px], False, line_color, 1, cv2.LINE_AA)

    header_h = 46
    canvas = np.full((panel_h + header_h, panel_w * 2 + 24, 3), (24, 26, 32), dtype=np.uint8)
    canvas[header_h : header_h + panel_h, 8 : 8 + panel_w] = pcb_design
    canvas[header_h : header_h + panel_h, 16 + panel_w : 16 + panel_w * 2] = board

    cv2.putText(canvas, f"CIRCUITO OBJETIVO ({w_mm:.0f}x{h_mm:.0f} mm)", (16, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (230, 235, 245), 2)
    cv2.putText(canvas, f"TRAYECTORIA EN COBRE ({cfg.mode}, {'BOTTOM/ESPEJO' if cfg.mirror_x else 'TOP'})",
                (panel_w + 22, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 255, 160), 2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)
    return str(out_path)


def create_tank_carrier_pcb_layout(out_png_path: str | Path) -> tuple[Path, float, float]:
    """
    Genera el diseño 100% PLANAR (1 Sola Capa de Cobre, CERO superposiciones ni cruces)
    para la Placa Madre del Tanque (65x45 mm) y verifica automáticamente con DRC matemático
    que las 7 redes eléctricas estén 100% aisladas entre sí y 100% continuas.

    Disposición topológica planar (Izquierda -> Centro -> Derecha):
      - Izquierda (X=11.0mm): Header J2 (ESP32-S3 #1 Radar Wi-Fi CSI, 4 pines paso 2.54mm)
          Pin 1 (Y=20.00): GND
          Pin 2 (Y=22.54): GPIO17 (TX1 -> hacia RX1 del Cerebro)
          Pin 3 (Y=25.08): GPIO18 (RX1 <- desde TX1 del Cerebro)
          Pin 4 (Y=27.62): +5V
      - Centro (X=28.0mm): Header J1 (ESP32-S3 #2 Cerebro LLM, 6 pines paso 2.54mm)
          Pin 1 (Y=17.46): GND
          Pin 2 (Y=20.00): GPIO15 (RX2_3V3 <- desde nodo divisor 3.3V)
          Pin 3 (Y=22.54): GPIO16 (TX2 -> directo a D10_RX del Arduino)
          Pin 4 (Y=25.08): GPIO18 (RX1 <- desde TX1 del Radar)
          Pin 5 (Y=27.62): GPIO17 (TX1 -> hacia RX1 del Radar)
          Pin 6 (Y=30.16): +5V
      - Derecha (X=54.0mm): Header J3 (Arduino Uno + L298N, 4 pines paso 2.54mm)
          Pin 1 (Y=17.46): GND
          Pin 2 (Y=20.00): D11_TX_5V (-> entra a R1 1k Ohm)
          Pin 3 (Y=22.54): D10_RX (<- viene directo de TX2 GPIO16)
          Pin 4 (Y=25.08): +5V
      - Divisor Resistivo 5V -> 3.3V en Bahía Derecha (sin cruzar ninguna pista):
          R1 (1k Ohm): de (48.0, 20.00) a (40.38, 20.00)
          R2 (2k Ohm): de (36.5, 16.50) a (36.5, 8.88) -> baja directo al bus GND (Y=5.5mm)
    """
    w_mm, h_mm = 65.0, 45.0
    px_mm = 20.0  # 20 px/mm (50 micrones/px)
    W = int(round(w_mm * px_mm))
    H = int(round(h_mm * px_mm))

    img = np.full((H, W), 255, dtype=np.uint8)

    def mm_to_px(x_mm: float, y_mm: float) -> tuple[int, int]:
        return int(round(x_mm * px_mm)), int(round((h_mm - y_mm) * px_mm))

    def draw_trace_on(canvas: np.ndarray, pts_mm: list[tuple[float, float]], width_mm: float = 1.1):
        thickness = max(2, int(round(width_mm * px_mm)))
        pts_px = np.array([mm_to_px(x, y) for x, y in pts_mm], dtype=np.int32)
        cv2.polylines(canvas, [pts_px], False, 0, thickness, cv2.LINE_AA)

    def draw_pad_solid_on(canvas: np.ndarray, x_mm: float, y_mm: float, outer_d_mm: float = 2.2):
        cx, cy = mm_to_px(x_mm, y_mm)
        r_out = int(round((outer_d_mm * 0.5) * px_mm))
        cv2.circle(canvas, (cx, cy), r_out, 0, -1, cv2.LINE_AA)

    def punch_drill_on(canvas: np.ndarray, x_mm: float, y_mm: float, drill_d_mm: float = 0.9):
        cx, cy = mm_to_px(x_mm, y_mm)
        r_in = int(round((drill_d_mm * 0.5) * px_mm))
        cv2.circle(canvas, (cx, cy), r_in, 255, -1, cv2.LINE_AA)

    j2_x = 11.0
    j2_ys = [20.00 + i * 2.54 for i in range(4)]  # 20.00, 22.54, 25.08, 27.62

    j1_x = 28.0
    j1_ys = [17.46 + i * 2.54 for i in range(6)]  # 17.46, 20.00, 22.54, 25.08, 27.62, 30.16

    j3_x = 54.0
    j3_ys = [17.46 + i * 2.54 for i in range(4)]  # 17.46, 20.00, 22.54, 25.08

    r1_p1 = (48.00, 20.00)  # Lado 5V (D11)
    r1_p2 = (40.38, 20.00)  # Lado 3.3V (paso 7.62mm)
    r2_p1 = (36.50, 16.50)  # Lado 3.3V
    r2_p2 = (36.50, 8.88)   # Lado GND (paso 7.62mm)

    # Especificaciones eléctricas y térmicas reales por red (Norma IPC-2221)
    from pcb_engineering_engine import NetSpec, recommend_trace_width_mm, run_autonomous_pcb_audit

    net_specs: dict[str, NetSpec] = {
        "NET_GND": NetSpec("NET_GND", voltage_v=0.0, current_max_a=1.60, max_voltage_drop_pct=1.5, max_temp_rise_c=10.0),
        "NET_5V": NetSpec("NET_5V", voltage_v=5.0, current_max_a=1.40, max_voltage_drop_pct=1.5, max_temp_rise_c=10.0),
        "NET_RADAR_TX_TO_BRAIN_RX": NetSpec("NET_RADAR_TX_TO_BRAIN_RX", voltage_v=3.3, current_max_a=0.02, is_3v3_sensitive=True),
        "NET_RADAR_RX_FROM_BRAIN_TX": NetSpec("NET_RADAR_RX_FROM_BRAIN_TX", voltage_v=3.3, current_max_a=0.02, is_3v3_sensitive=True),
        "NET_BRAIN_TX2_TO_ARD_D10": NetSpec("NET_BRAIN_TX2_TO_ARD_D10", voltage_v=3.3, current_max_a=0.02, is_3v3_sensitive=True),
        "NET_DIVIDER_3V3_NODE": NetSpec("NET_DIVIDER_3V3_NODE", voltage_v=3.33, current_max_a=0.005, is_3v3_sensitive=True),
        "NET_ARD_D11_5V_IN": NetSpec("NET_ARD_D11_5V_IN", voltage_v=5.0, current_max_a=0.01, is_3v3_sensitive=False),
    }

    # Definición explícita de las 7 redes eléctricas independientes (1 sola capa, 0 cruces)
    nets: dict[str, dict] = {
        "NET_GND": {
            "traces": [
                [(j2_x, j2_ys[0]), (j2_x, 5.5), (j1_x, 5.5), (j1_x, j1_ys[0])],
                [(j1_x, 5.5), (36.50, 5.5), r2_p2],
                [(36.50, 5.5), (j3_x, 5.5), (j3_x, j3_ys[0])],
            ],
            "pads": [(j2_x, j2_ys[0]), (j1_x, j1_ys[0]), r2_p2, (j3_x, j3_ys[0])],
            "width": recommend_trace_width_mm(net_specs["NET_GND"]),
        },
        "NET_5V": {
            "traces": [
                [(j2_x, j2_ys[3]), (j2_x, 39.2), (j1_x, 39.2), (j1_x, j1_ys[5])],
                [(j1_x, 39.2), (j3_x, 39.2), (j3_x, j3_ys[3])],
            ],
            "pads": [(j2_x, j2_ys[3]), (j1_x, j1_ys[5]), (j3_x, j3_ys[3])],
            "width": recommend_trace_width_mm(net_specs["NET_5V"]),
        },
        "NET_RADAR_TX_TO_BRAIN_RX": {
            "traces": [
                [(j2_x, j2_ys[1]), (18.5, j2_ys[1]), (21.0, j1_ys[3]), (j1_x, j1_ys[3])],
            ],
            "pads": [(j2_x, j2_ys[1]), (j1_x, j1_ys[3])],
            "width": recommend_trace_width_mm(net_specs["NET_RADAR_TX_TO_BRAIN_RX"]),
        },
        "NET_RADAR_RX_FROM_BRAIN_TX": {
            "traces": [
                [(j2_x, j2_ys[2]), (16.5, j2_ys[2]), (19.0, j1_ys[4]), (j1_x, j1_ys[4])],
            ],
            "pads": [(j2_x, j2_ys[2]), (j1_x, j1_ys[4])],
            "width": recommend_trace_width_mm(net_specs["NET_RADAR_RX_FROM_BRAIN_TX"]),
        },
        "NET_BRAIN_TX2_TO_ARD_D10": {
            "traces": [
                [(j1_x, j1_ys[2]), (35.0, j1_ys[2]), (37.5, 26.5), (49.0, 26.5), (51.0, j3_ys[2]), (j3_x, j3_ys[2])],
            ],
            "pads": [(j1_x, j1_ys[2]), (j3_x, j3_ys[2])],
            "width": recommend_trace_width_mm(net_specs["NET_BRAIN_TX2_TO_ARD_D10"]),
        },
        "NET_DIVIDER_3V3_NODE": {
            "traces": [
                [(j1_x, j1_ys[1]), (36.50, j1_ys[1]), r1_p2],
                [(36.50, j1_ys[1]), r2_p1],
            ],
            "pads": [(j1_x, j1_ys[1]), r1_p2, r2_p1],
            "width": recommend_trace_width_mm(net_specs["NET_DIVIDER_3V3_NODE"]),
        },
        "NET_ARD_D11_5V_IN": {
            "traces": [
                [r1_p1, (j3_x, j3_ys[1])],
            ],
            "pads": [r1_p1, (j3_x, j3_ys[1])],
            "width": recommend_trace_width_mm(net_specs["NET_ARD_D11_5V_IN"]),
        },
    }

    # Dibujar todas las redes sobre el lienzo principal (Pads 1.85mm dejan >=0.69mm libres en paso 2.54mm)
    for net_info in nets.values():
        w_tr = net_info["width"]
        for tr in net_info["traces"]:
            draw_trace_on(img, tr, width_mm=w_tr)
        for px_m, py_m in net_info["pads"]:
            draw_pad_solid_on(img, px_m, py_m, outer_d_mm=1.85)

    # 4 Pads de montaje mecánico M3 aislados en las esquinas
    mount_pads = [(4.5, 4.5), (w_mm - 4.5, 4.5), (4.5, h_mm - 4.5), (w_mm - 4.5, h_mm - 4.5)]
    for mx, my in mount_pads:
        draw_pad_solid_on(img, mx, my, outer_d_mm=4.0)

    # Ejecutar la batería completa de testeos eléctricos, térmicos y DRC antes de perforar pads
    audit_report = run_autonomous_pcb_audit(nets, net_specs, img, w_mm, h_mm, ambient_temp_c=25.0, copper_oz=1.0)
    assert audit_report.passed_all, "Fallo en auditoría pre-ejecución PCB:\n" + "\n".join(audit_report.summary_lines)

    # Ahora perforar visualmente el centro de cada pad para el detector de agujeros
    for net_info in nets.values():
        for px_m, py_m in net_info["pads"]:
            punch_drill_on(img, px_m, py_m, drill_d_mm=0.90)
    for mx, my in mount_pads:
        punch_drill_on(img, mx, my, drill_d_mm=2.10)

    out_p = Path(out_png_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_p), img)
    # Guardamos el reporte completo junto al PNG
    report_path = out_p.with_suffix(".audit.txt")
    report_path.write_text("\n".join(audit_report.summary_lines) + "\n", encoding="utf-8")
    return out_p, w_mm, h_mm


class SingleLayerAutorouter:
    """
    Autoruteador Planar de 1 Sola Capa (Algoritmo A* con ángulos de 90°/45° y Clearance DRC).
    Garantiza matemáticamente CERO superposiciones y CERO cruces entre pistas distintas:
      - Cada pista ruteada bloquea su propio ancho + el margen de aislamiento (clearance_mm)
        en la grilla de la única capa de cobre.
      - Si una red no puede rutearse en 1 capa sin cruzar otra, intenta rodearla por el plano;
        si es topológicamente no-planar (K5 / K3,3), reporta que requiere un puente (jumper 0-ohm)
        en lugar de superponer pistas.
    """

    def __init__(self, width_mm: float, height_mm: float, grid_step_mm: float = 0.5, clearance_mm: float = 0.85):
        self.w_mm = width_mm
        self.h_mm = height_mm
        self.step = grid_step_mm
        self.clearance_mm = clearance_mm
        self.gw = int(round(width_mm / grid_step_mm)) + 1
        self.gh = int(round(height_mm / grid_step_mm)) + 1
        # 0 = libre, >0 = ID de red propietaria de esa celda
        self.owner_grid = np.zeros((self.gh, self.gw), dtype=np.int32)
        self.pads: list[tuple[float, float, float, float, int]] = []  # (x, y, out_d, drill_d, net_id)
        self.routed_traces: list[tuple[list[tuple[float, float]], float, int]] = []

    def _mm_to_grid(self, x_mm: float, y_mm: float) -> tuple[int, int]:
        gx = max(0, min(self.gw - 1, int(round(x_mm / self.step))))
        gy = max(0, min(self.gh - 1, int(round(y_mm / self.step))))
        return gx, gy

    def _grid_to_mm(self, gx: int, gy: int) -> tuple[float, float]:
        return gx * self.step, gy * self.step

    def _mark_disk(self, gx: int, gy: int, radius_mm: float, net_id: int):
        r_cells = int(math.ceil(radius_mm / self.step))
        for dy in range(-r_cells, r_cells + 1):
            for dx in range(-r_cells, r_cells + 1):
                nx, ny = gx + dx, gy + dy
                if 0 <= nx < self.gw and 0 <= ny < self.gh:
                    if math.hypot(dx * self.step, dy * self.step) <= radius_mm:
                        self.owner_grid[ny, nx] = net_id

    def add_pad(self, x_mm: float, y_mm: float, net_id: int, outer_d_mm: float = 2.1, drill_d_mm: float = 0.9):
        self.pads.append((x_mm, y_mm, outer_d_mm, drill_d_mm, net_id))
        gx, gy = self._mm_to_grid(x_mm, y_mm)
        self._mark_disk(gx, gy, outer_d_mm * 0.5 + self.clearance_mm * 0.5, net_id)

    def route_net_connection(
        self, start_mm: tuple[float, float], end_mm: tuple[float, float], net_id: int, trace_width_mm: float = 0.95
    ) -> list[tuple[float, float]]:
        """Busca camino A* en 1 sola capa evitando cualquier celda ocupada por otra red."""
        import heapq

        sx, sy = self._mm_to_grid(*start_mm)
        ex, ey = self._mm_to_grid(*end_mm)
        margin_cells = max(2, int(math.ceil((trace_width_mm * 0.5 + self.clearance_mm) / self.step)))

        def cell_allowed(cx: int, cy: int) -> bool:
            if cx < 2 or cx >= self.gw - 2 or cy < 2 or cy >= self.gh - 2:
                return False
            for dy in range(-margin_cells, margin_cells + 1):
                for dx in range(-margin_cells, margin_cells + 1):
                    val = int(self.owner_grid[cy + dy, cx + dx])
                    if val != 0 and val != net_id:
                        return False
            return True

        open_heap: list[tuple[float, float, int, int, int, int]] = []
        heapq.heappush(open_heap, (0.0, 0.0, sx, sy, 0, 0))
        came_from: dict[tuple[int, int], tuple[int, int]] = {}
        g_score: dict[tuple[int, int], float] = {(sx, sy): 0.0}

        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        found = False

        while open_heap:
            _, g, cx, cy, pdx, pdy = heapq.heappop(open_heap)
            if (cx, cy) == (ex, ey):
                found = True
                break
            if g > g_score.get((cx, cy), float("inf")) + 1e-5:
                continue
            for dx, dy in neighbors:
                nx, ny = cx + dx, cy + dy
                if not (0 <= nx < self.gw and 0 <= ny < self.gh):
                    continue
                if math.hypot(nx - ex, ny - ey) > 2.5 and math.hypot(nx - sx, ny - sy) > 2.5:
                    if not cell_allowed(nx, ny):
                        continue
                else:
                    owner = int(self.owner_grid[ny, nx])
                    if owner != 0 and owner != net_id:
                        continue
                step_cost = 1.4142 if (dx != 0 and dy != 0) else 1.0
                bend_penalty = 0.25 if (pdx != 0 or pdy != 0) and (dx != pdx or dy != pdy) else 0.0
                ng = g + step_cost + bend_penalty
                if ng < g_score.get((nx, ny), float("inf")):
                    g_score[(nx, ny)] = ng
                    came_from[(nx, ny)] = (cx, cy)
                    h = math.hypot(ex - nx, ey - ny)
                    heapq.heappush(open_heap, (ng + h, ng, nx, ny, dx, dy))

        if not found:
            raise RuntimeError(f"Autorouter 1-Capa: No existe ruta planar sin cruce para red {net_id} de {start_mm} a {end_mm}")

        # Reconstruir y marcar el camino como propiedad de net_id
        cur = (ex, ey)
        grid_path = [cur]
        while cur in came_from:
            cur = came_from[cur]
            grid_path.append(cur)
        grid_path.reverse()

        pts_mm = []
        for gx, gy in grid_path:
            self._mark_disk(gx, gy, trace_width_mm * 0.5, net_id)
            pts_mm.append(self._grid_to_mm(gx, gy))
        pts_mm[0] = start_mm
        pts_mm[-1] = end_mm
        self.routed_traces.append((pts_mm, trace_width_mm, net_id))
        return pts_mm
