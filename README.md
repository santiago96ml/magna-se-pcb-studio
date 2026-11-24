# Magna SE Mini — Laserless PCB & Circuit Studio

Turn an entry-level FDM 3D printer (**Hellbot Magna SE Mini**, Ender-3 style Marlin G-code) into a **laserless DIY PCB & engraving station** with built-in **IPC-2221 electrical/thermal simulation**, **4-corner bilinear Z-height auto-leveling**, and **OpenCV camera AOI inspection**.

## Features

- **Three Laserless PCB Modes (`gcode_generator.py`)**:
  - `pcb_marker_resist`: Plots etch-resist traces and filled copper pads with an indelible permanent marker (e.g. `0.75 mm` / `0.4 mm` felt-tip) ready for Ferric Chloride (`FeCl3`) etching.
  - `pcb_isolation`: Multi-pass concentric isolation routing (`1..4` passes) around copper traces and pads.
  - `pcb_drill_holes`: Automated center-punch / THT drilling G-code for `2.54 mm` headers and mounting pads.
  - `relief_pocket`: Full-area copper/wood relief pocketing & halftone woodcut engraving.
- **4-Corner Bilinear Z Auto-Leveling**: Probes/sets `BL`, `BR`, `TL`, `TR` corners over blank FR4 / Pertinax copper clad and interpolates Z across every G-code segment to compensate for warped beds at micron precision.
- **IPC-2221 Electrical & Thermal Engineering Suite (`pcb_engineering_engine.py`)**:
  - Calculates trace widths per current/copper weight (`1 oz`), peak-voltage clearances, trace DC resistance, IR drop, Joule heating, and final copper temperature.
  - Runs 5 pre-execution checks (Single-layer DRC island continuity, Thermal rise, IR voltage drop, Euclidean clearance, 3.3V/5V ERC).
- **Webcam AOI Inspection (`inspect_pcb_with_camera_aoi`)**: Homography-aligns a webcam photo of the plotted/etched board against the reference image to flag shorts or broken traces.
- **Single-Layer A\* Planar Autorouter (`SingleLayerAutorouter`)**: Enforces DRC clearance and prevents overlapping traces without jumpers.
- **3D-Printable Spring Marker Holder**: STL included in [`trabajos/soporte_fibra_magna_se.stl`](./trabajos/soporte_fibra_magna_se.stl).

## Quick Start

```bash
pip install -r requirements.txt
python magna_studio.py
```

Or on Windows, double-click **`ABRIR_MAGNA_STUDIO.bat`**.

*(Requires Python with `tkinter` for the desktop GUI; `gcode_generator.py` and `pcb_engineering_engine.py` also work headlessly.)*

## Running Tests

```bash
pip install -r requirements.txt pytest
pytest -v
```

## License

MIT — see [LICENSE](./LICENSE).
