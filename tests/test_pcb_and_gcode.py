from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gcode_generator import (
    SingleLayerAutorouter,
    XiloConfig,
    create_tank_carrier_pcb_layout,
    generate_woodcut_gcode,
)
from pcb_engineering_engine import (
    NetSpec,
    ipc2221_min_clearance_mm,
    ipc2221_min_trace_width_mm,
    recommend_trace_width_mm,
)


def test_ipc2221_calculations():
    w_sig = ipc2221_min_trace_width_mm(current_a=0.1, temp_rise_c=10.0)
    w_pwr = ipc2221_min_trace_width_mm(current_a=2.0, temp_rise_c=10.0)
    assert w_pwr > w_sig > 0.0
    assert ipc2221_min_clearance_mm(5.0) >= 0.35
    rec = recommend_trace_width_mm(NetSpec("PWR", voltage_v=5.0, current_max_a=1.5))
    assert rec >= 1.40


def test_single_layer_autorouter():
    router = SingleLayerAutorouter(width_mm=50.0, height_mm=40.0, grid_step_mm=1.0, clearance_mm=1.0)
    router.add_pad(5.0, 5.0, net_id=1)
    router.add_pad(45.0, 35.0, net_id=1)
    pts = router.route_net_connection((5.0, 5.0), (45.0, 35.0), net_id=1, trace_width_mm=1.0)
    assert len(pts) >= 2
    assert pts[0] == (5.0, 5.0)
    assert pts[-1] == (45.0, 35.0)


def test_tank_carrier_pcb_audit_and_gcode_generation():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        pcb_png = tmp_dir / "carrier.png"
        out_png, w_mm, h_mm = create_tank_carrier_pcb_layout(pcb_png)
        assert out_png.exists()
        assert w_mm == 65.0 and h_mm == 45.0
        assert (tmp_dir / "carrier.audit.txt").exists()

        for mode in ("pcb_marker_resist", "pcb_isolation", "pcb_drill_holes"):
            gcode_out = tmp_dir / f"carrier_{mode}.gcode"
            preview_out = tmp_dir / f"carrier_{mode}.png"
            cfg = XiloConfig(
                width_mm=w_mm,
                mode=mode,
                z_corner_bl=0.0,
                z_corner_br=0.02,
                z_corner_tl=-0.01,
                z_corner_tr=0.01,
            )
            stats = generate_woodcut_gcode(out_png, gcode_out, cfg, out_preview_path=preview_out)
            assert gcode_out.exists()
            assert preview_out.exists()
            assert stats.total_lines > 20
            content = gcode_out.read_text(encoding="utf-8")
            assert "M107" in content and "G21" in content
