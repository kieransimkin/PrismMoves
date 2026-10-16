import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from prismmoves.backend import PixelCueBackend, validate_profile
from prismmoves.cli import main
from prismmoves.report import build_report, html_report, markdown, svg_summary


def profile():
    return {"schema": "pixelcue.visual-analysis/1", "source": {"sha256": "f"*64,
        "dimensions": [32, 32], "analysis_dimensions": [32, 32], "color_handling": "sRGB",
        "background": "#ffffff", "notes": []}, "measurements": {
        "colour": {"palette": [{"hex": "#ff0000", "sample_fraction": 1}],
            "mean_saturation": 1, "relative_luminance_percentiles": [.2126]*5},
        "texture": {"grayscale_entropy_bits": 0},
        "composition": {"edge_density_above_0_08": 0, "horizontal_mirror_mae": 0,
            "vertical_mirror_mae": 0, "grid": [{"column": 0, "row": 0, "mean_luminance": .2126}]}},
        "models": [], "limits": ["Synthetic protocol fixture, not a real model result."]}


@pytest.mark.parametrize("url", ["https://example.org", "http://example.org", "http://127.0.0.1:8765/path",
    "http://user:pass@localhost", "http://localhost?secret=1"])
def test_only_explicit_loopback_service(url):
    with pytest.raises(ValueError):
        PixelCueBackend(url)


def test_schema_validation():
    with pytest.raises(ValueError):
        validate_profile({"schema": "other"})
    with pytest.raises(ValueError):
        validate_profile({"schema": "pixelcue.visual-analysis/1"})


def test_reports_escape_untrusted_text_and_preserve_unknowns():
    p = profile()
    p["models"] = [{"name": "blip", "status": "completed", "result": {"caption": "<script>alert(1)</script>"}}]
    report = build_report([{"id": "A01", "name": "<script>.png", "role": "unknown", "lineage": "unknown",
        "status": "completed", "analysis": p}], {"mode": "fixture"})
    text = markdown(report)
    output = html_report(report, text)
    assert "<script>" not in output
    assert "&lt;script&gt;" in output
    assert "Caption hypothesis" in text
    assert "core invariants" in text
    assert "No semantic" not in text
    import xml.etree.ElementTree as ET
    graphic = svg_summary(report)
    ET.fromstring(graphic)
    assert '<script>' not in graphic
    assert 'visual-summary.svg' in text


def test_http_cli_calls_pixelcue_and_writes_durable_report(tmp_path):
    calls = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps({"service": "pixelcue", "version": "fixture",
                "visual_analysis": {"route": "/v1/analysis"}}).encode())
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append((self.path, payload))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps(profile()).encode())
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        out = tmp_path/"report"
        result = main([str(tmp_path/"one.png"), "--out", str(out), "--pixelcue-url",
            f"http://127.0.0.1:{server.server_port}"])
        assert result == 0
        assert calls[0][0] == "/v1/analysis"
        assert calls[0][1]["allow_download"] is False
        saved = json.loads((out/"analysis.json").read_text(encoding="utf-8"))
        assert saved["items"][0]["analysis"]["source"]["sha256"] == "f"*64
        assert all((out/name).is_file() for name in ("analysis.json", "report.html", "report.md", "SHA256SUMS.txt"))
        with pytest.raises(SystemExit):
            main(["one.png", "--out", str(out)])
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_partial_failures_remain_in_ledger(tmp_path, monkeypatch):
    monkeypatch.setattr(PixelCueBackend, "identity", lambda self: {"mode": "fixture"})
    def unavailable(self, *args, **kwargs):
        raise FileNotFoundError("Missing input")
    monkeypatch.setattr(PixelCueBackend, "analyse", unavailable)
    out = tmp_path/"partial"
    assert main(["missing.png", "--out", str(out)]) == 2
    report = json.loads((out/"analysis.json").read_text())
    assert report["items"][0]["status"] == "unavailable"
    assert "Missing input" in (out/"report.md").read_text()


def test_unavailable_model_has_nonzero_exit_and_retained_measurements(tmp_path, monkeypatch):
    monkeypatch.setattr(PixelCueBackend, "identity", lambda self: {"mode": "fixture"})
    p = profile()
    p["models"] = [{"name": "clip", "status": "unavailable", "reason": "No cached weights"}]
    monkeypatch.setattr(PixelCueBackend, "analyse", lambda *a, **kw: p)
    out = tmp_path/"models"
    assert main(["one.png", "--out", str(out), "--models", "clip"]) == 2
    assert "No cached weights" in (out/"report.md").read_text()


def test_model_paths_must_name_selected_models(tmp_path):
    assert main(["one.png", "--out", str(tmp_path/"unused"), "--model-path", "wrong=path"]) == 1
