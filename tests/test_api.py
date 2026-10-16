import json
import os
import subprocess
import sys

import numpy as np
import pytest
from fastapi.testclient import TestClient

from dronesim.api.app import create_app
from dronesim.sim.params import DroneParams
from dronesim.sim.recording import build_recording
from dronesim.sim.runner import RunConfig, run_avoidance

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIDEO_BYTES = bytes(range(256)) * 8          # 2048 known bytes


def scenario(speed=1.5, range_m=2.7):
    return {"approach_speed_mps": speed, "reference_time_s": 0.5,
            "drone": {"position_m": [0, 0, 0], "velocity_mps": [speed, 0, 0]},
            "obstacle": {"center_m": [range_m, 0.0, 0.0], "width_m": 0.5, "height_m": 0.5,
                         "thickness_m": 0.05, "normal": [-1, 0, 0], "static": True},
            "assumptions": {"size_is_assumed": True, "fov_is_assumed": True}}


@pytest.fixture()
def env(tmp_path):
    data, vids = tmp_path / "runs", tmp_path / "videos"
    (data / "demo").mkdir(parents=True)
    (data / "bare").mkdir()
    (data / "nomotion").mkdir()
    vids.mkdir()
    (vids / "clip.mp4").write_bytes(VIDEO_BYTES)
    (tmp_path / "secret.mp4").write_bytes(b"TOP SECRET")
    (data / "demo" / "obstacle_track.json").write_text(
        json.dumps({"video": {"path": "C:\\some\\where\\clip.mp4"}, "samples": [{"frame": 0}]}))
    (data / "demo" / "obstacle_metric.json").write_text(json.dumps({"scenario": scenario(), "samples": []}))
    (data / "nomotion" / "obstacle_metric.json").write_text(json.dumps({"scenario": None, "samples": []}))
    (data / "bare" / "obstacle_track.json").write_text(json.dumps({"video": {"path": "../secret.mp4"}}))
    # directories that EXIST and hold real files, so only the API's own guards can keep them private
    (data / ".hidden").mkdir()
    (data / ".hidden" / "obstacle_track.json").write_text(json.dumps({"video": {"path": "x.mp4"}, "secret": 1}))
    (tmp_path / "outside").mkdir()
    (tmp_path / "outside" / "obstacle_track.json").write_text(json.dumps({"video": {"path": "x.mp4"}, "secret": 2}))
    made = subprocess.run(["cmd", "/c", "mklink", "/J", str(data / "escape"), str(tmp_path / "outside")],
                          capture_output=True, text=True) if os.name == "nt" else None
    if os.name != "nt":
        os.symlink(tmp_path / "outside", data / "escape")
    return TestClient(create_app(data, [vids], frontend_dist=tmp_path / "none")), data, tmp_path


# ------------------------------------------------------------------ listing / files
def test_health_and_run_listing(env):
    c, data, _ = env
    assert c.get("/api/health").json()["status"] == "ok"
    runs = {r["name"]: r for r in c.get("/api/runs").json()["runs"]}
    assert set(runs) == {"demo", "bare", "nomotion"}
    assert runs["demo"] == {"name": "demo", "track": True, "metric": True, "recording": False, "video": True}
    assert runs["bare"]["video"] is False                       # stored path points outside: not served


def test_serves_json_files_verbatim(env):
    c, data, _ = env
    assert c.get("/api/runs/demo/track").json() == json.loads((data / "demo" / "obstacle_track.json").read_text())
    assert c.get("/api/runs/demo/metric").json()["scenario"]["approach_speed_mps"] == 1.5


def test_missing_files_give_404_with_a_hint(env):
    c, _, _ = env
    r = c.get("/api/runs/demo/recording")
    assert r.status_code == 404 and "run_scenario" in r.json()["detail"]
    r = c.get("/api/runs/bare/metric")
    assert r.status_code == 404 and "reconstruction" in r.json()["detail"]


def test_unknown_and_malicious_run_names_are_rejected(env):
    c, _, tmp = env
    for bad in ("nope", "..", "%2e%2e", "demo%2f..%2f..", "..%5c..", ".hidden", "a" * 100):
        assert c.get(f"/api/runs/{bad}/track").status_code in (404, 400), bad
    assert c.get("/api/runs/..%2Fsecret/video").status_code in (404, 400)


def test_guards_hold_for_real_directories_that_should_stay_private(env):
    c, data, _ = env
    assert (data / ".hidden" / "obstacle_track.json").is_file()
    assert c.get("/api/runs/.hidden/track").status_code == 404          # name rule (leading dot)
    if (data / "escape").exists():                                      # junction/symlink out of the data dir
        assert c.get("/api/runs/escape/track").status_code == 404        # containment rule
        listing = c.get("/api/runs")
        assert listing.status_code == 200                                # one bad entry must not break the list
        assert "escape" not in [r["name"] for r in listing.json()["runs"]]
    else:
        pytest.skip("could not create a junction/symlink on this system")


# ------------------------------------------------------------------ video
def test_video_is_served_and_supports_range_requests(env):
    c, _, _ = env
    r = c.get("/api/runs/demo/video")
    assert r.status_code == 200 and r.content == VIDEO_BYTES and r.headers["content-type"].startswith("video/")
    part = c.get("/api/runs/demo/video", headers={"Range": "bytes=100-199"})
    assert part.status_code == 206 and part.content == VIDEO_BYTES[100:200]
    assert part.headers["content-range"] == f"bytes 100-199/{len(VIDEO_BYTES)}"


def test_video_path_traversal_via_stored_path_is_neutralised(env):
    c, _, _ = env
    r = c.get("/api/runs/bare/video")        # track says "../secret.mp4"; secret.mp4 exists one level up
    assert r.status_code == 404 and b"SECRET" not in r.content


# ------------------------------------------------------------------ simulate
def test_simulate_returns_a_valid_recording_matching_direct_run(env):
    c, data, _ = env
    r = c.post("/api/runs/demo/simulate", json={})
    assert r.status_code == 200
    rec = r.json()
    assert rec["schema"] == "sim_recording/v1" and rec["summary"]["collided"] is False
    assert rec["summary"]["min_clearance_m"] >= rec["environment"]["safety_margin_m"]
    cfg = RunConfig()
    direct = build_recording(run_avoidance(scenario(), DroneParams(), cfg), DroneParams(), cfg,
                             source={"metric_json": "obstacle_metric.json"})
    assert rec == json.loads(json.dumps(direct))               # API result == running the library directly
    assert not (data / "demo" / "sim_recording.json").exists()  # not saved unless asked


def test_simulate_options_change_the_result(env):
    c, _, _ = env
    left = c.post("/api/runs/demo/simulate", json={"dodge": "left"}).json()
    right = c.post("/api/runs/demo/simulate", json={"dodge": "right"}).json()
    assert left["planning"]["plan"]["params"]["direction_deg"] == 0
    assert right["planning"]["plan"]["params"]["direction_deg"] == 180
    yl, yr = np.array(left["flight"]["position_m"])[:, 1], np.array(right["flight"]["position_m"])[:, 1]
    assert yl.max() > 0.5 and yr.min() < -0.5                   # actually flew to opposite sides
    wide = c.post("/api/runs/demo/simulate", json={"dodge": "left", "margin_m": 0.6}).json()
    assert wide["planning"]["plan"]["params"]["offset_m"] > left["planning"]["plan"]["params"]["offset_m"]


def test_simulate_save_then_recording_endpoint_returns_it(env):
    c, data, _ = env
    rec = c.post("/api/runs/demo/simulate", json={"save": True}).json()
    assert (data / "demo" / "sim_recording.json").is_file()
    assert c.get("/api/runs/demo/recording").json() == rec
    assert {r["name"]: r for r in c.get("/api/runs").json()["runs"]}["demo"]["recording"] is True


def test_unavoidable_is_a_normal_response_not_an_error(env):
    c, _, _ = env
    r = c.post("/api/runs/demo/simulate", json={"speed_mps": 8.0})
    assert r.status_code == 200
    rec = r.json()
    assert rec["flight"] is None and "UNAVOIDABLE" in rec["planning"]["note"]


def test_simulate_validates_input(env):
    c, _, _ = env
    for bad in ({"speed_mps": -1}, {"speed_mps": 100}, {"dodge": "sideways"}, {"margin_m": -0.1},
                {"altitude_m": 0.0}, {"speed_mps": "fast"}):
        assert c.post("/api/runs/demo/simulate", json=bad).status_code == 422, bad
    assert c.post("/api/runs/nomotion/simulate", json={}).status_code == 400
    assert c.post("/api/runs/nope/simulate", json={}).status_code == 404
    assert c.post("/api/runs/bare/simulate", json={}).status_code == 404     # no metric file


# ------------------------------------------------------------------ misc
def test_cors_allows_the_dev_frontend_only(env):
    c, _, _ = env
    ok = c.get("/api/health", headers={"Origin": "http://localhost:5173"})
    assert ok.headers.get("access-control-allow-origin") == "http://localhost:5173"
    bad = c.get("/api/health", headers={"Origin": "http://evil.example"})
    assert "access-control-allow-origin" not in bad.headers


def test_api_imports_no_yolo_opencv_or_torch():
    code = ("import sys; sys.path.insert(0, %r); import dronesim.api.app; "
            "bad = [m for m in ('ultralytics','cv2','torch') if m in sys.modules]; assert not bad, bad; print('ok')" % ROOT)
    out = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True)
    assert out.returncode == 0 and "ok" in out.stdout, out.stderr


def test_default_app_serves_the_real_run_if_present():
    from dronesim.api.app import app
    c = TestClient(app)
    names = [r["name"] for r in c.get("/api/runs").json()["runs"]]
    if "testv2" in names:
        r = c.get("/api/runs/testv2/recording")
        assert r.status_code == 200 and r.json()["schema"] == "sim_recording/v1"
        assert c.get("/api/runs/testv2/video").status_code == 200
