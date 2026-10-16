import json
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from dronesim.api.app import ProcessRequest, create_app, pipeline_commands

BYTES = bytes(range(256)) * 4


def wait_job(client, job_id, timeout=10.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = client.get(f"/api/jobs/{job_id}").json()
        if j["status"] != "running":
            return j
        time.sleep(0.02)
    raise AssertionError("job did not finish")


def fake_pipeline(calls):
    def run(video, run_dir, opts, report):
        calls.append((video.name, dict(opts)))
        run_dir.mkdir(parents=True, exist_ok=True)
        for i, label in enumerate(["detect", "reconstruct", "simulate"]):
            report(label, i, 3, f"{label} ok")
        for f in ("obstacle_track.json", "obstacle_metric.json", "sim_recording.json"):
            (run_dir / f).write_text(json.dumps({"schema": "x", "video": {"path": str(video)}}))
    return run


@pytest.fixture()
def env(tmp_path):
    data, vids = tmp_path / "runs", tmp_path / "videos"
    data.mkdir()
    vids.mkdir()
    for n in ("alpha.mp4", "beta.mp4", "Clip_3.MOV"):
        (vids / n).write_bytes(BYTES)
    (vids / "notes.txt").write_text("not a video")
    (vids / "bad name.mp4").write_bytes(BYTES)          # space: not a valid run name, must not be listed
    (vids / "sub").mkdir()
    (vids / "sub" / "hidden.mp4").write_bytes(BYTES)    # subfolders are not scanned
    # alpha is already processed; "orphan" has results but its video file is gone
    for run in ("alpha", "orphan"):
        (data / run).mkdir()
        for f in ("obstacle_track.json", "obstacle_metric.json", "sim_recording.json"):
            (data / run / f).write_text("{}")
    calls = []
    return TestClient(create_app(data, [vids], frontend_dist=tmp_path / "none", pipeline_runner=fake_pipeline(calls))), data, vids, calls


# ------------------------------------------------------------------ listing
def test_lists_every_video_with_processed_status_and_orphan_runs(env):
    c, *_ = env
    items = {i["name"]: i for i in c.get("/api/videos").json()["videos"]}
    assert set(items) == {"alpha", "beta", "Clip_3", "orphan"}          # no txt, no bad name, no subfolder
    assert items["alpha"]["recording"] is True and items["alpha"]["file"] == "alpha.mp4"
    assert items["beta"]["recording"] is False and items["beta"]["track"] is False and items["beta"]["file"] == "beta.mp4"
    assert items["Clip_3"]["file"] == "Clip_3.MOV" and items["Clip_3"]["size_mb"] is not None
    assert items["orphan"]["file"] is None and items["orphan"]["recording"] is True
    assert [i["name"] for i in c.get("/api/videos").json()["videos"]] == sorted(items)


def test_video_file_preview_is_served_with_range_and_only_for_listed_videos(env):
    c, *_ = env
    r = c.get("/api/videos/beta.mp4/file", headers={"Range": "bytes=10-19"})
    assert r.status_code == 206 and r.content == BYTES[10:20]
    for bad in ("nope.mp4", "notes.txt", "bad name.mp4", "..%2Fsecret.mp4", "sub%2Fhidden.mp4", "%2e%2e%2f%2e%2e%2fetc"):
        assert c.get(f"/api/videos/{bad}/file").status_code == 404, bad


# ------------------------------------------------------------------ processing job
def test_process_runs_the_pipeline_in_the_background_and_reports_progress(env):
    c, data, _, calls = env
    r = c.post("/api/videos/beta.mp4/process", json={"width_m": 0.6, "height_m": 0.55, "hfov_deg": 70})
    assert r.status_code == 202
    job = wait_job(c, r.json()["id"])
    assert job["status"] == "done" and job["run"] == "beta" and job["step_index"] == 3 and job["error"] is None
    assert any("simulate ok" in line for line in job["log"])
    assert calls == [("beta.mp4", {"width_m": 0.6, "height_m": 0.55, "hfov_deg": 70, "conf": None})]
    assert (data / "beta" / "sim_recording.json").is_file()
    items = {i["name"]: i for i in c.get("/api/videos").json()["videos"]}
    assert items["beta"]["recording"] is True                          # now listed as processed


def test_failure_is_reported_with_the_reason(tmp_path):
    def boom(video, run_dir, opts, report):
        report("detect", 0, 3, "")
        raise RuntimeError("Detecting failed: No detections found")
    data, vids = tmp_path / "runs", tmp_path / "videos"
    data.mkdir(); vids.mkdir()
    (vids / "v.mp4").write_bytes(BYTES)
    c = TestClient(create_app(data, [vids], frontend_dist=tmp_path / "none", pipeline_runner=boom))
    job = wait_job(c, c.post("/api/videos/v.mp4/process", json={}).json()["id"])
    assert job["status"] == "error" and "No detections found" in job["error"]


def test_pipeline_that_produces_no_recording_is_an_error(tmp_path):
    data, vids = tmp_path / "runs", tmp_path / "videos"
    data.mkdir(); vids.mkdir()
    (vids / "v.mp4").write_bytes(BYTES)
    c = TestClient(create_app(data, [vids], frontend_dist=tmp_path / "none", pipeline_runner=lambda *a: None))
    job = wait_job(c, c.post("/api/videos/v.mp4/process", json={}).json()["id"])
    assert job["status"] == "error" and "no recording" in job["error"]


def test_already_processed_needs_force_and_only_one_job_runs_at_a_time(tmp_path):
    gate = threading.Event()

    def slow(video, run_dir, opts, report):
        gate.wait(5)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "sim_recording.json").write_text("{}")
    data, vids = tmp_path / "runs", tmp_path / "videos"
    data.mkdir(); vids.mkdir()
    for n in ("a.mp4", "b.mp4"):
        (vids / n).write_bytes(BYTES)
    c = TestClient(create_app(data, [vids], frontend_dist=tmp_path / "none", pipeline_runner=slow))
    first = c.post("/api/videos/a.mp4/process", json={})
    assert first.status_code == 202
    busy = c.post("/api/videos/b.mp4/process", json={})
    assert busy.status_code == 409 and "Another video" in busy.json()["detail"]
    gate.set()
    assert wait_job(c, first.json()["id"])["status"] == "done"
    again = c.post("/api/videos/a.mp4/process", json={})
    assert again.status_code == 409 and "already processed" in again.json()["detail"]
    assert c.post("/api/videos/a.mp4/process", json={"force": True}).status_code == 202


def test_process_validates_input_and_names(env):
    c, *_ = env
    assert c.post("/api/videos/nope.mp4/process", json={}).status_code == 404
    assert c.post("/api/videos/..%2Fx.mp4/process", json={}).status_code == 404
    for bad in ({"width_m": -1}, {"width_m": 0}, {"height_m": 999}, {"hfov_deg": 5}, {"hfov_deg": 200}, {"conf": 0.0}, {"conf": 1.5}, {"width_m": "wide"}):
        assert c.post("/api/videos/beta.mp4/process", json=bad).status_code == 422, bad
    assert c.get("/api/jobs/doesnotexist").status_code == 404


# ------------------------------------------------------------------ the real pipeline commands
def test_real_pipeline_commands_use_the_three_cli_steps_and_pass_the_real_size():
    run_dir = Path("data/runs/clip")
    steps = pipeline_commands(Path("clip.mp4"), run_dir, ProcessRequest(width_m=0.6, height_m=0.5, hfov_deg=75, conf=0.4).model_dump(exclude={"force"}))
    labels = [s[0] for s in steps]
    assert len(steps) == 3 and "YOLO" in labels[0] and "Reconstructing" in labels[1] and "Simulating" in labels[2]
    export = steps[0][1]
    assert export[1:3] == ["-m", "dronesim.vision.export"] and export[3] == "clip.mp4"
    for flag, val in (("--width-m", 0.6), ("--height-m", 0.5), ("--hfov-deg", 75), ("--conf", 0.4)):
        assert float(export[export.index(flag) + 1]) == val
    assert export[export.index("--out") + 1] == str(run_dir)
    assert steps[1][1][2] == "dronesim.vision.reconstruction" and steps[1][1][3].endswith("obstacle_track.json")
    assert steps[2][1][2] == "dronesim.run_scenario" and steps[2][1][3].endswith("obstacle_metric.json")
    bare = pipeline_commands(Path("clip.mp4"), run_dir, ProcessRequest().model_dump(exclude={"force"}))[0][1]
    assert "--width-m" not in bare and "--hfov-deg" not in bare            # unset = the documented assumed defaults
