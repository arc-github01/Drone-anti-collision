"""Backend API: serves processed runs (vision data, simulation recordings, source video) and
re-simulates with new settings WITHOUT re-running YOLO.

Run:  python -m dronesim.api            (http://127.0.0.1:8000, docs at /docs)

Layout it expects:  <data_dir>/<run>/obstacle_track.json, obstacle_metric.json, sim_recording.json
The source video is found by the file name stored in obstacle_track.json, inside `video_dirs`.
This module deliberately imports no YOLO/OpenCV/torch.
"""
import json
import os
import re
import subprocess
import sys
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from dronesim.sim.params import DroneParams
from dronesim.sim.recording import build_recording, load_recording, save_recording
from dronesim.sim.runner import RunConfig, run_avoidance
from dronesim.sim.wind import WindConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
FILES = {"track": "obstacle_track.json", "metric": "obstacle_metric.json", "recording": "sim_recording.json"}
DODGE_DEG = {"left": (0,), "up": (90,), "right": (180,), "down": (270,)}


class SimulateRequest(BaseModel):
    """Everything optional; omitted = the same defaults as `python -m dronesim.run_scenario`."""
    speed_mps: Optional[float] = Field(None, gt=0.05, le=15.0, description="override approach speed")
    dodge: Literal["auto", "left", "right", "up", "down"] = "auto"
    altitude_m: Optional[float] = Field(None, ge=0.2, le=100.0, description="start altitude above the floor")
    floor_m: Optional[float] = Field(None, ge=-100.0, le=100.0)
    no_floor: bool = False
    margin_m: Optional[float] = Field(None, ge=0.0, le=3.0, description="required clearance beyond the drone body")
    drone_radius_m: Optional[float] = Field(None, gt=0.0, le=1.0)
    mass_kg: Optional[float] = Field(None, gt=0.01, le=5.0, description="drone mass override (kg)")
    max_thrust_per_motor_n: Optional[float] = Field(None, gt=0.05, le=5.0, description="max thrust per motor override (N)")
    # Wind: direction is where the wind blows TOWARD. Azimuth from +x (forward/approach axis) toward +y (left),
    # elevation from the horizontal toward +z (up): 0/0 = tailwind, 180/0 = headwind, 90/0 = toward the left, 0/90 = updraft.
    wind_speed_mps: float = Field(0.0, ge=0.0, le=30.0)
    wind_azimuth_deg: float = Field(0.0, ge=-360.0, le=360.0)
    wind_elevation_deg: float = Field(0.0, ge=-90.0, le=90.0)
    turbulence_sigma_mps: float = Field(0.0, ge=0.0, le=5.0, description="per-axis gust standard deviation")
    turbulence_tau_s: float = Field(2.0, ge=0.2, le=30.0, description="gust correlation time")
    wind_seed: int = Field(0, ge=0, le=2**31 - 1)
    save: bool = Field(False, description="also write the result as the run's sim_recording.json")


VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


class ProcessRequest(BaseModel):
    """Optional settings for processing a video. Leave sizes empty to use the assumed defaults (0.5 m, 82.6 deg)."""
    width_m: Optional[float] = Field(None, gt=0.02, le=50.0, description="real width of the obstacle")
    height_m: Optional[float] = Field(None, gt=0.02, le=50.0, description="real height of the obstacle")
    hfov_deg: Optional[float] = Field(None, ge=20.0, le=150.0, description="camera horizontal field of view")
    conf: Optional[float] = Field(None, ge=0.05, le=0.95, description="YOLO confidence threshold")
    force: bool = Field(False, description="re-process even if a recording already exists")


def pipeline_commands(video: Path, run_dir: Path, opts: dict) -> List[tuple]:
    """The three command-line steps that turn a video into a recording: (label, argv)."""
    py = sys.executable
    export = [py, "-m", "dronesim.vision.export", str(video), "--out", str(run_dir)]
    for flag, key in (("--width-m", "width_m"), ("--height-m", "height_m"), ("--hfov-deg", "hfov_deg"), ("--conf", "conf")):
        if opts.get(key) is not None:
            export += [flag, str(opts[key])]
    return [
        ("Detecting and tracking objects (YOLO)", export),
        ("Reconstructing metric obstacle data", [py, "-m", "dronesim.vision.reconstruction", str(run_dir / FILES["track"])]),
        ("Simulating avoidance", [py, "-m", "dronesim.run_scenario", str(run_dir / FILES["metric"])]),
    ]


def default_pipeline_runner(video: Path, run_dir: Path, opts: dict, report: Callable[[str, int, int, str], None]) -> None:
    steps = pipeline_commands(video, run_dir, opts)
    run_dir.mkdir(parents=True, exist_ok=True)
    for i, (label, argv) in enumerate(steps):
        report(label, i, len(steps), "")
        r = subprocess.run(argv, cwd=str(REPO_ROOT), capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=1800)
        tail = "\n".join(((r.stdout or "") + (r.stderr or "")).strip().splitlines()[-6:])
        if r.returncode != 0:
            raise RuntimeError(f"{label} failed: {tail}")
        report(label, i, len(steps), tail)


@dataclass
class Job:
    id: str
    video: str
    run: str
    status: str = "running"            # running | done | error
    step: str = "Starting"
    step_index: int = 0
    n_steps: int = 3
    error: Optional[str] = None
    log: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"id": self.id, "video": self.video, "run": self.run, "status": self.status, "step": self.step,
                "step_index": self.step_index, "n_steps": self.n_steps, "error": self.error, "log": self.log[-12:]}


def create_app(data_dir: Optional[os.PathLike] = None, video_dirs: Optional[List[os.PathLike]] = None,
               frontend_dist: Optional[os.PathLike] = None, pipeline_runner: Optional[Callable] = None) -> FastAPI:
    data = Path(data_dir or REPO_ROOT / "data" / "runs").resolve()
    vdirs = [Path(d).resolve() for d in (video_dirs if video_dirs is not None else [REPO_ROOT])]
    app = FastAPI(title="Drone Navigator Telemetry - simulation API", version="0.1")
    app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                       allow_methods=["GET", "POST"], allow_headers=["*"])

    def run_dir(run: str) -> Path:
        if not RUN_NAME.match(run):
            raise HTTPException(404, f"Unknown run '{run}'")
        d = (data / run).resolve()
        if data not in d.parents or not d.is_dir():
            raise HTTPException(404, f"Unknown run '{run}'")
        return d

    def read_json(run: str, kind: str):
        p = run_dir(run) / FILES[kind]
        if not p.is_file():
            hint = {"track": f"python -m dronesim.vision.export <video> --out {data / run}",
                    "metric": f"python -m dronesim.vision.reconstruction {data / run / FILES['track']}",
                    "recording": f"python -m dronesim.run_scenario {data / run / FILES['metric']}"}[kind]
            raise HTTPException(404, f"{FILES[kind]} not found for run '{run}'. Create it with: {hint}")
        return p

    def video_path(run: str) -> Optional[Path]:
        p = run_dir(run) / FILES["track"]
        if not p.is_file():
            return None
        try:
            name = os.path.basename(json.loads(p.read_text(encoding="utf-8"))["video"]["path"].replace("\\", "/"))
        except (KeyError, ValueError, TypeError):
            return None
        if not name:
            return None
        for d in [run_dir(run), *vdirs]:      # basename only: a stored path can never escape these folders
            c = (d / name).resolve()
            if c.is_file() and d.resolve() in c.parents:
                return c
        return None

    @app.get("/api/health")
    def health():
        return {"status": "ok", "data_dir": str(data)}

    @app.get("/api/runs")
    def list_runs():
        runs = []
        # Only real run folders inside data_dir: unsafe names and links that escape it are skipped, not fatal.
        safe = lambda p: p.is_dir() and RUN_NAME.match(p.name) and data in p.resolve().parents
        for d in sorted(p for p in data.iterdir() if safe(p)) if data.is_dir() else []:
            runs.append({"name": d.name, **{k: (d / f).is_file() for k, f in FILES.items()},
                         "video": video_path(d.name) is not None})
        return {"runs": runs}

    # ------------------------------------------------------------------ videos + processing jobs
    runner = pipeline_runner or default_pipeline_runner
    jobs: Dict[str, Job] = {}
    jobs_lock = threading.Lock()

    def video_files() -> Dict[str, Path]:
        """Videos found directly inside the configured video folders, by file name (no subfolders, no traversal)."""
        found: Dict[str, Path] = {}
        for d in vdirs:
            if d.is_dir():
                for p in sorted(d.iterdir()):
                    if p.is_file() and p.suffix.lower() in VIDEO_EXT and RUN_NAME.match(p.stem) and p.name not in found:
                        found[p.name] = p
        return found

    def run_flags(run: str) -> dict:
        d = (data / run)
        return {k: (d / f).is_file() for k, f in FILES.items()}

    @app.get("/api/videos")
    def list_videos():
        """Every video in the project folder (processed or not), plus processed runs whose video file is gone."""
        items, stems = [], set()
        for name, p in video_files().items():
            stems.add(p.stem)
            items.append({"name": p.stem, "file": name, "size_mb": round(p.stat().st_size / 1e6, 2), "video": True, **run_flags(p.stem)})
        if data.is_dir():
            for d in sorted(x for x in data.iterdir() if x.is_dir() and RUN_NAME.match(x.name) and data in x.resolve().parents):
                if d.name not in stems:
                    items.append({"name": d.name, "file": None, "size_mb": None, "video": video_path(d.name) is not None, **run_flags(d.name)})
        return {"videos": sorted(items, key=lambda i: i["name"])}

    @app.get("/api/videos/{name}/file")
    def get_video_file(name: str):
        p = video_files().get(name)
        if p is None:
            raise HTTPException(404, f"Unknown video '{name}'")
        return FileResponse(p)

    @app.post("/api/videos/{name}/process", status_code=202)
    def process_video(name: str, req: ProcessRequest):
        """Run YOLO + reconstruction + simulation for a video in the background; poll /api/jobs/{id}."""
        p = video_files().get(name)
        if p is None:
            raise HTTPException(404, f"Unknown video '{name}'")
        run_dir = data / p.stem
        if (run_dir / FILES["recording"]).is_file() and not req.force:
            raise HTTPException(409, f"'{p.stem}' is already processed (send force=true to process it again).")
        with jobs_lock:
            if any(j.status == "running" for j in jobs.values()):
                raise HTTPException(409, "Another video is being processed; wait for it to finish.")
            job = Job(uuid.uuid4().hex[:12], name, p.stem)
            jobs[job.id] = job
        opts = req.model_dump(exclude={"force"})

        def report(step: str, idx: int, total: int, tail: str):
            job.step, job.step_index, job.n_steps = step, idx, total
            if tail:
                job.log.append(f"{step}: {tail.splitlines()[-1]}")

        def work():
            try:
                runner(p, run_dir, opts, report)
                if not (run_dir / FILES["recording"]).is_file():
                    raise RuntimeError("The pipeline finished but produced no recording (no approaching obstacle was measured).")
                job.status, job.step, job.step_index = "done", "Done", job.n_steps
            except Exception as e:  # shown to the user, not swallowed
                job.status, job.error = "error", str(e)[-900:]

        threading.Thread(target=work, daemon=True).start()
        return job.to_dict()

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str):
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, f"Unknown job '{job_id}'")
        return job.to_dict()

    @app.get("/api/runs/{run}/track")
    def get_track(run: str):
        return FileResponse(read_json(run, "track"), media_type="application/json")

    @app.get("/api/runs/{run}/metric")
    def get_metric(run: str):
        return FileResponse(read_json(run, "metric"), media_type="application/json")

    @app.get("/api/runs/{run}/recording")
    def get_recording(run: str):
        return FileResponse(read_json(run, "recording"), media_type="application/json")

    @app.get("/api/runs/{run}/video")
    def get_video(run: str):
        p = video_path(run)
        if p is None:
            raise HTTPException(404, f"Source video for run '{run}' not found")
        return FileResponse(p)     # Starlette FileResponse honours HTTP Range requests (seeking)

    @app.post("/api/runs/{run}/simulate")
    def simulate(run: str, req: SimulateRequest):
        """Re-run avoidance + physics from the stored metric scenario and return the new recording.
        Unavoidable scenarios return 200 with flight=null and the reason in planning.note."""
        metric = json.loads(read_json(run, "metric").read_text(encoding="utf-8"))
        if metric.get("scenario") is None:
            raise HTTPException(400, "This run has no measurable approach, so there is nothing to simulate.")
        cfg, params = RunConfig(), DroneParams()
        if req.speed_mps is not None: cfg.approach_speed_mps = req.speed_mps
        if req.altitude_m is not None: cfg.start_altitude_m = req.altitude_m
        if req.floor_m is not None: cfg.floor_z_m = req.floor_m
        if req.no_floor: cfg.floor_z_m = None
        if req.margin_m is not None: cfg.safety_margin_m = req.margin_m
        if req.drone_radius_m is not None: cfg.drone_radius_m = req.drone_radius_m
        if req.dodge != "auto": cfg.planner.direction_deg = DODGE_DEG[req.dodge]
        if req.mass_kg is not None: params.mass_kg = req.mass_kg
        if req.max_thrust_per_motor_n is not None: params.max_thrust_per_motor_n = req.max_thrust_per_motor_n
        cfg.wind = WindConfig(req.wind_speed_mps, req.wind_azimuth_deg, req.wind_elevation_deg,
                              req.turbulence_sigma_mps, req.turbulence_tau_s, req.wind_seed)
        run_result = run_avoidance(metric["scenario"], params, cfg)
        rec = build_recording(run_result, params, cfg, source={"metric_json": FILES["metric"]})
        if req.save:
            save_recording(rec, str(run_dir(run) / FILES["recording"]))
        return rec

    dist = Path(frontend_dist) if frontend_dist else REPO_ROOT / "frontend" / "dist"
    if dist.is_dir():     # serve the built website from the same origin (Part 8)
        app.mount("/", StaticFiles(directory=dist, html=True), name="frontend")
    return app


app = create_app()
