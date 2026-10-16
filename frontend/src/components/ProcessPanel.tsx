import { useEffect, useRef, useState } from "react";
import { api } from "../api";
import type { Job, RunInfo } from "../types";

const num = (s: string) => (s.trim() === "" ? undefined : Number(s));

/** Process (or re-process) a video: YOLO tracking -> metric reconstruction -> avoidance simulation, run on the server. */
export function ProcessPanel({ item, reprocess, onDone, onClose }: { item: RunInfo; reprocess: boolean; onDone: () => void; onClose?: () => void }) {
  const [width, setWidth] = useState("");
  const [height, setHeight] = useState("");
  const [fov, setFov] = useState("");
  const [conf, setConf] = useState("");
  const [job, setJob] = useState<Job | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const timer = useRef<number | undefined>(undefined);

  useEffect(() => () => window.clearTimeout(timer.current), []);
  useEffect(() => {
    setJob(null);
    setErr(null);
  }, [item.name]);

  const poll = (id: string) => {
    api
      .job(id)
      .then((j) => {
        setJob(j);
        if (j.status === "running") timer.current = window.setTimeout(() => poll(id), 800);
        else if (j.status === "done") onDone();
      })
      .catch((e) => setErr((e as Error).message));
  };

  const start = async () => {
    setErr(null);
    try {
      const j = await api.processVideo(item.file!, {
        width_m: num(width),
        height_m: num(height),
        hfov_deg: num(fov),
        conf: num(conf),
        force: reprocess,
      });
      setJob(j);
      poll(j.id);
    } catch (e) {
      setErr((e as Error).message);
    }
  };

  const running = job?.status === "running";
  const pct = job ? Math.round((100 * (job.status === "done" ? job.n_steps : job.step_index)) / job.n_steps) : 0;
  return (
    <section className="card process" data-testid="process-panel">
      <div className="process-head">
        <h2>{reprocess ? `Re-process ${item.name}` : `Process ${item.name}`}</h2>
        {onClose && (
          <button onClick={onClose} disabled={running} aria-label="Close">
            ✕
          </button>
        )}
      </div>
      {!reprocess && (
        <p className="muted">
          This video has not been processed yet. Processing runs YOLO on every frame, measures the obstacle, then simulates the avoidance. It can take a minute
          or two on a CPU.
        </p>
      )}
      {item.file && <video className="preview" src={api.videoFileUrl(item.file)} controls muted playsInline preload="metadata" data-testid="preview" />}
      <div className="form">
        <label>
          Real obstacle width (m)
          <input value={width} onChange={(e) => setWidth(e.target.value)} placeholder="0.5 (assumed)" inputMode="decimal" data-testid="proc-width" />
        </label>
        <label>
          Real obstacle height (m)
          <input value={height} onChange={(e) => setHeight(e.target.value)} placeholder="0.5 (assumed)" inputMode="decimal" data-testid="proc-height" />
        </label>
        <label>
          Camera field of view (°)
          <input value={fov} onChange={(e) => setFov(e.target.value)} placeholder="82.6 (assumed)" inputMode="decimal" />
        </label>
        <label>
          Detection confidence
          <input value={conf} onChange={(e) => setConf(e.target.value)} placeholder="0.30" inputMode="decimal" />
        </label>
      </div>
      <p className="muted small">
        Pixels are not metres: distances and speeds depend on the real obstacle size and the camera field of view. Leave them empty to use the assumed
        values, or enter what you know.
      </p>
      <div className="btns">
        <button className="primary" onClick={start} disabled={running || !item.file} data-testid="process-btn">
          {running ? "Processing…" : reprocess ? "Re-process video" : "Process video"}
        </button>
      </div>
      {job && (
        <div className="progress" data-testid="job-status">
          <div className="bar">
            <div style={{ width: `${pct}%` }} className={job.status === "error" ? "err" : ""} />
          </div>
          <p className={job.status === "error" ? "error" : "muted small"} role={job.status === "error" ? "alert" : undefined}>
            {job.status === "running" && `Step ${job.step_index + 1} of ${job.n_steps}: ${job.step}…`}
            {job.status === "done" && "Done. Loading the result…"}
            {job.status === "error" && `Failed: ${job.error}`}
          </p>
        </div>
      )}
      {err && (
        <p className="error" role="alert" data-testid="process-error">
          {err}
        </p>
      )}
    </section>
  );
}
