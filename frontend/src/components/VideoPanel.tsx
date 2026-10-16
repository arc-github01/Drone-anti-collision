import { useEffect, useRef, useState } from "react";
import { api } from "../api";
import { Playback, usePlayback } from "../playback";
import { videoTime } from "../replay";
import type { Metric, Recording, Track, TrackSample } from "../types";

interface Props {
  run: string;
  track: Track | null;
  metric: Metric | null;
  rec: Recording;
  pb: Playback;
  hasVideo: boolean;
}

const nearest = <T extends { frame: number }>(rows: T[], frame: number): T | null =>
  rows.length ? rows[Math.min(rows.length - 1, Math.max(0, Math.round(frame)))] : null;

export function VideoPanel({ run, track, metric, rec, pb, hasVideo }: Props) {
  usePlayback(pb);
  const video = useRef<HTMLVideoElement>(null);
  const overlay = useRef<HTMLCanvasElement>(null);
  const [ended, setEnded] = useState(false);
  const sync = rec.video_sync;
  const vt = videoTime(sync, pb.time);
  const fps = track?.video.fps ?? 24;
  const sample: TrackSample | null = track && vt !== null ? nearest(track.samples, vt * fps) : null;
  const m = metric && vt !== null ? nearest(metric.samples, vt * fps) : null;

  // Keep the <video> locked to the playhead: seek when scrubbing/paused, play at a matching rate when playing.
  useEffect(() => {
    const v = video.current;
    if (!v || vt === null) return;
    const dur = v.duration || Infinity;
    const target = Math.min(vt, dur - 0.001);
    setEnded(vt > dur);
    if (pb.playing && vt <= dur) {
      const rate = Math.min(16, Math.max(0.0625, (sync.time_scale ?? 1) * pb.speed));
      if (Math.abs(v.playbackRate - rate) > 1e-3) v.playbackRate = rate;
      if (Math.abs(v.currentTime - target) > 0.3) v.currentTime = target;
      if (v.paused) void v.play().catch(() => undefined);
    } else {
      if (!v.paused) v.pause();
      if (Math.abs(v.currentTime - target) > 0.02) v.currentTime = target;
    }
  });

  // Detection overlay drawn in the video's own pixel space, scaled to the displayed size.
  useEffect(() => {
    const c = overlay.current;
    const v = video.current;
    if (!c || !v || !track) return;
    const W = v.clientWidth;
    const H = v.clientHeight;
    if (c.width !== W) c.width = W;
    if (c.height !== H) c.height = H;
    const g = c.getContext("2d")!;
    g.clearRect(0, 0, W, H);
    if (!sample?.bbox) return;
    const k = W / track.video.width;
    const [x1, y1, x2, y2] = sample.bbox.map((n) => n * k);
    g.lineWidth = 2;
    g.strokeStyle = sample.detected ? "#7bd88f" : "#ffb703";
    g.strokeRect(x1, y1, x2 - x1, y2 - y1);
    g.fillStyle = g.strokeStyle;
    g.font = "12px ui-monospace, monospace";
    g.fillText(`${track.target.yolo_class ?? "obstacle"} #${track.target.track_id}  ${(sample.conf * 100).toFixed(0)}%`, x1 + 4, Math.max(14, y1 - 6));
  });

  const f = (n: number | null | undefined, d = 2, u = "") => (n === null || n === undefined ? "—" : `${n.toFixed(d)}${u}`);
  return (
    <section className="card video" data-testid="video-panel">
      <h2>Original video + detection</h2>
      {!hasVideo || !track ? (
        <p className="muted">No source video or detection data for this run.</p>
      ) : (
        <>
          <div className="video-wrap">
            <video ref={video} src={api.videoUrl(run)} muted playsInline preload="auto" data-testid="video" />
            <canvas ref={overlay} className="overlay" />
            {ended && <div className="video-ended">Video ended — simulation continues</div>}
            {sync.time_scale === null && <div className="video-ended">Video not synced (speed overridden)</div>}
          </div>
          <table className="telemetry" data-testid="telemetry">
            <tbody>
              <tr>
                <th>Video time</th>
                <td>{f(vt, 2, " s")}</td>
                <th>Frame</th>
                <td>{sample?.frame ?? "—"}</td>
              </tr>
              <tr>
                <th>Detection</th>
                <td>{sample ? (sample.detected ? "tracked" : sample.ttc_source === "extrapolated" ? "lost (extrapolating)" : "none") : "—"}</td>
                <th>Confidence</th>
                <td>{sample?.detected ? f(sample.conf, 2) : "—"}</td>
              </tr>
              <tr>
                <th>Box (px)</th>
                <td>{sample?.w_px ? `${sample.w_px.toFixed(0)} × ${sample.h_px?.toFixed(0)}` : "—"}</td>
                <th>Growth</th>
                <td>{f(sample?.growth_px_s, 1, " px/s")}</td>
              </tr>
              <tr>
                <th>Time to contact</th>
                <td>{f(sample?.ttc_s, 1, " s")}</td>
                <th>Range (est.)</th>
                <td>{f(m?.range_m, 2, " m")}</td>
              </tr>
              <tr>
                <th>Closing speed</th>
                <td>{f(m?.closing_speed_mps, 2, " m/s")}</td>
                <th>Panel size</th>
                <td>{m?.width_m ? `${m.width_m.toFixed(2)} × ${m.height_m?.toFixed(2)} m` : "—"}</td>
              </tr>
            </tbody>
          </table>
          {sync.time_scale !== null && Math.abs(sync.time_scale - 1) > 1e-6 && (
            <p className="muted small" data-testid="sync-note">
              Approach speed was overridden, so video time runs at {sync.time_scale.toFixed(2)}× simulation time (approximate sync).
            </p>
          )}
          {track.other_tracks && track.other_tracks.length > 0 && (
            <p className="muted small" data-testid="other-tracks">
              Also detected (not used as the obstacle):{" "}
              {track.other_tracks.map((o) => `${o.yolo_class} #${o.track_id} (${o.frames} frames)`).join(", ")}.
            </p>
          )}
          <p className="muted small">
            Pixel sizes are measured; metres use a configured panel size and camera field of view (assumed values), so range and speed are estimates.
          </p>
        </>
      )}
    </section>
  );
}
