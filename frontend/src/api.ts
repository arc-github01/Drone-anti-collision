import type { Job, Metric, ProcessRequest, Recording, RunInfo, SimulateRequest, Track } from "./types";

async function json<T>(url: string, init?: RequestInit): Promise<T> {
  const r = await fetch(url, init);
  if (!r.ok) {
    let detail = `${r.status} ${r.statusText}`;
    try {
      const body = await r.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* keep the status text */
    }
    throw new Error(detail);
  }
  return r.json() as Promise<T>;
}

export const api = {
  runs: () => json<{ runs: RunInfo[] }>("/api/runs").then((r) => r.runs),
  /** Every video in the project folder (processed or not) plus processed runs without a video file. */
  videos: () => json<{ videos: RunInfo[] }>("/api/videos").then((r) => r.videos),
  videoFileUrl: (file: string) => `/api/videos/${encodeURIComponent(file)}/file`,
  processVideo: (file: string, req: ProcessRequest) =>
    json<Job>(`/api/videos/${encodeURIComponent(file)}/process`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req),
    }),
  job: (id: string) => json<Job>(`/api/jobs/${encodeURIComponent(id)}`),
  track: (run: string) => json<Track>(`/api/runs/${encodeURIComponent(run)}/track`),
  metric: (run: string) => json<Metric>(`/api/runs/${encodeURIComponent(run)}/metric`),
  recording: (run: string) => json<Recording>(`/api/runs/${encodeURIComponent(run)}/recording`),
  videoUrl: (run: string) => `/api/runs/${encodeURIComponent(run)}/video`,
  simulate: (run: string, req: SimulateRequest) =>
    json<Recording>(`/api/runs/${encodeURIComponent(run)}/simulate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req),
    }),
};
