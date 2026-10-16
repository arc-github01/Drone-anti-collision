import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api } from "./api";
import { Hud } from "./components/Hud";
import { MotorGraphs } from "./components/MotorGraphs";
import { PlanPanel, SimPanel, Transport } from "./components/Panels";
import { ProcessPanel } from "./components/ProcessPanel";
import { Scene3D } from "./components/Scene3D";
import { VideoPanel } from "./components/VideoPanel";
import { Playback } from "./playback";
import { Replay } from "./replay";
import type { Metric, Recording, RunInfo, SimulateRequest, Track } from "./types";
import { NO_WIND, WindDraft, windVector } from "./windMath";

type View = "avoid" | "nominal";

export default function App() {
  const pb = useRef(new Playback()).current;
  const [runs, setRuns] = useState<RunInfo[]>([]);
  const [run, setRun] = useState<string>("");
  const [saved, setSaved] = useState<Recording | null>(null);
  const [override, setOverride] = useState<Recording | null>(null);
  const [track, setTrack] = useState<Track | null>(null);
  const [metric, setMetric] = useState<Metric | null>(null);
  const [view, setView] = useState<View>("avoid");
  const [loadErr, setLoadErr] = useState<string | null>(null);
  const [simErr, setSimErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [modelScale, setModelScale] = useState(3);
  const [showThrust, setShowThrust] = useState(true);
  const [showClearance, setShowClearance] = useState(true);
  const [showNoAvoid, setShowNoAvoid] = useState(true);
  const [windDraft, setWindDraft] = useState<WindDraft>(NO_WIND);
  const [showProcess, setShowProcess] = useState(false);

  const rec = override ?? saved;

  const refreshVideos = useCallback(
    (select?: string) =>
      api
        .videos()
        .then((r) => {
          setRuns(r);
          const pick = select ?? r.find((x) => x.recording)?.name ?? r[0]?.name;
          if (pick) setRun((cur) => (select || !cur ? pick : cur));
          else setLoadErr("No videos found. Put a .mp4 in the project folder and reload.");
        })
        .catch((e) => setLoadErr(`Cannot reach the API (${e.message}). Start it with: python -m dronesim.api`)),
    [],
  );
  useEffect(() => {
    void refreshVideos();
  }, [refreshVideos]);

  useEffect(() => {
    if (!run) return;
    setSaved(null);
    setOverride(null);
    setTrack(null);
    setMetric(null);
    setLoadErr(null);
    setSimErr(null);
    setShowProcess(false);
    const info = runs.find((r) => r.name === run);
    if (info && !info.recording) return; // not processed yet: the Process panel is shown instead
    api
      .recording(run)
      .then((r) => {
        setSaved(r);
        const c = r.wind?.config;
        setWindDraft(c ? { speed: c.speed_mps, azimuth: c.azimuth_deg, elevation: c.elevation_deg, sigma: c.turbulence_sigma_mps, tau: c.turbulence_tau_s, seed: c.seed } : NO_WIND);
      })
      .catch((e) => setLoadErr(e.message));
    if (info?.track) api.track(run).then(setTrack).catch(() => undefined);
    if (info?.metric) api.metric(run).then(setMetric).catch(() => undefined);
  }, [run, runs]);

  // Which flight drives the scene/graphs: the avoiding one, or the straight (no-avoidance) one.
  const flightKey: View = view === "avoid" && rec?.flight ? "avoid" : rec?.nominal_flight ? "nominal" : "avoid";
  const replay = useMemo(() => {
    const f = rec && (flightKey === "avoid" ? rec.flight : rec.nominal_flight);
    return f ? new Replay(f) : null;
  }, [rec, flightKey]);
  const nominalReplay = useMemo(() => (rec?.nominal_flight ? new Replay(rec.nominal_flight) : null), [rec]);

  useEffect(() => {
    pb.pause();
    pb.setDuration(replay?.duration ?? 0);
    pb.seek(0);
  }, [replay, pb]);

  const simulate = useCallback(
    async (req: SimulateRequest) => {
      setBusy(true);
      setSimErr(null);
      try {
        const r = await api.simulate(run, req);
        if (req.save) setSaved(r);
        else setOverride(r);
        if (req.save) setOverride(null);
        setView("avoid");
      } catch (e) {
        setSimErr((e as Error).message);
      } finally {
        setBusy(false);
      }
    },
    [run],
  );

  const info = runs.find((r) => r.name === run);
  const rc = rec?.wind?.config;
  const windStale =
    !!rec &&
    (windDraft.speed !== (rc?.speed_mps ?? 0) ||
      (windDraft.speed > 0 && (windDraft.azimuth !== (rc?.azimuth_deg ?? 0) || windDraft.elevation !== (rc?.elevation_deg ?? 0))) ||
      windDraft.sigma !== (rc?.turbulence_sigma_mps ?? 0));
  const windVec = windVector(Number.isFinite(windDraft.speed) ? windDraft.speed : 0, Number.isFinite(windDraft.azimuth) ? windDraft.azimuth : 0, Number.isFinite(windDraft.elevation) ? windDraft.elevation : 0);
  const closest = rec ? (flightKey === "avoid" ? rec.summary : rec.nominal_summary)?.t_min_clearance_s ?? null : null;

  return (
    <div className="app">
      <header>
        <h1>Drone Navigator — avoidance simulation</h1>
        <label>
          Run
          <select value={run} onChange={(e) => setRun(e.target.value)} data-testid="run-select">
            {runs.map((r) => (
              <option key={r.name} value={r.name}>
                {r.file ?? r.name}
                {r.recording ? " ✓ processed" : " — not processed"}
              </option>
            ))}
          </select>
        </label>
        {rec && info?.file && (
          <button onClick={() => setShowProcess((v) => !v)} data-testid="reprocess-toggle" title="Run detection and simulation again for this video">
            Re-process…
          </button>
        )}
        {rec && (
          <div className="seg" role="group" aria-label="Flight to show">
            <button className={flightKey === "avoid" ? "on" : ""} disabled={!rec.flight} onClick={() => setView("avoid")}>
              With avoidance
            </button>
            <button className={flightKey === "nominal" ? "on" : ""} disabled={!rec.nominal_flight} onClick={() => setView("nominal")}>
              Without avoidance
            </button>
          </div>
        )}
        {override && <span className="badge">unsaved re-simulation</span>}
      </header>

      {info && !info.recording && info.file && (
        <ProcessPanel
          item={info}
          reprocess={false}
          onDone={() => {
            setSaved(null);
            void refreshVideos(info.name);
          }}
        />
      )}
      {showProcess && info && rec && (
        <ProcessPanel
          item={info}
          reprocess
          onClose={() => setShowProcess(false)}
          onDone={() => {
            setShowProcess(false);
            setSaved(null);
            void refreshVideos(info.name);
          }}
        />
      )}

      {loadErr && (
        <div className="banner bad" role="alert" data-testid="load-error">
          {loadErr}
        </div>
      )}
      {rec && !rec.flight && (
        <div className="banner bad" data-testid="unavoidable">
          {rec.planning.note}. Showing the straight flight (no avoidance) instead.
        </div>
      )}
      {rec && rec.flight && flightKey === "nominal" && <div className="banner warn">Showing the straight flight WITHOUT avoidance.</div>}

      {rec && replay ? (
        <main>
          <div className="left">
            <div className="scene" data-testid="scene">
              <Scene3D
                rec={rec}
                replay={replay}
                pb={pb}
                nominalReplay={nominalReplay}
                modelScale={modelScale}
                showThrust={showThrust}
                showClearance={showClearance}
                showNoAvoid={showNoAvoid && flightKey === "avoid"}
                wind={windVec}
              />
              <Hud rec={rec} replay={replay} pb={pb} label={flightKey === "avoid" ? "with avoidance" : "NO avoidance"} />
              <div className="legend">
                <span><i style={{ background: "#ff6b6b" }} />intended (straight)</span>
                <span><i style={{ background: "#5aa9ff" }} />planned detour</span>
                <span><i style={{ background: "#fff" }} />actual (simulated)</span>
                {showNoAvoid && flightKey === "avoid" && nominalReplay && <span><i style={{ background: "#ff4d4d" }} />flown without avoidance</span>}
                <span><i style={{ background: "#ffb703" }} />keep-out zone</span>
              </div>
              <div className="scene-opts">
                <label>
                  Drone model ×{modelScale}
                  <input type="range" min={1} max={8} step={1} value={modelScale} onChange={(e) => setModelScale(+e.target.value)} />
                </label>
                <label className="check"><input type="checkbox" checked={showThrust} onChange={(e) => setShowThrust(e.target.checked)} />thrust arrows</label>
                <label className="check"><input type="checkbox" checked={showClearance} onChange={(e) => setShowClearance(e.target.checked)} />clearance</label>
                <label className="check"><input type="checkbox" checked={showNoAvoid} onChange={(e) => setShowNoAvoid(e.target.checked)} />no-avoid path</label>
              </div>
              <Transport pb={pb} />
            </div>
            {/* directly below the 3D scene */}
            <MotorGraphs replay={replay} pb={pb} motorNames={rec.drone.motor_layout.names} hoverCommand={rec.drone.hover_command} closestApproachT={closest} />
          </div>
          <aside>
            <VideoPanel run={run} track={track} metric={metric} rec={rec} pb={pb} hasVideo={!!info?.video} />
            <PlanPanel rec={rec} />
            <SimPanel busy={busy} onSimulate={simulate} onReset={() => setOverride(null)} modified={!!override} error={simErr} wind={windDraft} onWind={setWindDraft} windStale={windStale} />
          </aside>
        </main>
      ) : (
        !loadErr && !(info && !info.recording) && <div className="banner">Loading…</div>
      )}
    </div>
  );
}
