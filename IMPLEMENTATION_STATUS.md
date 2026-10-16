# Implementation Status

Goal: video -> YOLO obstacle data (length, breadth, approach speed) -> physics simulation of a
quadcopter avoiding the obstacle via 4 motor commands -> 3D website + 4 motor-vs-time graphs.
Wind comes AFTER the baseline works.

## Defaults (assumed until changed)
- Obstacle: highest-confidence tracked detection; real size is a configured value (default 0.5 m x 0.5 m), not measured.
- Stack: Python (NumPy) sim + FastAPI backend; Vite + React + TypeScript + React Three Fiber frontend.
- Frames: world ENU (z up), body FLU, SI units. Fixed physics dt = 1 ms.
- Tello parameters are configurable assumed values, not measured.

## Parts
| # | Part | Output | Tests |
|---|------|--------|-------|
| 0 | Scaffold: `dronesim/` package, config dataclasses, install pytest/fastapi/uvicorn | package skeleton | import test |
| 1 | Vision -> data: track objects, record per-frame timestamp, bbox width/height (px), growth rate, time-to-contact | `obstacle_track.json` | synthetic tracks + real run on `testv2.mp4` |
| 2 | Obstacle reconstruction: px -> metric length/breadth/range/closing speed using configured size + camera FOV | obstacle in world frame | pinhole math tests |
| 3 | Drone physics: rigid body, gravity, drag, thrust `T=kT*w^2`, torques, motor lag/limits | physics engine | hover, free fall, torque signs, saturation |
| 4 | Flight controller: trajectory -> 4 motor commands (cascaded PID + mixer) | controller | hover, step, tracking |
| 5 | Collision prediction + avoidance planner (reject unsafe/infeasible paths) + sim runner | `sim_recording.json` | collision, infeasible rejection, end-to-end avoid |
| 6 | Recording/replay (no YOLO rerun) | replayable history | round trip, determinism |
| 7 | Backend API (runs, recording, video) | FastAPI | API tests |
| 8 | 3D website: scene, trajectories, clearance, attitude + 4 motor graphs below, scrub/replay, video | frontend | build + replay check |
| 9 | Docs: assumptions + uncertainty | README section | - |
| 10 | Wind (later) | wind model into drag term | wind tests |

## Completed
- Repo inspection and plan.
- **Part 0 (scaffold):** `dronesim/` package, `dronesim/config.py` (vision/obstacle/camera configs; size and FOV flagged as assumed). pytest, fastapi, uvicorn installed in `venv`.
- **Part 1 (vision -> data):** 
  - `dronesim/vision/detector.py` YOLO + ByteTrack wrapper (timestamps = frame/fps).
  - `dronesim/vision/obstacle_estimator.py` pure-NumPy w/h (px), growth rate and time-to-contact via a linear fit of 1/size (exact for constant closing speed); extrapolates TTC across detector dropouts (<= 2 s).
  - `dronesim/vision/export.py` picks the obstacle track (largest accumulated box area) and writes `data/runs/<video>/obstacle_track.json`.
  - Run: `venv\Scripts\python.exe -m dronesim.vision.export testv2.mp4`
  - Tests: `venv\Scripts\python.exe -m pytest tests -q` -> 7 passed.
  - Real run on `testv2.mp4`: a cardboard square panel tracked (YOLO calls it "refrigerator") in 204/240 frames; TTC ~7 s early, ~4 s at frame 204.
- Existing `object_detection_for_rescue_operations.py` is untouched and still works.
- **Part 2 (metric reconstruction):** `dronesim/vision/reconstruction.py`.
  - Pinhole model: `f = (W/2)/tan(hfov/2)`, range `Z = f*S/size_px` (geometric mean of width and height estimates), closing speed `v = Z*growth/size`, lateral/vertical offsets `Z*dpx/f`.
  - Dropout frames carry range forward at the last closing speed (flagged `extrapolated`).
  - `build_scenario()` outputs the static-world scenario for the simulator: drone at origin facing +x with the measured approach speed; square-panel obstacle centre, size, thickness, normal. Frames: camera +Z fwd/+X right/+Y up -> sim +x fwd/+y left/+z up.
  - Run: `venv\Scripts\python.exe -m dronesim.vision.reconstruction data/runs/testv2/obstacle_track.json` -> `obstacle_metric.json`.
  - Tests: 13 passed total (6 new: focal length, range, speed/offsets, dropout, scenario frame convention, no-detection).
  - `testv2.mp4` result: range0 = 2.72 m, approach = 0.33 m/s, TTC0 = 8.3 s, obstacle 0.5x0.5 m (ASSUMED size and FOV).

- **Part 3 (drone physics):** `dronesim/sim/`
  - `params.py` ASSUMED Tello-class parameters (80 g, 4.5 cm arms, hover command 0.70, T_max 0.40 N/motor); everything configurable.
  - `quaternion.py` body->world quaternion helpers. `motors.py` command -> omega (lag) -> thrust `kT*w^2` -> electrical power as four distinct quantities, plus the thrust<->wrench mixer. `quadrotor.py` 17-state rigid body (p, v, q, w, rotor speeds), fixed-step RK4, zero-order-hold commands; `step(u)` is the only way to move the drone.
  - Models gravity, quadratic drag, rotor thrust and reaction torque (+ rotor spin-up reaction), gyroscopic coupling `w x Jw`, angular damping, first-order motor lag, command saturation [0,1]. Wind is a zero-default hook (`wind_fn`), unused for now.
  - Frames: world ENU, body FLU. Motors (X layout): M1 front-right CCW, M2 rear-right CW, M3 rear-left CCW, M4 front-left CW. Front motors up -> nose up (-w_y); right motors up -> -w_x; CCW motors faster -> body yaw -w_z.
  - Tests: 28 passed total (15 new): hover hold, analytic free fall, terminal velocity, tilted-thrust a_x = g tan(theta), torque signs, tau = J*alpha, torque-free L and energy conservation, motor lag 63.2 %, saturation, mixer round-trip, step-size convergence, exact fixed dt, determinism. Mutation-checked: flipping pitch/yaw torque sign, the gyroscopic term or quaternion order makes tests fail.

- **Part 4 (flight controller):** `dronesim/sim/controller.py`, `dronesim/sim/loop.py`
  - Cascade: position/velocity PD + I + acceleration feed-forward -> desired force vector (acceleration and tilt limited) -> geometric attitude control (rotation matrices, no Euler singularity) -> body-rate P loop with `w x Jw` cancellation -> torques -> mixer. Output is only 4 motor commands in [0,1].
  - `desaturate()` fits [thrust, torque] into 0 <= T_i <= Tmax. Priority: roll/pitch torque > collective thrust > yaw (yaw cut first by bisection, then roll/pitch scaled, then common-mode thrust clamped).
  - Integral term has anti-windup (only near the target, frozen while saturated); found by tuning: without it a 2 m step overshot 10 % and settled slowly.
  - `loop.simulate()` runs controller at 250 Hz (zero-order hold) over physics at 500 Hz and returns a plain-array history (t, position, velocity, quaternion, body rates, command, omega, thrust, power, saturated) - the basis for Part 6 recording.
  - Tests: 39 passed total (11 new): mixer/desaturation (feasible unchanged, yaw sacrificed first, attitude beats thrust), hover hold, 2 m step (settles within 2 cm, peak 2.016 m = 0.8 % overshoot, tilt <= limit), altitude step, recovery from 20/-15 deg tilt + rates, constant-velocity line (< 3 cm), smooth curve with accel feed-forward (< 5 cm), extreme 50 m request stays finite/bounded/airborne, tilt limit binding. Mutation-checked: attitude sign, error-matrix sign, accel feed-forward and tilt limit all make tests fail (the tilt-limit gap was found by this check and a dedicated test added).

- **Part 5 (collision prediction, avoidance planner, runner):** `dronesim/sim/collision.py`, `planner.py`, `runner.py`
  - `collision.py`: axis-aligned box SDF, spherical drone (radius 0.10 m assumed), optional floor, safety margin (0.25 m assumed), `predict_collision()` (first contact / first margin violation / min clearance).
  - `planner.py`: predicts collision on the nominal straight path; if needed sweeps 912 candidate minimum-jerk sidesteps (8 directions x 19 offsets x 3 blends x 2 timings) that hold while passing the panel then return to the line. Each candidate is REJECTED if it exceeds 80 % of the acceleration / thrust / tilt limits or lacks obstacle/floor clearance (margin + 0.05 m reserved for tracking error); rejection counts by reason are reported. Cheapest feasible candidate wins. Fallback: braking stop. If neither fits: plan = None, "UNAVOIDABLE", nothing is faked.
  - `runner.py`: `run_avoidance(scenario, params, cfg)` flies the plan through controller + physics and measures clearance on the ACTUAL flown path; also flies the straight path (no avoidance) for comparison. Configurable start altitude / floor / margin / approach-speed override.
  - Real `testv2.mp4` scenario (0.33 m/s, 2.72 m): straight flight hits the panel (clearance -0.124 m); planned detour (0.6 m down, 3 s blend) flies with actual min clearance 0.347 m, tracking error 2.5 mm, no saturation. At 2 m/s it still avoids (0.38 m); at 8 m/s it is correctly reported unavoidable.
  - Tests: 52 passed total (13 new): box SDF (face/edge/corner/inside), clearance with radius/floor, contact prediction, clear path, detour constraints/smoothness/returns to line, rejection with reasons, too-fast = unavoidable, floor, brake fallback, flown avoidance vs flown collision, video-scale speed, unavoidable flies nothing, determinism. Mutation-checked (wrong derivative scaling, skipped clearance check, skipped accel limit all fail tests).

- **Part 6 (recording and replay):** `dronesim/sim/recording.py`, `dronesim/run_scenario.py`
  - `build_recording()` -> `sim_recording.json` (schema `sim_recording/v1`, ~2 MB for a 15 s flight at 250 Hz): column-oriented flown history (t, position, velocity, quaternion, Euler, body rates, per-motor **command / omega / thrust / electrical power**, setpoint, clearance, saturation flag), the no-avoidance flight, intended (nominal) and planned trajectories, obstacle box + floor + margin, drone parameters (flagged ASSUMED), planner stats/rejections, summaries, video-time sync. No wall-clock fields: identical runs give byte-identical files.
  - `Replay` (NumPy + JSON only; no YOLO/OpenCV/torch/physics/planner imports, tested in a clean interpreter): `at(t)` is a pure function of `t` (linear interpolation, normalised-lerp attitude, zero-order-held motor commands, clamped ends), `motor_series(i)` = exactly the controller output for graph i, `path(t)` for the trail, `video_time(t)` for sync, `recompute_clearance()` self-check.
  - CLI: `venv\Scripts\python.exe -m dronesim.run_scenario data/runs/testv2/obstacle_metric.json [--speed S] [--dodge left|right|up|down|auto] [--altitude M] [--floor M|--no-floor] [--margin M]`.
  - Tests: 68 passed total (16 new): JSON round trip, equal column lengths + fixed dt, schema check, determinism, recorded thrust = kT w^2 / power formula / velocity = d(position)/dt (so history is integrated physics, not scripted), clearance recomputed from geometry (incl. penetrating flight), sample-exact and interpolated replay, order-independent scrubbing, motor graph data = controller output, video sync (and scaled sync when speed overridden), unavoidable run records without flight, replay imports. Mutation-checked (ZOH, time lookup, wrong graph source, missing inside-box geometry all fail tests; the last was a gap found and closed).
  - Planner default now mildly prefers sideways dodges (`vertical_penalty_m` 0.05 -> 0.15) and `--dodge` forces a direction: a purely vertical dodge commands all four motors identically (collective thrust only), which makes four overlapping graph lines.
  - Current `data/runs/testv2/sim_recording.json`: video speed 0.33 m/s, 0.7 m left detour, flown min clearance 0.351 m (straight flight: -0.124 m, collision).

- **Part 7 (backend API):** `dronesim/api/app.py`, `dronesim/api/__main__.py`
  - Start: `venv\Scripts\python.exe -m dronesim.api` -> http://127.0.0.1:8000 (interactive docs at `/docs`).
  - `GET /api/health`; `GET /api/runs` (runs + which files exist); `GET /api/runs/{run}/track | metric | recording` (JSON files; 404 with the exact CLI command that creates a missing one); `GET /api/runs/{run}/video` (source video, HTTP Range/206 so the browser can seek); `POST /api/runs/{run}/simulate` (body: `speed_mps, dodge auto|left|right|up|down, altitude_m, floor_m, no_floor, margin_m, drone_radius_m, save`) re-runs planning + controller + physics from the stored metric scenario and returns a new recording - no YOLO. Not saved unless `save: true`. Unavoidable scenarios are a normal 200 with `flight: null` and the reason in `planning.note`; out-of-range input -> 422.
  - Safety: run names must match `[A-Za-z0-9][A-Za-z0-9_.-]*`; resolved paths must stay inside the data dir (junctions/symlinks that escape are refused and skipped in the listing); the video is located by the BASENAME stored in the track file, only inside the run folder or configured video dirs; CORS allows only the Vite dev origins. The module imports no YOLO/OpenCV/torch (tested).
  - Serves `frontend/dist` from the same origin if it exists (Part 8).
  - Tests: 83 passed total (15 new): listing, verbatim JSON, 404 hints, malicious names, real private directories (hidden dir, escaping junction), video 200 + Range 206 with exact bytes, stored-path traversal, simulate == direct library run, options change flights (left vs right, wider margin -> bigger offset), save then fetch, unavoidable is not an error, input validation, CORS, no heavy imports, real `testv2` run. Mutation-checked: unsanitised video path, open CORS, swapped dodge mapping and missing containment each fail tests.
  - Live check with the real server on `testv2`: runs list, 206 video range, 2.1 MB recording, `simulate {speed 2.0, dodge right}` -> 0.9 m right detour, flown clearance 0.379 m; bad input -> 422; unknown run -> 404.
  - Bug found by the new tests and fixed: one run folder that is a junction pointing outside the data dir made `GET /api/runs` fail for ALL runs; it is now skipped.

- **Part 8 (3D website):** `frontend/` (Vite + React 19 + TypeScript + React Three Fiber/three.js; graphs are hand-written SVG)
  - Run it: `cd frontend && npm install && npm run build`, then `venv\Scripts\python.exe -m dronesim.api` and open http://127.0.0.1:8000 (the API serves `frontend/dist`). Dev mode: `npm run dev` (http://localhost:5173, proxies `/api` to :8000).
  - Layout: 3D scene (drone with attitude, rotors coloured per motor, thrust arrows, square panel, keep-out zone, intended/planned/actual/no-avoidance trajectories, clearance line + value) with HUD (status, attitude indicator, velocity, altitude, clearance) and a transport bar (play/pause, step, scrub, 0.25-4x, loop). **Directly below the scene: four separate motor graphs** stacked on a shared time axis, each with a playhead, live readout, "closest approach" marker, click/drag scrubbing, quantity selector (command / angular speed / thrust / electrical power), y-scale modes (zoom shared by all four, zoom each, full range) and a delta-from-hover view. Right column: original video with detection box overlay + telemetry (box px, growth, TTC, estimated range/closing speed), avoidance-plan summary (with rejection counts, flown vs no-avoidance clearance, assumptions), and a Re-simulate form (speed, dodge, margin, altitude, floor, save).
  - Everything shown comes from the recording: `frontend/src/replay.ts` is a TypeScript port of the Python `Replay` (pure function of time, zero-order-held commands, nlerp attitude); the browser never runs physics, so the timestep is the recording's fixed 4 ms and independent of render FPS. Rotor spin is decorative (integral of recorded rotor speed, slowed 67x); the airframe is drawn x3 life size (slider; collision sphere stays true size).
  - Unavoidable runs show a red banner with the reason and switch to the straight flight; API/validation errors are shown in the form.
  - Tests: vitest 16 passed (`cd frontend && npm test`): TS Replay vs 47 reference answers generated by Python (`python tools/make_replay_golden.py`) to 1e-9, interpolation/clamping/order-independent scrubbing, motor series = controller output, video-time mapping, quaternion double-cover, graph math (decimation never drops a spike, axis mapping, ranges). `npm run build` type-checks and builds. Headless-Chrome smoke test (`node e2e/smoke.mjs`, needs the server running): 17 checks passed on the real `testv2` run - four graphs rendered directly under the scene and equally wide, WebGL canvas non-blank, scrubber and graph clicks move all four playheads together, HUD clearance 0.351 m at closest approach, playback advances, video seeks to 3.17 s at sim 3.0 s and shows detection telemetry, re-simulate (2 m/s, right) and unavoidable (8 m/s) flows, input error shown, no console errors. Mutation-checked: wrong command hold, wrong index lookup, wrong motor series and removed quaternion sign fix each fail tests (the last was a gap found and closed).
  - Python suite unchanged and green: 83 passed.

## Known limitations (Part 8)
- At the video's real 0.33 m/s the four motors differ by only ~0.2 % of full command; at the default y-zoom the traces are clearly separate but their vertical differences are tiny - read the axis numbers (a note says so). At 2 m/s the differences are obvious.
- Video sync is exact only without a speed override; with an override it is scaled/approximate (the panel says so). The video ends at ~10 s while the simulation runs ~14.7 s; the panel says "Video ended".
- The airframe is drawn x3 larger than life by default (visual only); rotor spin speed is decorative.
- The JS bundle is ~1.2 MB (three.js), un-split. No mobile-specific layout testing beyond a single-column fallback; no accessibility audit beyond labels on controls and graphs.
- The e2e smoke test needs Chrome/Edge installed and a running server; it is not part of `npm test`.
- Frontend renders recorded data only: changing drone parameters needs an API/CLI change (not exposed in the form).

- **Part 10 (wind, done before docs as requested):**
  - **Model** (`dronesim/sim/wind.py`): steady 3D wind `w = V (cos el cos az, cos el sin az, sin el)` in the world frame (x forward/approach, y left, z up); direction = where the wind blows TOWARD (az 0 = tailwind, 180 = headwind, +90 = toward the left; el +90 = updraft). Optional per-axis Ornstein-Uhlenbeck gusts (exact discretisation, mean 0, variance sigma^2, autocorrelation exp(-lag/tau)), seeded; same seed gives the same gusts whatever the flight length. Drag uses air-relative velocity `F = -c_d |v - w| (v - w)` (already in the physics; now verified analytically in 3D).
  - **Controller:** disturbance observer (critically damped Luenberger on velocity, bandwidth 5 rad/s) estimates the unmodelled acceleration from the rotors' measured thrust + attitude and cancels it; it never reads the wind. Verified: estimate matches true drag to ~0.1 %, steady hover tilt matches `atan(c_d w^2 / m g)`-type force balance to 0.01 deg in any 3D direction, steady error ~1 cm; without it a 5 m/s headwind leaves 0.45 m. Integral term scaled to 5 % when the observer is on (it caused a slow 8 s tail). New `limited` flag = tilt/acceleration limit active.
  - **Planner:** limits are checked on the force the ROTORS must supply, `a_req = a_plan + (c_d/m)|v - w|(v - w)`; reports when the wind alone exceeds the drone's authority (e.g. 12 m/s: needs 54 deg tilt, "UNAVOIDABLE ... Wind 12.0 m/s ..."). Runner: 6 s unrecorded pre-roll so the drone starts in equilibrium with the wind; with gusts it verifies the flown clearance and re-plans with extra buffer (up to 2x).
  - **Recording/API/CLI:** `wind` block (config, vector, headwind/crosswind/vertical components, hover requirement), per-sample `wind_mps`, `limited`, `disturbance_est_mps2`, `limited_fraction`, planning attempts. API `POST .../simulate` takes `wind_speed_mps, wind_azimuth_deg, wind_elevation_deg, turbulence_sigma_mps, turbulence_tau_s, wind_seed` (validated). CLI: `--wind-speed --wind-azimuth --wind-elevation --turbulence --turbulence-tau --wind-seed`.
  - **UI:** wind section in Re-simulate: speed, SEPARATE azimuth and elevation (slider + number), presets (headwind/tailwind/left/right/updraft/downdraft/none), turbulence sigma (+ gust time/seed), live vector + headwind/crosswind/vertical readout, validation, "settings changed - press Re-simulate" notice; wind arrow in the 3D scene; HUD wind vector, air-relative speed and limit-active flag; plan panel wind description, hover requirement, at-limit share.
  - **Tests:** Python 142 passed (60 new: direction maths, OU statistics with derived tolerances, step-size independence, drag law in 3D, controller rejection in 9 wind directions, force-balance tilt, beyond-authority case, planner formulas, mirror symmetry, pre-roll equilibrium, gust re-plan, determinism, recording/API). Frontend 25 vitest passed (TS replay matches Python on a wind+gust recording, wind-direction maths matches Python values). Headless-Chrome smoke test: all checks passed incl. entering a 3D wind, stale notice, HUD/plan wind info, validation. Bugs found by the new tests and fixed: station-keeping drag sign (planner), gust realisation depending on flight duration, 3 s pre-roll too short, integral wind-up tail. Mutation spot-checks caught 7 of 7 wind-maths mutations tried (direction formula, drag sign, observer signs, planner sign, hover sign, gust time constant); 2 more (pre-roll, re-plan buffer) were not re-run after an interrupted batch.
  - **Real result (`testv2`, 5 m/s wind toward az 135 / el 30):** avoids with 0.37 m flown clearance (straight flight: -0.12 m, collision), leaning ~13 deg; every 3D direction tried (head/tail/cross/up/down/diagonals) avoids with 0.35-0.43 m.

- **Video picker + in-UI processing (added after Part 10):**
  - `GET /api/videos` lists every video file in the project folder (mp4/mov/avi/mkv/webm, no subfolders) with processed status, plus processed runs whose file is gone; `GET /api/videos/{name}/file` previews a video (Range); `POST /api/videos/{name}/process` (optional `width_m, height_m, hfov_deg, conf, force`) runs the three CLI steps (YOLO export -> reconstruction -> run_scenario) as subprocesses in a background job (one at a time, 409 if busy or already processed without `force`); `GET /api/jobs/{id}` gives step/progress/error. The API server still imports no YOLO.
  - UI: the header dropdown lists all videos as "name.mp4 ✓ processed / — not processed"; an unprocessed video shows a Process panel (preview, optional real obstacle size / FOV / confidence, progress bar, errors); processed ones have "Re-process...". The track JSON now records `other_tracks` (other objects YOLO followed), shown under the video as "Also detected (not used as the obstacle)".
  - Findings on the new video: `testv3.mp4` = the cardboard panel (tracked in all 240 frames) + a grey crate (196 frames) + duplicate small tracks; only the panel is used. Processed in ~34 s. `test2.mp4` (street) processes without error but its "obstacle" is a bus assumed to be 0.5 x 0.5 m, so its metric numbers are meaningless unless the real size is entered.
  - Bugs found/fixed on the way: drei `Html` labels threw `removeChild` when the scene unmounted (replaced by a plain DOM overlay); HUD showed "-0.0"; a stale server held port 8000 (killed).
  - Tests: Python 150 passed (8 new: listing, preview+range, job lifecycle, failure, no-recording error, busy/force, validation/traversal, real command construction), vitest 33, headless Chrome: smoke 23 checks + a real "process testv3 from the dropdown" run + switching videos.

## Known limitations (Part 10)
- Wind model is uniform in space (no shear/obstacle wake), gusts are first-order Gauss-Markov, not the full Dryden spectrum. Wind does not change rotor thrust/torque (no inflow/flapping/ground effect) and drag acts at the centre of mass (no wind-induced torque); drag is isotropic with an ASSUMED coefficient, so absolute tilt/limits are only as good as that number.
- The controller assumes it measures rotor thrust and attitude accurately (no sensor noise). Very strong gusts (sigma >= ~2 m/s on a 4 m/s mean wind) push the demand past the tilt/acceleration limit; tracking error then reaches metres and some cases cannot be saved by re-planning - this is reported (At limit %, collided flag), not hidden.
- Pre-roll adds ~2 s to wind runs (a 5 s-class flight takes ~6-9 s of server time). The planner uses only the mean wind; gusts are handled by verify-and-replan.
- Video time sync is approximate whenever the approach speed is overridden (unchanged).

## Known limitations (Part 1)
- w/h are pixels. Metric size/range come in Part 2 and depend on the configured obstacle size (default 0.5 m, ASSUMED) and FOV (82.6 deg nominal, ASSUMED).
- YOLO loses the box from ~frame 195-228 when it fills the view; TTC is extrapolated across the gap, then restarts after re-detection (fit window needs 5 samples).
- TTC assumes roughly constant closing speed and a roughly head-on approach.
- `test2.mp4` (street traffic, many fragmenting car tracks) is not a good obstacle test; `testv2.mp4` is the main scenario.
- ByteTrack is used because the default BoT-SORT's camera-motion step errored on these videos.
## Known limitations (Part 2)
- Metric scale is only as good as the configured obstacle size and FOV: range error scales 1:1 with size error, and with `tan(hfov/2)`. Pass `--width-m/--height-m/--hfov-deg` to `dronesim.vision.export` once measured.
- Assumes level, forward-pointing camera, no distortion, fronto-parallel static obstacle, head-on approach. Lateral/vertical offsets are relative to the camera axis at the reference time only.
- Real width/height in metres are inputs, not YOLO measurements: from a single camera, pixel size alone cannot give them.
- Approach speed (0.33 m/s) is slow because the clip is slow; the simulator can scale it via config for a more dynamic demo.

## Known limitations (Part 3)
- All parameters are assumed, not Tello measurements. Rotor inertia, drag coefficient and motor time constant in particular are rough guesses.
- Not modelled: rotor gyroscopic precession, blade flapping, ground effect, battery sag, prop-induced drag, ground/floor contact, motor slew limits (only first-order lag + saturation).
- Drag is isotropic quadratic; the real airframe is anisotropic.
- Pure-Python RK4 at dt = 2 ms: ~8 s of wall time for the 28 tests; a 20 s flight takes a few seconds.

## Known limitations (Part 4)
- Controller reads the true simulated state (no estimator, no sensor noise or delay). Gains are hand-tuned for the assumed parameters, not auto-tuned or robustness-analysed.
- Drag is not feed-forwarded; the integral term corrects slow offsets (will also handle constant wind later).
- The controller is a tracker: it does not plan or avoid anything. Collision avoidance is Part 5.

## Known limitations (Part 5)
- Planner assumes +x cruise, a single static box obstacle and perfect knowledge of its pose (from vision, assumed accurate); no replanning during flight, no vision noise/latency, no moving obstacles.
- Detour candidates are a fixed family on a coarse grid (not an optimal control solution). Feasibility is checked on the planned acceleration; the flown result is then verified separately.
- Start altitude (1.5 m) and floor are assumptions; in the real clip the box appears to rest on the ground, so a downward dodge may be physically invalid there - set `RunConfig.start_altitude_m` / `floor_z_m` to match reality.
- The video's approach is very slow (0.33 m/s); `RunConfig.approach_speed_mps` overrides it for dynamic demos (the video-derived TTC then no longer applies).
- Drag grows quickly with speed (about 4.8 m/s^2 at 8 m/s with the assumed coefficient), which the planner does not model; tracking error rises at speed (6.8 cm at 2 m/s).

## Known limitations (Part 6)
- At the video's 0.33 m/s the manoeuvre is so gentle that the four motor commands differ by only ~0.002 (0.2 % of full scale; roll ~3.4 deg). At 2 m/s: spread ~0.011, roll ~10 deg. Graphs need an auto-scaled y-axis and an optional "difference from hover" view to make this visible. A speed override breaks the video-time sync (scaled, approximate).
- Recording is JSON (~2 MB); fine for a browser, but not compact. Values are rounded to 6 decimals.
- Only the flown plan and the straight (no-avoidance) flight are recorded; rejected candidates are summarised as counts, not stored.

## Known limitations (Part 7)
- Single-process, no authentication, local use only (binds 127.0.0.1). `simulate` is synchronous and CPU-bound (about 3-10 s depending on speed); concurrent calls just queue in the threadpool. No job queue or caching.
- Running YOLO from the API (uploading a video) is deliberately not exposed; use `python -m dronesim.vision.export`, then `...reconstruction`, then `...run_scenario`.
- The re-simulation uses default drone parameters; drone parameters cannot be changed through the API yet.

## Next task
- Part 9: docs. Rewrite/extend the README with: what the system does, one-command-per-step quickstart (extract -> reconstruct -> run_scenario -> API -> website), architecture diagram, coordinate frames/units, equations used, the full list of assumptions and uncertainty (which numbers are measured vs configured vs assumed, Tello parameters are NOT measured), how to configure the obstacle size / camera FOV / drone parameters, how to run all tests, and known limitations. Keep `drone_readme.md` untouched (fix the yolov8n/yolov8s mismatch only by noting it).
- (Part 10, wind, is DONE - see above. The README should document the wind inputs: speed, azimuth, elevation, turbulence, and the direction convention.)
