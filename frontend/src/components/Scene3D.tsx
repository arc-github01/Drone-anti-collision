import { Grid, Line, OrbitControls } from "@react-three/drei";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { RefObject, useCallback, useEffect, useMemo, useRef } from "react";
import * as THREE from "three";
import { Playback } from "../playback";
import type { Replay } from "../replay";
import type { Recording, TrajectoryBlock, Vec3 } from "../types";
import { MOTOR_COLORS } from "./MotorGraphs";

// Simulator frame (ENU: x forward, y left, z up) -> three.js (x right, y up, z toward viewer).
// Proper rotation (det = +1): vectors map as (x, y, z) -> (x, z, -y); unit quaternions as (w, x, y, z) -> (w, x, z, -y).
export const simToThree = (p: ArrayLike<number>): [number, number, number] => [p[0], p[2], -p[1]];
const quatToThree = (q: ArrayLike<number>) => new THREE.Quaternion(q[1], q[3], -q[2], q[0]);

const flatThree = (pts: Vec3[], maxPts = 900) => {
  const step = Math.max(1, Math.ceil(pts.length / maxPts));
  const out: [number, number, number][] = [];
  for (let i = 0; i < pts.length; i += step) out.push(simToThree(pts[i]));
  out.push(simToThree(pts[pts.length - 1]));
  return out;
};

type Overlay = RefObject<HTMLDivElement | null>;

/** A text label that follows a 3D point. It lives in a plain DOM overlay next to the canvas (not inside the 3D
 *  renderer), so unmounting the scene can never throw. Returns update(point | null, text, colour). */
function useScreenLabel(overlay: Overlay, testId?: string) {
  const { camera, size } = useThree();
  const el = useMemo(() => document.createElement("div"), []);
  useEffect(() => {
    el.className = "clear-label floating";
    el.style.display = "none";
    if (testId) el.dataset.testid = testId;
    const host = overlay.current;
    host?.appendChild(el);
    return () => el.remove(); // no-op if the host is already gone
  }, [el, overlay, testId]);
  const v = useMemo(() => new THREE.Vector3(), []);
  return useCallback(
    (point: THREE.Vector3 | null, text: string, color: string) => {
      if (!point) {
        el.style.display = "none";
        return;
      }
      v.copy(point).project(camera);
      if (v.z > 1 || Math.abs(v.x) > 1.2 || Math.abs(v.y) > 1.2) {
        el.style.display = "none";
        return;
      }
      el.style.display = "block";
      el.style.transform = `translate(${(v.x * 0.5 + 0.5) * size.width}px, ${(-v.y * 0.5 + 0.5) * size.height}px) translate(-50%, -50%)`;
      if (el.textContent !== text) el.textContent = text;
      el.style.color = color;
    },
    [camera, size, el, v],
  );
}

/** Growing trail: the full path is in a fixed buffer; only the draw range moves with the playhead. */
function Trail({ replay, pb, color }: { replay: Replay; pb: Playback; color: string }) {
  const geom = useMemo(() => {
    const src = replay.allPositions;
    const arr = new Float32Array(replay.n * 3);
    for (let i = 0; i < replay.n; i++) {
      const [x, y, z] = simToThree([src[i * 3], src[i * 3 + 1], src[i * 3 + 2]]);
      arr.set([x, y, z], i * 3);
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(arr, 3));
    return g;
  }, [replay]);
  const line = useMemo(() => new THREE.Line(geom, new THREE.LineBasicMaterial({ color })), [geom, color]);
  useFrame(() => geom.setDrawRange(0, replay.indexAt(pb.time) + 1));
  return <primitive object={line} />;
}

function Obstacle({ rec }: { rec: Recording }) {
  const ob = rec.environment.obstacles[0];
  const [cx, cy, cz] = simToThree(ob.center_m);
  const [hx, hy, hz] = ob.half_extents_m;
  const infl = rec.drone.radius_m + rec.environment.safety_margin_m;
  return (
    <group position={[cx, cy, cz]}>
      <mesh>
        <boxGeometry args={[2 * hx, 2 * hz, 2 * hy]} />
        <meshStandardMaterial color="#b98d5b" roughness={0.9} />
      </mesh>
      <lineSegments>
        <edgesGeometry args={[new THREE.BoxGeometry(2 * hx, 2 * hz, 2 * hy)]} />
        <lineBasicMaterial color="#3a2a14" />
      </lineSegments>
      {/* keep-out zone: drone radius + safety margin around the panel */}
      <mesh>
        <boxGeometry args={[2 * (hx + infl), 2 * (hz + infl), 2 * (hy + infl)]} />
        <meshBasicMaterial color="#ffb703" transparent opacity={0.1} depthWrite={false} />
      </mesh>
      <lineSegments>
        <edgesGeometry args={[new THREE.BoxGeometry(2 * (hx + infl), 2 * (hz + infl), 2 * (hy + infl))]} />
        <lineBasicMaterial color="#ffb703" transparent opacity={0.45} />
      </lineSegments>
    </group>
  );
}

interface DroneProps {
  rec: Recording;
  replay: Replay;
  pb: Playback;
  modelScale: number;
  showThrust: boolean;
  showClearance: boolean;
  overlay: Overlay;
}

function Drone({ rec, replay, pb, modelScale, showThrust, showClearance, overlay }: DroneProps) {
  const group = useRef<THREE.Group>(null);
  const rotors = useRef<(THREE.Group | null)[]>([]);
  const arrows = useRef<(THREE.Mesh | null)[]>([]);
  const sphere = useRef<THREE.Mesh>(null);
  const setLabel = useScreenLabel(overlay, "clear-label");
  const mid = useMemo(() => new THREE.Vector3(), []);
  const pos = rec.drone.motor_layout.positions_xy_m;
  const margin = rec.environment.safety_margin_m;
  const ob = rec.environment.obstacles[0];

  // Decorative rotor spin: integral of the RECORDED rotor speed, slowed so it does not strobe at 60 fps.
  const angle = useMemo(() => {
    const SLOW = 0.015;
    return [0, 1, 2, 3].map((m) => {
      const w = replay.motorSeries(m, "motor_omega_radps");
      const a = new Float64Array(replay.n);
      for (let i = 1; i < replay.n; i++) a[i] = a[i - 1] + 0.5 * (w[i] + w[i - 1]) * (replay.t[i] - replay.t[i - 1]) * SLOW;
      return a;
    });
  }, [replay]);

  const clearGeom = useMemo(() => new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]), []);
  const clearLine = useMemo(() => new THREE.Line(clearGeom, new THREE.LineBasicMaterial({ color: "#7bd88f" })), [clearGeom]);

  useFrame(() => {
    const s = replay.at(pb.time);
    const g = group.current!;
    g.position.set(...simToThree(s.position));
    g.quaternion.copy(quatToThree(s.quaternion));
    const i = s.index;
    const j = Math.min(i + 1, replay.n - 1);
    const a = j === i ? 0 : (s.t - replay.t[i]) / (replay.t[j] - replay.t[i]);
    for (let m = 0; m < 4; m++) {
      const r = rotors.current[m];
      if (r) r.rotation.y = (1 - a) * angle[m][i] + a * angle[m][j];
      const ar = arrows.current[m];
      if (ar) {
        const len = s.motorThrust[m] * 0.45; // metres per newton: visual only
        ar.scale.y = Math.max(len, 1e-4) / modelScale;
        ar.position.y = (len / 2) / modelScale + 0.012;
        ar.visible = showThrust;
      }
    }
    // clearance: from the drone centre to the closest point on the panel
    const c = ob.center_m;
    const h = ob.half_extents_m;
    const near: Vec3 = [0, 1, 2].map((k) => Math.min(c[k] + h[k], Math.max(c[k] - h[k], s.position[k]))) as Vec3;
    const arr = clearGeom.attributes.position as THREE.BufferAttribute;
    arr.setXYZ(0, ...simToThree(s.position));
    arr.setXYZ(1, ...simToThree(near));
    arr.needsUpdate = true;
    clearLine.visible = showClearance;
    const col = s.clearance < 0 ? "#ff4d4d" : s.clearance < margin ? "#ffb703" : "#7bd88f";
    (clearLine.material as THREE.LineBasicMaterial).color.set(col);
    if (sphere.current) (sphere.current.material as THREE.MeshBasicMaterial).color.set(col);
    const a3 = simToThree(s.position);
    const b3 = simToThree(near);
    mid.set((a3[0] + b3[0]) / 2, (a3[1] + b3[1]) / 2, (a3[2] + b3[2]) / 2);
    setLabel(showClearance ? mid : null, `${s.clearance.toFixed(2)} m`, col);
  });

  const d = rec.drone.radius_m;
  return (
    <>
      <primitive object={clearLine} />
      <group ref={group}>
        {/* true-size collision sphere (radius from the recording) */}
        <mesh ref={sphere}>
          <sphereGeometry args={[d, 20, 14]} />
          <meshBasicMaterial color="#7bd88f" wireframe transparent opacity={0.28} />
        </mesh>
        {/* airframe, drawn modelScale x larger than life so it is visible; positions/clearance stay true-size */}
        <group scale={modelScale}>
          <mesh>
            <boxGeometry args={[0.07, 0.02, 0.05]} />
            <meshStandardMaterial color="#2b2f3a" />
          </mesh>
          <mesh position={[0.045, 0.004, 0]}>
            <boxGeometry args={[0.025, 0.012, 0.03]} />
            <meshStandardMaterial color="#e63946" />
          </mesh>
          {[45, -45].map((deg) => (
            <mesh key={deg} rotation={[0, (deg * Math.PI) / 180, 0]}>
              <boxGeometry args={[0.13, 0.008, 0.01]} />
              <meshStandardMaterial color="#454b5c" />
            </mesh>
          ))}
          {pos.map(([x, y], m) => (
            <group key={m} position={[x, 0.008, -y]}>
              <group ref={(el) => void (rotors.current[m] = el)}>
                <mesh>
                  <cylinderGeometry args={[0.034, 0.034, 0.003, 28]} />
                  <meshStandardMaterial color={MOTOR_COLORS[m]} transparent opacity={0.55} />
                </mesh>
                <mesh>
                  <boxGeometry args={[0.068, 0.004, 0.007]} />
                  <meshStandardMaterial color="#10131a" />
                </mesh>
              </group>
              <mesh ref={(el) => void (arrows.current[m] = el)}>
                <boxGeometry args={[0.006, 1, 0.006]} />
                <meshBasicMaterial color={MOTOR_COLORS[m]} />
              </mesh>
            </group>
          ))}
        </group>
      </group>
    </>
  );
}

function trajLine(t: TrajectoryBlock | undefined, color: string, opacity: number, dashed: boolean) {
  if (!t || t.position_m.length < 2) return null;
  return <Line points={flatThree(t.position_m)} color={color} lineWidth={1.6} dashed={dashed} dashSize={0.12} gapSize={0.08} transparent opacity={opacity} />;
}

/** Arrow showing the (entered) wind: it points where the wind blows TOWARD, length grows with speed. */
function WindArrow({ rec, wind, overlay }: { rec: Recording; wind: [number, number, number]; overlay: Overlay }) {
  const speed = Math.hypot(...wind);
  const arrow = useMemo(() => new THREE.ArrowHelper(new THREE.Vector3(1, 0, 0), new THREE.Vector3(), 1, 0x37d6c2, 0.18, 0.1), []);
  const setLabel = useScreenLabel(overlay, "wind-label");
  const tip = useMemo(() => new THREE.Vector3(), []);
  const dir = new THREE.Vector3(...simToThree(wind));
  if (dir.lengthSq() > 0) dir.normalize();
  const len = Math.min(3, 0.3 + 0.3 * speed);
  const base = simToThree(rec.start_m);
  const origin: [number, number, number] = [base[0] - 0.2 - dir.x * len * 0.5, base[1] + 1.0 - dir.y * len * 0.5, base[2] - dir.z * len * 0.5];
  if (speed >= 1e-6) {
    arrow.setDirection(dir);
    arrow.setLength(len, 0.18, 0.1);
  }
  useFrame(() => {
    if (speed < 1e-6) return setLabel(null, "", "");
    tip.set(origin[0] + dir.x * len, origin[1] + dir.y * len + 0.12, origin[2] + dir.z * len);
    setLabel(tip, `wind ${speed.toFixed(1)} m/s`, "#37d6c2");
  });
  if (speed < 1e-6) return null;
  return (
    <group position={origin}>
      <primitive object={arrow} />
    </group>
  );
}

export interface SceneProps {
  wind: [number, number, number];
  rec: Recording;
  replay: Replay;
  pb: Playback;
  nominalReplay: Replay | null;
  modelScale: number;
  showThrust: boolean;
  showClearance: boolean;
  showNoAvoid: boolean;
}

export function Scene3D({ rec, replay, pb, nominalReplay, modelScale, showThrust, showClearance, showNoAvoid, wind }: SceneProps) {
  const start = simToThree(rec.start_m);
  const ob = simToThree(rec.environment.obstacles[0].center_m);
  // aim at the middle of the flight path (at flight altitude) and sit behind-above-beside it
  const mid: [number, number, number] = [(start[0] + ob[0]) / 2 + 0.2, start[1] - 0.2, (start[2] + ob[2]) / 2];
  const reach = Math.max(3, Math.abs(ob[0] - start[0]) + 1.5);
  const floor = rec.environment.floor_z_m ?? 0;
  const plan = rec.planning.plan;
  const overlay = useRef<HTMLDivElement>(null);

  return (
    <>
    <div ref={overlay} className="scene-overlay" />
    <Canvas
      shadows={false}
      dpr={[1, 2]}
      camera={{ position: [mid[0] - reach * 0.38, mid[1] + 0.5 + reach * 0.16, mid[2] + reach * 0.72], fov: 50, near: 0.05, far: 200 }}
      gl={{ preserveDrawingBuffer: true }}
      data-testid="scene-canvas"
    >
      <color attach="background" args={["#0d1118"]} />
      <hemisphereLight args={["#bcd4ff", "#1a1d24", 1.1]} />
      <directionalLight position={[4, 8, 5]} intensity={1.4} />
      <Grid position={[0, floor, 0]} args={[40, 40]} infiniteGrid cellSize={0.5} sectionSize={2.5} cellColor="#252b38" sectionColor="#3b4458" fadeDistance={26} fadeStrength={1.5} />
      <Obstacle rec={rec} />
      <WindArrow rec={rec} wind={wind} overlay={overlay} />
      {trajLine(rec.planning.nominal_trajectory, "#ff6b6b", 0.85, true)}
      {plan && trajLine(plan.trajectory, "#5aa9ff", 0.9, true)}
      {showNoAvoid && nominalReplay && <Trail replay={nominalReplay} pb={pb} color="#ff4d4d" />}
      <Trail replay={replay} pb={pb} color="#ffffff" />
      <Drone rec={rec} replay={replay} pb={pb} modelScale={modelScale} showThrust={showThrust} showClearance={showClearance} overlay={overlay} />
      <OrbitControls target={mid} enableDamping makeDefault maxDistance={60} />
    </Canvas>
    </>
  );
}
