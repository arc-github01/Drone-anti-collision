// Shapes of the JSON served by the backend (see dronesim/sim/recording.py and dronesim/vision/*).

export type Vec3 = [number, number, number];
export type Vec4 = [number, number, number, number];

export interface FlightBlock {
  t: number[];
  position_m: Vec3[];
  velocity_mps: Vec3[];
  quaternion_wxyz: Vec4[];
  euler_rpy_deg: Vec3[];
  body_rates_radps: Vec3[];
  motor_command: Vec4[];
  motor_omega_radps: Vec4[];
  motor_thrust_n: Vec4[];
  motor_power_w: Vec4[];
  setpoint_position_m: Vec3[];
  clearance_m: number[];
  saturated: boolean[];
  limited?: boolean[]; // tilt/acceleration limit clipped the demand (absent in older recordings)
  wind_mps?: Vec3[]; // wind velocity used by the drag law (absent in older recordings)
  disturbance_est_mps2?: Vec3[];
}

export interface TrajectoryBlock {
  t: number[];
  position_m: Vec3[];
  velocity_mps: Vec3[];
}

export interface Summary {
  min_clearance_m: number;
  t_min_clearance_s: number;
  collided: boolean;
  max_tilt_deg: number;
  peak_speed_mps: number;
  max_tracking_error_m: number;
  saturated_fraction: number;
  limited_fraction?: number;
}

export interface WindInfo {
  config: { speed_mps: number; azimuth_deg: number; elevation_deg: number; turbulence_sigma_mps: number; turbulence_tau_s: number; seed: number };
  mean_vector_mps: Vec3;
  components_approach_frame: { headwind_mps: number; crosswind_left_mps: number; vertical_up_mps: number };
  station_keeping: { accel_mps2: number; thrust_n: number; tilt_deg: number };
  preroll_s: number;
  convention: string;
  model: string;
}

export interface Recording {
  schema: string;
  physics: { dt_s: number; sample_dt_s: number; control_hz: number };
  drone: {
    params: Record<string, number | number[]>;
    hover_command: number;
    radius_m: number;
    motor_layout: { positions_xy_m: [number, number][]; spin_dir: number[]; names: string[] };
  };
  environment: {
    obstacles: { center_m: Vec3; half_extents_m: Vec3 }[];
    safety_margin_m: number;
    floor_z_m: number | null;
  };
  scenario: { approach_speed_mps: number; initial_range_m: number; initial_ttc_s: number | null };
  start_m: Vec3;
  approach_speed_mps: number;
  video_sync: { reference_video_time_s: number; time_scale: number | null };
  assumptions: string[];
  wind?: WindInfo;
  planning: {
    attempts?: { extra_buffer_m: number; min_clearance_m: number }[];
    nominal_limits?: { peak_tilt_deg: number; peak_thrust_n: number; peak_accel_mps2: number } | null;
    nominal_violations?: string[];
    needs_avoidance: boolean;
    note: string;
    n_candidates: number;
    n_feasible: number;
    rejections: Record<string, number>;
    plan: null | {
      kind: string;
      params: Record<string, number>;
      metrics: Record<string, number>;
      trajectory: TrajectoryBlock;
    };
    nominal_trajectory: TrajectoryBlock;
  };
  flight: FlightBlock | null;
  summary: Summary | null;
  nominal_flight: FlightBlock | null;
  nominal_summary: Summary | null;
}

export interface TrackSample {
  frame: number;
  t: number;
  detected: boolean;
  conf: number;
  bbox: [number, number, number, number] | null;
  w_px: number | null;
  h_px: number | null;
  growth_px_s: number | null;
  ttc_s: number | null;
  ttc_source: string;
}

export interface Track {
  video: { path: string; fps: number; width: number; height: number; n_frames: number };
  target: { track_id: number; yolo_class: string | null; detected_frames: number };
  other_tracks?: { track_id: number; yolo_class: string; frames: number; first_frame: number; last_frame: number; max_area_px: number }[];
  samples: TrackSample[];
}

export interface MetricSample {
  frame: number;
  t: number;
  source: string;
  range_m: number | null;
  closing_speed_mps: number | null;
  width_m: number | null;
  height_m: number | null;
}

export interface Metric {
  scenario: { approach_speed_mps: number; initial_range_m: number; initial_ttc_s: number | null } | null;
  samples: MetricSample[];
}

export interface RunInfo {
  name: string;
  track: boolean;
  metric: boolean;
  recording: boolean;
  video: boolean;
  file?: string | null; // video file name when the video is in the project folder
  size_mb?: number | null;
}

export interface Job {
  id: string;
  video: string;
  run: string;
  status: "running" | "done" | "error";
  step: string;
  step_index: number;
  n_steps: number;
  error: string | null;
  log: string[];
}

export interface ProcessRequest {
  width_m?: number;
  height_m?: number;
  hfov_deg?: number;
  conf?: number;
  force?: boolean;
}

export interface SimulateRequest {
  speed_mps?: number;
  dodge?: "auto" | "left" | "right" | "up" | "down";
  altitude_m?: number;
  floor_m?: number;
  no_floor?: boolean;
  margin_m?: number;
  mass_kg?: number;
  max_thrust_per_motor_n?: number;
  wind_speed_mps?: number;
  wind_azimuth_deg?: number;
  wind_elevation_deg?: number;
  turbulence_sigma_mps?: number;
  turbulence_tau_s?: number;
  wind_seed?: number;
  save?: boolean;
}

export type MotorQuantity = "motor_command" | "motor_omega_radps" | "motor_thrust_n" | "motor_power_w";
