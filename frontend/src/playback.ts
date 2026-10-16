// Playback clock. It only moves a playhead through RECORDED simulation time; physics never runs in the
// browser, so the physics timestep is fixed (in the recording) and independent of the render frame rate.
import { useEffect, useState } from "react";

export class Playback {
  private t = 0;
  private dur = 0;
  private _playing = false;
  private _speed = 1;
  private _loop = true;
  private raf = 0;
  private last = 0;
  private listeners = new Set<() => void>();

  get time() {
    return this.t;
  }
  get duration() {
    return this.dur;
  }
  get playing() {
    return this._playing;
  }
  get speed() {
    return this._speed;
  }
  get loop() {
    return this._loop;
  }

  subscribe(fn: () => void) {
    this.listeners.add(fn);
    return () => {
      this.listeners.delete(fn);
    };
  }
  private emit() {
    this.listeners.forEach((f) => f());
  }

  setDuration(d: number) {
    this.dur = Math.max(0, d);
    if (this.t > this.dur) this.t = this.dur;
    this.emit();
  }
  seek(t: number) {
    this.t = Math.min(this.dur, Math.max(0, t));
    this.emit();
  }
  setSpeed(s: number) {
    this._speed = s;
    this.emit();
  }
  setLoop(l: boolean) {
    this._loop = l;
    this.emit();
  }
  step(dt: number) {
    this.pause();
    this.seek(this.t + dt);
  }

  play() {
    if (this._playing) return;
    if (this.t >= this.dur) this.t = 0;
    this._playing = true;
    this.last = performance.now();
    const tick = (now: number) => {
      if (!this._playing) return;
      const dtWall = Math.min(0.1, (now - this.last) / 1000); // clamp so a stalled tab doesn't jump
      this.last = now;
      this.t += dtWall * this._speed;
      if (this.t >= this.dur) {
        if (this._loop) this.t = 0;
        else {
          this.t = this.dur;
          this._playing = false;
        }
      }
      this.emit();
      if (this._playing) this.raf = requestAnimationFrame(tick);
    };
    this.raf = requestAnimationFrame(tick);
    this.emit();
  }
  pause() {
    if (!this._playing) return;
    this._playing = false;
    cancelAnimationFrame(this.raf);
    this.emit();
  }
  toggle() {
    if (this._playing) this.pause();
    else this.play();
  }
}

/** Re-render the calling component whenever playback changes (the playhead moves). */
export function usePlayback(pb: Playback) {
  const [, force] = useState(0);
  useEffect(() => pb.subscribe(() => force((n) => n + 1)), [pb]);
  return pb;
}
