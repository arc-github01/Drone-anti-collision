// Pure helpers for the motor graphs (unit-tested in graphMath.test.ts).

/** y-range covering all series, padded, never narrower than `minSpan` (so tiny differences are still drawn
 *  at a readable scale but a flat line is not blown up into noise). */
export function yRange(series: ArrayLike<number>[], minSpan: number, padFrac = 0.12): [number, number] {
  let lo = Infinity;
  let hi = -Infinity;
  for (const s of series)
    for (let i = 0; i < s.length; i++) {
      const v = s[i];
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
  if (!isFinite(lo)) return [0, 1];
  let span = hi - lo;
  if (span < minSpan) {
    const mid = (lo + hi) / 2;
    lo = mid - minSpan / 2;
    hi = mid + minSpan / 2;
    span = minSpan;
  }
  return [lo - span * padFrac, hi + span * padFrac];
}

/** Min/max decimation: at most ~2 points per pixel column, but every local extreme is kept,
 *  so short spikes in a motor command never disappear when the graph is narrow. */
export function decimate(t: ArrayLike<number>, v: ArrayLike<number>, px: number): { t: number[]; v: number[] } {
  const n = t.length;
  const cols = Math.max(1, Math.floor(px));
  if (n <= cols * 2) return { t: Array.from(t), v: Array.from(v) };
  const t0 = t[0];
  const span = t[n - 1] - t0 || 1;
  const outT: number[] = [];
  const outV: number[] = [];
  let i = 0;
  for (let c = 0; c < cols && i < n; c++) {
    const tEnd = t0 + (span * (c + 1)) / cols;
    let iMin = i;
    let iMax = i;
    let j = i;
    while (j < n && (t[j] <= tEnd || c === cols - 1)) {
      if (v[j] < v[iMin]) iMin = j;
      if (v[j] > v[iMax]) iMax = j;
      j++;
    }
    if (j === i) continue;
    const [a, b] = iMin <= iMax ? [iMin, iMax] : [iMax, iMin];
    outT.push(t[a]);
    outV.push(v[a]);
    if (b !== a) {
      outT.push(t[b]);
      outV.push(v[b]);
    }
    i = j;
  }
  return { t: outT, v: outV };
}

export const timeToX = (t: number, t0: number, t1: number, x0: number, x1: number) =>
  x0 + ((t - t0) / (t1 - t0 || 1)) * (x1 - x0);

export const xToTime = (x: number, t0: number, t1: number, x0: number, x1: number) =>
  Math.min(t1, Math.max(t0, t0 + ((x - x0) / (x1 - x0 || 1)) * (t1 - t0)));

/** "Nice" tick positions for an axis. */
export function ticks(lo: number, hi: number, target = 5): number[] {
  const raw = (hi - lo) / Math.max(1, target);
  const mag = Math.pow(10, Math.floor(Math.log10(raw || 1)));
  const step = [1, 2, 5, 10].map((m) => m * mag).find((s) => s >= raw) ?? mag * 10;
  const out: number[] = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-12; v += step) out.push(+v.toPrecision(12));
  return out;
}
