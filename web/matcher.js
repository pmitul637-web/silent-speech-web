// Mouth-movement matcher. Shared by app.js and the offline benchmark, so a
// score measured on the laptop is the same code that runs on the phone.
//
// Compared with the original single-nearest-clip DTW it:
//  - halves the frame rate before matching (4x less work, no accuracy loss),
//  - scales each blendshape by how much it varies, so small but telling
//    movements (pucker, press) count as much as jaw opening,
//  - can score a phrase by its few closest clips rather than one, so a
//    single odd clip can't cause a wrong match,
//  - limits how far DTW may stretch time, and stops early on clips that
//    can no longer win.
// Velocity features and per-clip centring are available but off: neither
// helped on real recorded clips.

export const DEFAULTS = {
  step: 2,           // average every N camera frames
  center: false,     // subtract each clip's own average shape
  zscore: true,      // scale each blendshape by its spread across all clips
  deltaWeight: 0,    // weight of velocity features (0 = off; didn't help on real clips)
  k: 1,              // a phrase's score = mean of its k closest clips (1 tested best)
  band: 0.4,         // max time stretch, as a fraction of clip length
};

function downsample(frames, step) {
  if (step <= 1) return frames.map((f) => Float32Array.from(f));
  const out = [];
  for (let i = 0; i < frames.length; i += step) {
    const end = Math.min(i + step, frames.length);
    const dim = frames[i].length;
    const v = new Float32Array(dim);
    for (let t = i; t < end; t++) for (let d = 0; d < dim; d++) v[d] += frames[t][d];
    for (let d = 0; d < dim; d++) v[d] /= end - i;
    out.push(v);
  }
  return out;
}

// Spread of each blendshape over every reference frame, used to put them on
// one scale. Floored so a channel that never moves can't blow up.
export function channelScale(clipsByName) {
  let dim = 0;
  let n = 0;
  let sum;
  let sq;
  for (const clips of Object.values(clipsByName)) {
    for (const clip of clips) {
      for (const f of clip) {
        if (!sum) { dim = f.length; sum = new Float64Array(dim); sq = new Float64Array(dim); }
        for (let d = 0; d < dim; d++) { sum[d] += f[d]; sq[d] += f[d] * f[d]; }
        n++;
      }
    }
  }
  if (!n) return null;
  const scale = new Float32Array(dim);
  for (let d = 0; d < dim; d++) {
    const mean = sum[d] / n;
    const sd = Math.sqrt(Math.max(0, sq[d] / n - mean * mean));
    scale[d] = 1 / Math.max(sd, 0.02);
  }
  return scale;
}

export function features(frames, scale, opt = DEFAULTS) {
  const x = downsample(frames, opt.step);
  const T = x.length;
  if (!T) return [];
  const dim = x[0].length;

  if (opt.center) {
    const mean = new Float32Array(dim);
    for (const f of x) for (let d = 0; d < dim; d++) mean[d] += f[d] / T;
    for (const f of x) for (let d = 0; d < dim; d++) f[d] -= mean[d];
  }
  if (opt.zscore && scale) {
    for (const f of x) for (let d = 0; d < dim; d++) f[d] *= scale[d];
  }
  if (!opt.deltaWeight) return x;

  const out = new Array(T);
  for (let t = 0; t < T; t++) {
    const prev = x[Math.max(0, t - 1)];
    const next = x[Math.min(T - 1, t + 1)];
    const v = new Float32Array(dim * 2);
    v.set(x[t]);
    for (let d = 0; d < dim; d++) v[dim + d] = opt.deltaWeight * (next[d] - prev[d]);
    out[t] = v;
  }
  return out;
}

function frameDist(a, b) {
  let s = 0;
  for (let d = 0; d < a.length; d++) {
    const diff = a[d] - b[d];
    s += diff * diff;
  }
  return Math.sqrt(s);
}

// DTW distance, normalised by path length. `abandon` is a normalised score
// this pair must beat; once every cell in a row is already worse, it gives
// up and returns Infinity. `openEnd` matches the query against the START of
// the reference, for guessing while a phrase is still being mouthed.
export function dtw(a, b, band = DEFAULTS.band, abandon = Infinity, openEnd = false) {
  const n = a.length;
  const m = b.length;
  if (!n || !m) return Infinity;
  const w = openEnd
    ? m
    : Math.max(Math.abs(n - m), Math.ceil(band * Math.max(n, m)));
  const limit = abandon * (n + m);
  let prev = new Float64Array(m + 1).fill(Infinity);
  let cur = new Float64Array(m + 1).fill(Infinity);
  prev[0] = 0;
  for (let i = 1; i <= n; i++) {
    cur.fill(Infinity);
    const centre = openEnd ? i : Math.round((i * m) / n);
    const lo = Math.max(1, centre - w);
    const hi = Math.min(m, centre + w);
    let rowMin = Infinity;
    for (let j = lo; j <= hi; j++) {
      const best = Math.min(prev[j], cur[j - 1], prev[j - 1]);
      if (best === Infinity) continue;
      const c = frameDist(a[i - 1], b[j - 1]) + best;
      cur[j] = c;
      if (c < rowMin) rowMin = c;
    }
    if (rowMin > limit) return Infinity;
    [prev, cur] = [cur, prev];
  }
  if (!openEnd) return prev[m] / (n + m);
  let best = Infinity;
  const from = Math.max(1, Math.ceil(n * 0.4));
  for (let j = from; j <= m; j++) {
    const v = prev[j] / (n + j);
    if (v < best) best = v;
  }
  return best;
}

// Turn {phrase: [clip, ...]} into something fast to match against.
export function buildIndex(clipsByName, opt = DEFAULTS) {
  const scale = opt.zscore ? channelScale(clipsByName) : null;
  const entries = [];
  for (const [name, clips] of Object.entries(clipsByName)) {
    if (!Array.isArray(clips) || !clips.length) continue;
    entries.push({ name, feats: clips.map((c) => features(c, scale, opt)) });
  }
  const index = { opt, scale, entries, spread: {} };
  index.spread = phraseSpread(index);
  return index;
}

// For each phrase, how far its own clips typically sit from each other.
// "Is this close enough?" is judged against this, so the strictness setting
// means the same thing for a tight phrase and a sloppy one.
// Uses the 10 newest clips: all-pairs work grows with the square of clips.
function phraseSpread(index) {
  const out = {};
  for (const e of index.entries) {
    const feats = e.feats.slice(-10);
    const near = [];
    for (let i = 0; i < feats.length; i++) {
      let best = Infinity;
      for (let j = 0; j < feats.length; j++) {
        if (i === j) continue;
        const d = dtw(feats[i], feats[j], index.opt.band, best);
        if (d < best) best = d;
      }
      if (best < Infinity) near.push(best);
    }
    near.sort((a, b) => a - b);
    out[e.name] = near.length ? near[Math.floor(near.length / 2)] : null;
  }
  return out;
}

// Phrases with their scores, closest first. Only the top `keep` are
// guaranteed to be present: with k = 1, a phrase that can no longer reach
// the top `keep` is dropped part-way, which is most of the speed-up once
// many phrases are trained.
export function rank(index, frames, { openEnd = false, keep = 3 } = {}) {
  const q = features(frames, index.scale, index.opt);
  if (!q.length) return [];
  const k = index.opt.k;
  const out = [];
  const cutoff = () => {
    if (k !== 1 || out.length < keep) return Infinity;
    return out.map((o) => o.distance).sort((a, b) => a - b)[keep - 1];
  };
  for (const e of index.entries) {
    const top = [];   // k smallest distances, sorted
    for (const f of e.feats) {
      const own = top.length >= k ? top[top.length - 1] : Infinity;
      const bar = Math.min(own, cutoff());
      const d = dtw(q, f, index.opt.band, bar, openEnd);
      if (d >= bar) continue;
      top.push(d);
      top.sort((a, b) => a - b);
      if (top.length > k) top.pop();
    }
    if (!top.length) continue;
    const distance = top.reduce((s, d) => s + d, 0) / top.length;
    out.push({ name: e.name, distance });
  }
  out.sort((a, b) => a.distance - b.distance);
  return out;
}

// How far ahead of the runner-up the winner is: 1 = a tie, 2 = twice as close.
export function margin(ranked) {
  if (!ranked.length) return 0;
  if (ranked.length < 2) return Infinity;
  return ranked[1].distance / Math.max(ranked[0].distance, 1e-9);
}

// Is the winner close enough to its own phrase's clips to accept at all?
// `strictness` is how many times the phrase's normal spread we allow.
export function accepted(index, ranked, strictness) {
  if (!ranked.length) return false;
  const spread = index.spread[ranked[0].name];
  if (spread == null) return true;   // only one clip: nothing to compare to
  return ranked[0].distance <= spread * strictness;
}
