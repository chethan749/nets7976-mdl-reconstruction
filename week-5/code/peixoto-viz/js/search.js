// The moves of Peixoto (2025) §V, one at a time, exactly as the paper specifies them, for the step-by-step view (§5 of
// the page). The fit in fits.js uses faster deterministic steps; here every proposal is searched by random bisection
// (Ref. [63]) and evaluated with the exact change in Σ. No DOM code, so it runs in Node too.
//
// Every builder returns a plan: { kind, ..., g } where g(x) is ΔΣ in bits for the proposal x (negative = shorter
// description = accepted), with "data" and "prior" parts available from parts(x).

import { MDL, LAMBDA, priorParts, sumParts } from "./fits.js";

const LN2 = Math.log(2), R = 0.381966; // golden-section fraction, for the deterministic search
export const WMAX = 1; // allowed range of weights for this problem: [−1, 1]

// Bracketing search for the minimum of g on [lo, hi]. Keeps a bracket [a, c] and the best point b inside it.
// random: each new probe is uniform in the bracket; otherwise it is the golden-section point of the larger side.
export function bisect(g, lo, hi, { rng, random = true, tol = 2e-3, maxIter = 60 } = {}) {
  let a = lo, c = hi, b = random ? a + (c - a) * rng.uniform() : a + (1 - R) * (c - a), gb = g(b);
  const steps = [{ a, c, x: b, gx: gb, b, gb, better: true }];
  while (c - a > tol && steps.length < maxIter) {
    const x = random ? a + (c - a) * rng.uniform() : c - b > b - a ? b + R * (c - b) : b - R * (b - a);
    const gx = g(x), better = gx < gb;
    if (better) { if (x < b) c = b; else a = b; b = x; gb = gx; }
    else if (x < b) a = x; else c = x;
    steps.push({ a, c, x, gx, b, gb, better });
  }
  return { steps, x: b, g: gb };
}

// The same on the integers 0..n−1 (a sorted list of categories). Stops when every index in the bracket is probed.
export function bisectDiscrete(g, n, { rng, random = true } = {}) {
  const seen = new Map(), G = (i) => { if (!seen.has(i)) seen.set(i, g(i)); return seen.get(i); };
  let a = 0, c = n - 1, b = random ? Math.floor(rng.uniform() * n) : Math.floor((n - 1) / 2), gb = G(b);
  const steps = [{ a, c, x: b, gx: gb, b, gb, better: true }];
  for (;;) {
    const free = [];
    for (let i = a; i <= c; i++) if (!seen.has(i)) free.push(i);
    if (!free.length) break;
    let x;
    if (random) x = free[Math.floor(rng.uniform() * free.length)];
    else { const t = c - b > b - a ? b + Math.ceil((c - b) / 2) : b - Math.ceil((b - a) / 2); x = free.reduce((u, v) => (Math.abs(v - t) < Math.abs(u - t) ? v : u)); }
    const gx = G(x), better = gx < gb;
    if (better) { if (x < b) c = b; else a = b; b = x; gb = gx; }
    else if (x < b) a = x; else c = x;
    steps.push({ a, c, x, gx, b, gb, better });
  }
  return { steps, x: b, g: gb };
}

// An MDL object holding exactly the state of one snapshot of the fit
export function stateFrom(F, snap, seed = 1) {
  const W = new Float64Array(F.P);
  for (let p = 0; p < F.P; p++) W[p] = snap.cat[p] >= 0 ? snap.z[snap.cat[p]] : 0;
  F.setW(W);
  const mdl = new MDL(F, seed);
  mdl.cat.set(snap.cat); mdl.z = snap.z.slice(); mdl.m = snap.m.slice(); mdl.E = snap.E; mdl.LL = F.logLik();
  return mdl;
}

const bits = (n) => n / LN2;

// ---------------------------------------------------------------------------
// Step 0: candidate pairs, Eqs. (28)–(29)
// ---------------------------------------------------------------------------

// Score every zero pair: the best change in log-posterior from switching it on at the smallest positive or
// smallest-magnitude negative category (Eq. 28), or the gradient magnitude if there are no categories yet (Eq. 29).
export function candidates(mdl, kappa = 1) {
  const F = mdl.F, P0 = mdl.prior(), K = mdl.z.length;
  const pos = mdl.z.map((v, k) => [v, k]).filter(([v]) => v > 0).sort((a, b) => a[0] - b[0])[0];
  const neg = mdl.z.map((v, k) => [v, k]).filter(([v]) => v < 0).sort((a, b) => b[0] - a[0])[0];
  const scores = [];
  for (let p = 0; p < F.P; p++) {
    if (mdl.cat[p] >= 0) continue;
    let s;
    if (!K) s = Math.abs(F.grad(p));
    else {
      s = -Infinity;
      for (const opt of [pos, neg]) if (opt) s = Math.max(s, F.deltaPair(p, opt[0]) - (mdl.priorMove(-1, opt[1]) - P0));
      s = bits(s);
    }
    scores.push({ p, s });
  }
  scores.sort((a, b) => b.s - a.s);
  const n = Math.round(kappa * F.N);
  return { scores, keep: scores.slice(0, n).map((r) => r.p), gradient: !K, n };
}

// ---------------------------------------------------------------------------
// Step 1: edge updates on one pair, options (1)–(3)
// ---------------------------------------------------------------------------

export function edgeToCategory(mdl, p) {
  const F = mdl.F, c = mdl.cat[p], w = F.W[p], P0 = mdl.prior();
  const order = mdl.z.map((v, k) => k).filter((k) => k !== c).sort((a, b) => mdl.z[a] - mdl.z[b]);
  if (!order.length) return null;
  const parts = (i) => { const k = order[i]; return { data: -bits(F.deltaPair(p, mdl.z[k] - w)), prior: bits(mdl.priorMove(c, k) - P0) }; };
  const g = (i) => { const q = parts(i); return q.data + q.prior; };
  return { kind: "discrete", p, from: w, xs: order.map((k) => mdl.z[k]), counts: order.map((k) => mdl.m[k]), n: order.length, g, parts };
}

export function edgeToNewValue(mdl, p) {
  const F = mdl.F, c = mdl.cat[p], w = F.W[p], P0 = mdl.prior(), K = mdl.z.length;
  const parts = (v) => ({ data: -bits(F.deltaPair(p, v - w)), prior: bits(mdl.priorMove(c, K, v) - P0) });
  const g = (v) => { const q = parts(v); return q.data + q.prior; };
  return { kind: "continuous", p, from: w, lo: -WMAX, hi: WMAX, g, parts };
}

export function edgeToZero(mdl, p) {
  const F = mdl.F, c = mdl.cat[p], w = F.W[p];
  if (c < 0) return null;
  const q = { data: -bits(F.deltaPair(p, -w)), prior: bits(mdl.priorMove(c, -1) - mdl.prior()) };
  return { kind: "single", p, from: w, to: 0, ...q, total: q.data + q.prior, removed: [p], added: [] };
}

// ---------------------------------------------------------------------------
// Steps 2–3: edge moves and swaps (no search: one proposal, prior unchanged)
// ---------------------------------------------------------------------------

export function edgeMove(mdl, rng, cand) {
  const F = mdl.F, N = F.N;
  const nodes = [];
  for (let i = 0; i < N; i++) for (let u = 0; u < N; u++) if (u !== i && mdl.cat[F.idx[i * N + u]] >= 0) { nodes.push(i); break; }
  for (let tries = 0; tries < 50; tries++) {
    const i = nodes[Math.floor(rng.uniform() * nodes.length)];
    const inc = [], free = [];
    for (let u = 0; u < N; u++) {
      if (u === i) continue;
      const q = F.idx[i * N + u];
      if (mdl.cat[q] >= 0) inc.push(q); else if (cand.has(q)) free.push(q);
    }
    if (!inc.length || !free.length) continue;
    const pe = inc[Math.floor(rng.uniform() * inc.length)], pf = free[Math.floor(rng.uniform() * free.length)], w = F.W[pe];
    const data = -bits(F.deltaMulti([[pe, -w], [pf, w]]));
    return { kind: "single", node: i, removed: [pe], added: [pf], data, prior: 0, total: data };
  }
  return null;
}

export function edgeSwap(mdl, rng) {
  const F = mdl.F, N = F.N, nz = mdl.nonzero();
  for (let tries = 0; tries < 200; tries++) {
    const p = nz[Math.floor(rng.uniform() * nz.length)], q = nz[Math.floor(rng.uniform() * nz.length)];
    const i = F.I[p], j = F.J[p], u = F.I[q], v = F.J[q];
    if (new Set([i, j, u, v]).size < 4) continue;
    const iv = F.idx[i * N + v], uj = F.idx[u * N + j], W = F.W;
    const ch = [[iv, W[p] - W[iv]], [p, W[iv] - W[p]], [uj, W[q] - W[uj]], [q, W[uj] - W[q]]];
    const data = -bits(F.deltaMulti(ch));
    const added = [iv, uj].filter((x) => W[x] === 0), removed = [p, q].filter((x) => W[x === p ? iv : uj] === 0);
    return { kind: "single", removed, added, data, prior: 0, total: data };
  }
  return null;
}

// ---------------------------------------------------------------------------
// Steps 4–5: category values, merge, split, merge-split
// ---------------------------------------------------------------------------

function valuesDelta(F, groups) {
  const ch = [];
  for (const [mem, to] of groups) for (const p of mem) ch.push([p, to - F.W[p]]);
  return F.deltaMulti(ch);
}

export function categoryValue(mdl, k) {
  const F = mdl.F, mem = mdl.members(k), z = mdl.z[k];
  const parts = (v) => ({ data: -bits(valuesDelta(F, [[mem, v]])), prior: bits(LAMBDA * (Math.abs(v) - Math.abs(z))) });
  const g = (v) => { const q = parts(v); return q.data + q.prior; };
  return { kind: "continuous", k, from: z, members: mem, lo: -WMAX, hi: WMAX, g, parts };
}

export function merge(mdl, rng) {
  const K = mdl.z.length;
  if (K < 2) return null;
  const ord = mdl.z.map((v, k) => k).sort((a, b) => mdl.z[a] - mdl.z[b]);
  const r = Math.floor(rng.uniform() * (K - 1)), a = ord[r], b = ord[r + 1];
  const F = mdl.F, mem = mdl.members(a).concat(mdl.members(b)), P0 = mdl.prior();
  const m = mdl.m.slice(), z = mdl.z.slice();
  m[a] += m[b]; m.splice(b, 1); z.splice(b, 1);
  const ia = a > b ? a - 1 : a;
  const parts = (v) => { z[ia] = v; return { data: -bits(valuesDelta(F, [[mem, v]])), prior: bits(sumParts(priorParts(F.P, mdl.E, m, z)) - P0) }; };
  const g = (v) => { const q = parts(v); return q.data + q.prior; };
  return { kind: "continuous", pair: [a, b], from: (mdl.z[a] * mdl.m[a] + mdl.z[b] * mdl.m[b]) / (mdl.m[a] + mdl.m[b]), members: mem, lo: -WMAX, hi: WMAX, g, parts };
}

// Split category k as in §V: two values drawn uniformly in [z_{k−1}, z_{k+1}] (its neighbours in sorted order), then
// entries moved one at a time between the two whenever that lowers Σ, and both values re-searched by random bisection,
// until nothing changes. Returns the frames for the animation; F is left as it was.
export function split(mdl, k, rng, { random = true, rounds = 8 } = {}) {
  const F = mdl.F, mem = mdl.members(k);
  if (mem.length < 2) return null;
  const sorted = mdl.z.slice().sort((a, b) => a - b), r = sorted.indexOf(mdl.z[k]);
  const lo = r > 0 ? sorted[r - 1] : -WMAX, hi = r < sorted.length - 1 ? sorted[r + 1] : WMAX;
  const W0 = Float64Array.from(F.W), LL0 = F.logLik(), P0 = mdl.prior();
  const val = [lo + (hi - lo) * rng.uniform(), lo + (hi - lo) * rng.uniform()];
  const side = mem.map(() => (rng.uniform() < 0.5 ? 0 : 1));
  // the prior of the split state: category k replaced by the two new ones
  const priorNow = () => {
    const n = [side.filter((s) => s === 0).length, side.filter((s) => s === 1).length];
    const m = mdl.m.slice(), z = mdl.z.slice();
    m.splice(k, 1); z.splice(k, 1);
    for (const s of [0, 1]) if (n[s]) { m.push(n[s]); z.push(val[s]); }
    return sumParts(priorParts(F.P, mdl.E, m, z));
  };
  const setAll = () => { const ch = mem.map((p, e) => [p, val[side[e]] - F.W[p]]); F.applyMulti(ch); mem.forEach((p, e) => { F.W[p] = val[side[e]]; }); };
  const sigma = () => bits(-(F.logLik() - LL0) + priorNow() - P0);
  setAll();
  const frames = [{ step: "two random values", val: val.slice(), side: side.slice(), dS: sigma() }];
  for (let it = 0; it < rounds; it++) {
    let moved = 0;
    mem.forEach((p, e) => {
      const before = priorNow(), to = 1 - side[e];
      const dLL = F.deltaPair(p, val[to] - F.W[p]);
      side[e] = to;
      const dS = -dLL + priorNow() - before;
      if (dS < -1e-9) { F.apply(p, val[to] - F.W[p]); F.W[p] = val[to]; moved++; } else side[e] = 1 - to;
    });
    frames.push({ step: `move entries (${moved} moved)`, val: val.slice(), side: side.slice(), dS: sigma() });
    let changed = 0;
    for (const s of [0, 1]) {
      const grp = mem.filter((_, e) => side[e] === s);
      if (!grp.length) continue;
      const base = val[s];
      const res = bisect((v) => -valuesDelta(F, [[grp, v]]) + LAMBDA * (Math.abs(v) - Math.abs(base)), -WMAX, WMAX, { rng, random });
      if (res.g < -1e-9) { val[s] = res.x; changed++; }
      setAll();
    }
    frames.push({ step: `bisect both values`, val: val.slice(), side: side.slice(), dS: sigma() });
    if (!moved && !changed) break;
  }
  F.setW(W0);
  const last = frames[frames.length - 1];
  return { kind: "split", k, members: mem, frames, lo, hi, total: last.dS, from: mdl.z[k] };
}

// Merge-split: merge two neighbours (value by bisection), then split the merged category again
export function mergeSplit(mdl, rng, { random = true } = {}) {
  const mp = merge(mdl, rng);
  if (!mp) return null;
  const res = bisect(mp.g, mp.lo, mp.hi, { rng, random });
  // apply the merge to a copy of the state, split it, then put everything back
  const F = mdl.F, W0 = Float64Array.from(F.W), save = { cat: Int16Array.from(mdl.cat), z: mdl.z.slice(), m: mdl.m.slice(), LL: mdl.LL };
  const [a, b] = mp.pair, P0 = mdl.prior(), LL0 = F.logLik();
  F.applyMulti(mp.members.map((p) => [p, res.x - F.W[p]])); mp.members.forEach((p) => { F.W[p] = res.x; });
  for (let p = 0; p < F.P; p++) if (mdl.cat[p] === b) mdl.cat[p] = a;
  mdl.z[a] = res.x; mdl.m[a] += mdl.m[b]; mdl.m[b] = 0; mdl.LL = F.logLik(); mdl.prune(b);
  const ka = mdl.cat[mp.members[0]], dMerge = bits(-(mdl.LL - LL0) + mdl.prior() - P0);
  const sp = split(mdl, ka, rng, { random });
  // restore
  F.setW(W0); mdl.cat.set(save.cat); mdl.z = save.z; mdl.m = save.m; mdl.LL = save.LL;
  if (!sp) return null;
  sp.frames = sp.frames.map((f) => ({ ...f, dS: f.dS + dMerge }));
  return { ...sp, kind: "split", merged: [a, b], mergeValue: res.x, mergeSearch: res, total: sp.frames[sp.frames.length - 1].dS };
}
