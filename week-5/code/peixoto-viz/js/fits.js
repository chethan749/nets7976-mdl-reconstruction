// The three ways of choosing Ŵ that the page compares, all on a Fit from ising.js:
//   L1 (Laplace prior, Eqs. 7–10), MDL (the quantized hierarchical prior, Eq. 27, fitted as in §V),
//   and a normal density prior on the nonzero weights (Eq. 30).
// Everything is in nats; the page converts to bits.

import { Fit, makeRng, similarity, countEdges } from "./ising.js";

const LN2 = Math.log(2);

// ---------------------------------------------------------------------------
// L1: coordinate descent (one proximal Newton step per entry), over a path of λ
// ---------------------------------------------------------------------------

export function l1Fit(F, lam, { tol = 1e-4, maxSweeps = 200 } = {}) {
  const { P, W } = F;
  for (let s = 0; s < maxSweeps; s++) {
    let maxd = 0;
    for (let p = 0; p < P; p++) {
      const w = W[p];
      if (w === 0 && Math.abs(F.grad(p)) <= lam) continue; // stays exactly zero (subgradient condition)
      const [g, c] = F.gradCurv(p);
      const z = w + g / c, nw = Math.sign(z) * Math.max(Math.abs(z) - lam / c, 0);
      const d = nw - w;
      if (d !== 0) { F.apply(p, d); maxd = Math.max(maxd, Math.abs(d)); }
    }
    if (maxd < tol) return s + 1;
  }
  return maxSweeps;
}

// The largest λ at which the empty network is optimal; the path starts just above it
export function l1LambdaMax(data) {
  const F = new Fit(data);
  let m = 0;
  for (let p = 0; p < F.P; p++) m = Math.max(m, Math.abs(F.grad(p)));
  return m;
}

export function l1Grid(lamMax, n = 28, span = 400) {
  return Array.from({ length: n }, (_, k) => lamMax * 1.05 * span ** (-k / (n - 1)));
}

// Contiguous blocks of transitions for K-fold cross-validation
export function foldSplit(M, K, k) {
  const lo = Math.floor((k * M) / K), hi = Math.floor(((k + 1) * M) / K), tr = [], te = [];
  for (let t = 0; t < M; t++) (t >= lo && t < hi ? te : tr).push(t);
  return { train: Int32Array.from(tr), test: Int32Array.from(te) };
}

// fold = -1: fit the full data and return the weights at every λ.
// fold = k: fit without block k and return the held-out log-likelihood at every λ.
export function l1PathTask(data, lams, fold, K = 5) {
  if (fold < 0) {
    const F = new Fit(data);
    return { Ws: lams.map((lam) => { l1Fit(F, lam); return Float32Array.from(F.W); }) };
  }
  const { train, test } = foldSplit(data.M, K, fold);
  const F = new Fit(data, train), Ft = new Fit(data, test);
  return { held: lams.map((lam) => { l1Fit(F, lam); Ft.setW(F.W); return Ft.logLik(); }) };
}

// ---------------------------------------------------------------------------
// The MDL prior, Eq. (27), split into the parts of the hierarchy it comes from
// ---------------------------------------------------------------------------

export const LAMBDA = 1, DELTA = 1e-8;
const LOGQ = Math.log(Math.expm1(LAMBDA * DELTA)); // log(e^{λΔ} − 1)

let LF = new Float64Array(1);
function lf(n) {
  if (n >= LF.length) { const a = new Float64Array(2 * n + 2); a.set(LF); for (let k = LF.length; k < a.length; k++) a[k] = a[k - 1] + Math.log(k); LF = a; }
  return LF[n];
}
const logC = (n, k) => (k < 0 || k > n ? -Infinity : lf(n) - lf(k) - lf(n - k));

// E edges, m = counts per category, z = category values, P = number of pairs
export function priorParts(P, E, m, z) {
  const K = m.length;
  let zc = 0;
  for (const v of z) zc += LAMBDA * Math.abs(v) - LOGQ + LN2;
  let sm = 0;
  for (const c of m) sm += lf(c);
  return {
    edges: Math.log(P + 1) + logC(P, E),                          // −log P(E) P(A | E), Eq. (15)
    K: E > 0 ? Math.log(E) : 0,                                   // −log P(K | A)
    assign: E > 0 ? lf(E) - sm + logC(E - 1, K - 1) : 0,          // −log P(W | z, A), Eq. (24)
    values: zc,                                                    // −log P(z | λ, Δ), Eq. (25)
  };
}
export const sumParts = (q) => q.edges + q.K + q.assign + q.values;
const priorTotal = (P, E, m, z) => sumParts(priorParts(P, E, m, z));

// Golden-section search for the maximum of a unimodal f on [lo, hi]
function golden(f, lo, hi, iters = 16) {
  const r = (Math.sqrt(5) - 1) / 2;
  let a = lo, b = hi, c = b - r * (b - a), d = a + r * (b - a), fc = f(c), fd = f(d);
  for (let k = 0; k < iters; k++) {
    if (fc > fd) { b = d; d = c; fd = fc; c = b - r * (b - a); fc = f(c); }
    else { a = c; c = d; fc = fd; d = a + r * (b - a); fd = f(d); }
  }
  return fc > fd ? [c, fc] : [d, fd];
}

// ---------------------------------------------------------------------------
// MDL fit, §V: greedy moves on W and on the categories z, accepted only if Σ goes down
// ---------------------------------------------------------------------------

export class MDL {
  constructor(F, seed = 1) {
    this.F = F; this.rng = makeRng(seed);
    this.cat = new Int32Array(F.P).fill(-1);
    this.z = []; this.m = []; this.E = 0;
    this.LL = F.logLik();
  }

  prior() { return priorTotal(this.F.P, this.E, this.m, this.z); }
  sigma() { return -this.LL + this.prior(); }
  members(k) { const out = []; for (let p = 0; p < this.F.P; p++) if (this.cat[p] === k) out.push(p); return out; }

  // drop category k if it is empty (swap the last one into its slot)
  prune(k) {
    if (this.m[k] > 0) return;
    const last = this.z.length - 1;
    if (k !== last) {
      for (let p = 0; p < this.F.P; p++) if (this.cat[p] === last) this.cat[p] = k;
      this.z[k] = this.z[last]; this.m[k] = this.m[last];
    }
    this.z.pop(); this.m.pop();
  }

  // prior if pair p (now in category c, −1 = zero) moves to category k (−1 = zero, K = a new one with value v)
  priorMove(c, k, v) {
    const m = this.m.slice(), z = this.z.slice();
    let E = this.E;
    if (c >= 0) m[c]--; else E++;
    if (k === -1) E--;
    else if (k === z.length) { z.push(v); m.push(1); }
    else m[k]++;
    if (c >= 0 && m[c] === 0) { m.splice(c, 1); z.splice(c, 1); }
    return priorTotal(this.F.P, E, m, z);
  }

  commitMove(p, k, v, dLL) {
    const F = this.F, c = this.cat[p], w = F.W[p];
    const nv = k === -1 ? 0 : k === this.z.length ? v : this.z[k];
    if (k === this.z.length) { this.z.push(v); this.m.push(0); }
    F.apply(p, nv - w);
    if (Math.abs(nv - w) > 0 && F.W[p] !== nv) F.W[p] = nv; // keep exact category values
    if (c >= 0) this.m[c]--; else this.E++;
    if (k >= 0) { this.m[k]++; this.cat[p] = k; } else { this.cat[p] = -1; this.E--; }
    this.LL += dLL;
    if (c >= 0) this.prune(c);
  }

  // One visit to pair p: options (1) existing category, (2) new category, (3) zero, chosen at random
  edgeUpdate(p) {
    const F = this.F, c = this.cat[p], w = F.W[p], K = this.z.length;
    const opts = [];
    if (K > (c >= 0 ? 1 : 0)) opts.push(1);
    opts.push(2);
    if (c >= 0) opts.push(3);
    const opt = opts[Math.floor(this.rng.uniform() * opts.length)];
    const [g, cv] = F.gradCurv(p), P0 = this.prior();
    let k, v = 0;
    if (opt === 1) {
      // best category by the quadratic model of the likelihood, then checked exactly
      let best = Infinity;
      for (let q = 0; q < K; q++) {
        if (q === c) continue;
        const d = this.z[q] - w, pred = -(g * d - 0.5 * cv * d * d) + this.priorMove(c, q) - P0;
        if (pred < best) { best = pred; k = q; }
      }
      if (best > 3) return false; // clearly worse; skip the exact check
    } else if (opt === 2) {
      v = w + g / cv;
      if (Math.abs(v) < 1e-6) return false;
      k = K;
      const d = v - w;
      if (-(g * d - 0.5 * cv * d * d) + this.priorMove(c, k, v) - P0 > 3) return false;
    } else k = -1;
    const nv = k === -1 ? 0 : k === K ? v : this.z[k];
    const dLL = F.deltaPair(p, nv - w), dS = -dLL + this.priorMove(c, k, v) - P0;
    if (dS < -1e-9) { this.commitMove(p, k, v, dLL); return true; }
    return false;
  }

  nonzero() { const out = []; for (let p = 0; p < this.F.P; p++) if (this.cat[p] >= 0) out.push(p); return out; }

  // Edge moves: slide an edge to a new neighbour of one endpoint. The prior is unchanged.
  edgeMoves() {
    const F = this.F, N = F.N, rng = this.rng;
    let acc = 0;
    for (let i = 0; i < N; i++) {
      const inc = [], free = [];
      for (let u = 0; u < N; u++) if (u !== i) (this.cat[F.idx[i * N + u]] >= 0 ? inc : free).push(F.idx[i * N + u]);
      if (!inc.length || !free.length) continue;
      const pe = inc[Math.floor(rng.uniform() * inc.length)], pf = free[Math.floor(rng.uniform() * free.length)];
      const w = F.W[pe], ch = [[pe, -w], [pf, w]], dLL = F.deltaMulti(ch);
      if (dLL > 1e-9) {
        F.applyMulti(ch); F.W[pf] = w; F.W[pe] = 0;
        this.cat[pf] = this.cat[pe]; this.cat[pe] = -1; this.LL += dLL; acc++;
      }
    }
    return acc;
  }

  // Edge swaps: (i,j),(u,v) -> exchange the values of (i,v)<->(i,j) and (u,j)<->(u,v). Prior unchanged.
  edgeSwaps() {
    const F = this.F, N = F.N, nz = this.nonzero(), rng = this.rng;
    let acc = 0;
    if (nz.length < 2) return 0;
    for (let a = 0; a < N; a++) {
      const p = nz[Math.floor(rng.uniform() * nz.length)], q = nz[Math.floor(rng.uniform() * nz.length)];
      let [i, j] = [F.I[p], F.J[p]], [u, v] = [F.I[q], F.J[q]];
      if (rng.uniform() < 0.5) [i, j] = [j, i];
      if (new Set([i, j, u, v]).size < 4) continue;
      const iv = F.idx[i * N + v], ij = p, uj = F.idx[u * N + j], uv = q;
      const W = F.W, ch = [[iv, W[ij] - W[iv]], [ij, W[iv] - W[ij]], [uj, W[uv] - W[uj]], [uv, W[uj] - W[uv]]];
      const dLL = F.deltaMulti(ch);
      if (dLL > 1e-9) {
        const vals = [W[ij], W[iv], W[uv], W[uj]], cats = [this.cat[ij], this.cat[iv], this.cat[uv], this.cat[uj]];
        F.applyMulti(ch);
        W[iv] = vals[0]; W[ij] = vals[1]; W[uj] = vals[2]; W[uv] = vals[3];
        this.cat[iv] = cats[0]; this.cat[ij] = cats[1]; this.cat[uj] = cats[2]; this.cat[uv] = cats[3];
        this.LL += dLL; acc++;
      }
    }
    return acc;
  }

  // log-likelihood change if the members of each listed category take new values
  dLLvalues(groups) {
    const ch = [];
    for (const [mem, from, to] of groups) for (const p of mem) ch.push([p, to - from]);
    return [this.F.deltaMulti(ch), ch];
  }

  // Best common value for a set of entries currently at `from`, others fixed
  bestValue(mem, from, lo, hi, extra = []) {
    return golden((v) => this.dLLvalues([[mem, from, v], ...extra])[0] - LAMBDA * Math.abs(v), lo, hi);
  }

  // Weight category values: z_k <- z_k' by a 1-D search
  valueUpdates() {
    let acc = 0;
    for (let k = 0; k < this.z.length; k++) {
      const mem = this.members(k), z = this.z[k], span = 0.5 * Math.max(0.05, Math.abs(z));
      const [v] = this.bestValue(mem, z, z - span, z + span);
      const [dLL, ch] = this.dLLvalues([[mem, z, v]]);
      const dS = -dLL + LAMBDA * (Math.abs(v) - Math.abs(z));
      if (dS < -1e-9) { this.F.applyMulti(ch); for (const p of mem) this.F.W[p] = v; this.z[k] = v; this.LL += dLL; acc++; }
    }
    return acc;
  }

  // Merge two neighbouring categories into one
  merge() {
    const K = this.z.length;
    if (K < 2) return false;
    const ord = this.z.map((v, k) => k).sort((a, b) => this.z[a] - this.z[b]);
    const r = Math.floor(this.rng.uniform() * (K - 1)), a = ord[r], b = ord[r + 1];
    const ma = this.members(a), mb = this.members(b), za = this.z[a], zb = this.z[b];
    const [v] = golden((x) => this.dLLvalues([[ma, za, x], [mb, zb, x]])[0], za, zb, 14);
    const [dLL, ch] = this.dLLvalues([[ma, za, v], [mb, zb, v]]);
    const m = this.m.slice(), z = this.z.slice();
    m[a] += m[b]; z[a] = v; m.splice(b, 1); z.splice(b, 1);
    const dS = -dLL + priorTotal(this.F.P, this.E, m, z) - this.prior();
    if (dS >= -1e-9) return false;
    this.F.applyMulti(ch);
    for (const p of ma.concat(mb)) { this.F.W[p] = v; this.cat[p] = a; }
    this.z[a] = v; this.m[a] += this.m[b]; this.m[b] = 0; this.LL += dLL;
    this.prune(b);
    return true;
  }

  // Split one category in two: order its entries by the value each would prefer on its own,
  // cut where the two halves are most different, then search each half's value
  split() {
    const cand = this.m.map((c, k) => k).filter((k) => this.m[k] >= 2);
    if (!cand.length) return false;
    const k = cand[Math.floor(this.rng.uniform() * cand.length)], z = this.z[k], F = this.F;
    const pref = this.members(k).map((p) => { const [g, c] = F.gradCurv(p); return [p, z + g / c]; }).sort((a, b) => a[1] - b[1]);
    const n = pref.length, cum = [0];
    for (const [, v] of pref) cum.push(cum[cum.length - 1] + v);
    let best = -1, cut = 1;
    for (let s = 1; s < n; s++) {
      const ma = cum[s] / s, mb = (cum[n] - cum[s]) / (n - s), score = s * (n - s) * (mb - ma) ** 2;
      if (score > best) { best = score; cut = s; }
    }
    const A = pref.slice(0, cut).map((x) => x[0]), B = pref.slice(cut).map((x) => x[0]);
    const lo = pref[0][1], hi = pref[n - 1][1], pad = 0.1 * (hi - lo) + 1e-3;
    const [va] = this.bestValue(A, z, lo - pad, hi + pad);
    const [vb] = this.bestValue(B, z, lo - pad, hi + pad, [[A, z, va]]);
    if (Math.abs(va - vb) < 1e-9) return false;
    const [dLL, ch] = this.dLLvalues([[A, z, va], [B, z, vb]]);
    const m = this.m.slice(), zz = this.z.slice();
    m[k] = A.length; zz[k] = va; m.push(B.length); zz.push(vb);
    const dS = -dLL + priorTotal(F.P, this.E, m, zz) - this.prior();
    if (dS >= -1e-9) return false;
    F.applyMulti(ch);
    for (const p of A) F.W[p] = va;
    for (const p of B) { F.W[p] = vb; this.cat[p] = this.z.length; }
    this.z[k] = va; this.m[k] = A.length; this.z.push(vb); this.m.push(B.length); this.LL += dLL;
    return true;
  }

  snapshot(sweep, phase, moves) {
    return {
      sweep, phase, moves, LL: this.LL, parts: priorParts(this.F.P, this.E, this.m, this.z),
      W: Float32Array.from(this.F.W), z: this.z.slice(), m: this.m.slice(), E: this.E, cat: Int16Array.from(this.cat),
    };
  }

  // Run until a few sweeps in a row no longer lower Σ. record: keep snapshots for the animation.
  run({ maxSweeps = 80, patience = 3, record = false, chunk = 70 } = {}) {
    const F = this.F, snaps = [], order = Int32Array.from({ length: F.P }, (_, p) => p);
    if (record) snaps.push(this.snapshot(0, "start", 0));
    let calm = 0, sweep = 0;
    while (sweep < maxSweeps && calm < patience) {
      sweep++;
      const s0 = this.sigma();
      for (let p = F.P - 1; p > 0; p--) { const q = Math.floor(this.rng.uniform() * (p + 1)); [order[p], order[q]] = [order[q], order[p]]; }
      let acc = 0;
      for (let r = 0; r < F.P; r++) {
        if (this.edgeUpdate(order[r])) acc++;
        if (record && sweep <= 2 && (r + 1) % chunk === 0 && acc) { snaps.push(this.snapshot(sweep, "edge updates", acc)); acc = 0; }
      }
      if (record && acc) snaps.push(this.snapshot(sweep, "edge updates", acc));
      const steps = [["edge moves", () => this.edgeMoves()], ["edge swaps", () => this.edgeSwaps()],
        ["category values", () => this.valueUpdates()], ["merge", () => +this.merge()], ["split", () => +this.split()]];
      for (const [name, fn] of steps) { const a = fn(); if (record && a) snaps.push(this.snapshot(sweep, name, a)); }
      calm = s0 - this.sigma() < 1e-6 ? calm + 1 : 0;
    }
    return { sweeps: sweep, snaps };
  }

  // Σ with exactly K categories over the current set of edges (for "what the categories buy")
  forceK(K) {
    const F = this.F, nz = this.nonzero();
    if (nz.length < K) return null;
    const W0 = Float64Array.from(F.W), LL0 = this.LL;
    // each edge's preferred value with the rest held fixed, then 1-D k-means
    const pref = nz.map((p) => { const [g, c] = F.gradCurv(p); return F.W[p] + g / c; });
    const sorted = pref.slice().sort((a, b) => a - b);
    let cen = Array.from({ length: K }, (_, k) => sorted[Math.floor(((k + 0.5) * sorted.length) / K)]);
    let lab = new Int32Array(nz.length);
    for (let it = 0; it < 30; it++) {
      pref.forEach((v, e) => { let b = 0; for (let k = 1; k < K; k++) if (Math.abs(v - cen[k]) < Math.abs(v - cen[b])) b = k; lab[e] = b; });
      cen = cen.map((c, k) => { const s = pref.filter((_, e) => lab[e] === k); return s.length ? s.reduce((a, b) => a + b, 0) / s.length : c; });
    }
    // set each category to its best value given the others, twice round
    const groups = () => cen.map((_, k) => nz.filter((_, e) => lab[e] === k)).filter((g) => g.length);
    let G = groups(), vals = G.map((g) => F.W[g[0]]);
    const setAll = () => {
      const ch = [];
      G.forEach((g, k) => g.forEach((p) => ch.push([p, vals[k] - F.W[p]])));
      const d = F.deltaMulti(ch); F.applyMulti(ch);
      G.forEach((g, k) => g.forEach((p) => { F.W[p] = vals[k]; }));
      return d;
    };
    let LL = LL0 + setAll();
    for (let round = 0; round < 2; round++)
      G.forEach((g, k) => {
        const z = vals[k], span = 0.6 * Math.max(0.05, Math.abs(z));
        const [v, d] = golden((x) => F.deltaMulti(g.map((p) => [p, x - z])) - LAMBDA * Math.abs(x), z - span, z + span);
        const dLL = F.deltaMulti(g.map((p) => [p, v - z]));
        F.applyMulti(g.map((p) => [p, v - z])); g.forEach((p) => { F.W[p] = v; });
        vals[k] = v; LL += dLL;
      });
    const parts = priorParts(F.P, nz.length, G.map((g) => g.length), vals);
    // restore
    F.setW(W0); this.LL = LL0;
    return { K: G.length, LL, parts, z: vals, m: G.map((g) => g.length) };
  }
}

// ---------------------------------------------------------------------------
// Normal density prior on the nonzero weights, Eq. (30), with the same P(A | E) P(E)
// ---------------------------------------------------------------------------

export function normalPriorFit(F, mu, sd, { maxSweeps = 60, seed = 3 } = {}) {
  const rng = makeRng(seed), P = F.P, W = F.W, s2 = sd * sd;
  const logN = (w) => -((w - mu) ** 2) / (2 * s2) - Math.log(Math.sqrt(2 * Math.PI) * sd);
  const logA = (E) => -(Math.log(P + 1) + logC(P, E));
  let E = countEdges(W);
  for (let s = 0; s < maxSweeps; s++) {
    let gain = 0;
    for (let r = 0; r < P; r++) {
      const p = Math.floor(rng.uniform() * P), w = W[p], [g, c] = F.gradCurv(p);
      // maximize log-lik (quadratic model) + log N(w): a precision-weighted average
      const nw = (c * w + g + mu / s2) / (c + 1 / s2);
      const opts = [[nw, w === 0 ? 1 : 0]];
      if (w !== 0) opts.push([0, -1]);
      for (const [v, dE] of opts) {
        const dLL = F.deltaPair(p, v - w);
        const dObj = dLL + (v !== 0 ? logN(v) : 0) - (w !== 0 ? logN(w) : 0) + logA(E + dE) - logA(E);
        if (dObj > 1e-9) { F.apply(p, v - w); E += dE; gain += dObj; break; }
      }
    }
    if (gain < 1e-6) break;
  }
  return F.W;
}

// ---------------------------------------------------------------------------
// One point of the Fig. 2 comparison: MDL, the normal prior and L1 with 5-fold CV at one M
// ---------------------------------------------------------------------------

export function fitMDL(data, seed = 1) {
  const F = new Fit(data), mdl = new MDL(F, seed);
  mdl.run();
  return { W: Float64Array.from(F.W), mdl };
}

export function fitL1CV(data, folds = 5) {
  const lams = l1Grid(l1LambdaMax(data), 22, 300);
  const held = new Float64Array(lams.length);
  for (let k = 0; k < folds; k++) l1PathTask(data, lams, k, folds).held.forEach((v, i) => { held[i] += v; });
  let b = 0;
  for (let i = 1; i < lams.length; i++) if (held[i] > held[b]) b = i;
  const F = new Fit(data);
  for (let i = 0; i <= b; i++) l1Fit(F, lams[i]); // warm-started down the path
  return { W: Float64Array.from(F.W), lam: lams[b] };
}

export function comparePoint(net, data) {
  const out = {};
  const score = (W) => ({ ...similarity(net.W, W), E: countEdges(W) });
  out.mdl = score(fitMDL(data).W);
  const nz = Array.from(net.W).filter((w) => w !== 0), mu = nz.reduce((a, b) => a + b, 0) / nz.length;
  const sd = Math.sqrt(nz.reduce((a, b) => a + (b - mu) ** 2, 0) / nz.length);
  out.normal = score(normalPriorFit(new Fit(data), mu, sd));
  out.l1 = score(fitL1CV(data).W);
  return out;
}
