import * as I from "./ising.js";
import { l1Grid, l1LambdaMax, priorParts, sumParts } from "./fits.js";
import { C, el, frame, line, dot, label, refLine, legend, tiles, showTip, hideTip, fmt } from "../../../../week-3/code/viz-common/charts.js";
import { clipped } from "../../../../week-4/code/viz-common/charts4.js";
import { initReplicas } from "./replicas.js";
import { initMoves, movesReset, movesRedraw } from "./moves.js";

const $ = (id) => document.getElementById(id);
const clear = (id) => { const e = $(id); e.replaceChildren(); return e; };
const LN2 = Math.log(2), bits = (nats) => nats / LN2;
const Ms = [100, 200, 400, 600, 1000, 1600, 2400, 3200, 4000];
const MDLc = C.model, L1c = C.g2, NORMc = C.edge;

const S = {
  mode: "equal", M: 1000, seed: 1, gen: 0,
  net: null, data: null, lams: null, l1: null, li: 0, liCV: 0, liE: 0,
  fit: null, frame: 0, playing: null, densKind: "density", compare: null,
};

// ---------------------------------------------------------------------------
// Worker pool: the six L1 paths and the MDL fit run in parallel. A new data set
// kills whatever is still running, so dragging a slider never queues up stale work.
// ---------------------------------------------------------------------------

const POOL = Math.max(2, Math.min(7, (navigator.hardwareConcurrency || 4) - 1));
const workerUrl = new URL("./worker.js", import.meta.url);
const slots = Array.from({ length: POOL }, () => ({ w: new Worker(workerUrl, { type: "module" }), job: null }));
let queue = [], jobId = 0;

function pump() {
  for (const s of slots) {
    if (s.job || !queue.length) continue;
    const job = queue.shift();
    s.job = job;
    s.w.onmessage = ({ data }) => { s.job = null; job.resolve(data.result); pump(); };
    s.w.onerror = (e) => { setStatus(`a fit failed: ${e.message}`, true); console.error("worker:", e.message, `${e.filename}:${e.lineno}`); };
    s.w.postMessage({ ...job.msg, id: job.id });
  }
}
const runJob = (msg) => new Promise((resolve) => { queue.push({ id: ++jobId, msg, resolve }); pump(); });
function resetPool() {
  queue = [];
  for (const s of slots) if (s.job) { s.w.terminate(); s.w = new Worker(workerUrl, { type: "module" }); s.job = null; }
}

// ---------------------------------------------------------------------------
// Layout and network drawing
// ---------------------------------------------------------------------------

// Fruchterman–Reingold on the karate club, seeded, once
const POS = (() => {
  const N = 34, rng = I.makeRng(5), p = Array.from({ length: N }, () => [rng.uniform(), rng.uniform()]), k = Math.sqrt(1 / N);
  for (let it = 0; it < 500; it++) {
    const temp = 0.1 * (1 - it / 500) + 0.002, d = p.map(() => [0, 0]);
    for (let a = 0; a < N; a++) for (let b = a + 1; b < N; b++) {
      const dx = p[a][0] - p[b][0], dy = p[a][1] - p[b][1], r = Math.hypot(dx, dy) + 1e-6, f = (k * k) / r;
      d[a][0] += (dx / r) * f; d[a][1] += (dy / r) * f; d[b][0] -= (dx / r) * f; d[b][1] -= (dy / r) * f;
    }
    for (const [a, b] of I.KARATE) {
      const dx = p[a][0] - p[b][0], dy = p[a][1] - p[b][1], r = Math.hypot(dx, dy) + 1e-6, f = (r * r) / k;
      d[a][0] -= (dx / r) * f; d[a][1] -= (dy / r) * f; d[b][0] += (dx / r) * f; d[b][1] += (dy / r) * f;
    }
    p.forEach((q, a) => { const r = Math.hypot(...d[a]) + 1e-9, s = Math.min(r, temp); q[0] += (d[a][0] / r) * s; q[1] += (d[a][1] / r) * s; });
  }
  const xs = p.map((q) => q[0]), ys = p.map((q) => q[1]);
  const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
  return p.map(([x, y]) => [(x - x0) / (x1 - x0), (y - y0) / (y1 - y0)]);
})();
const CIRCLE = Array.from({ length: 34 }, (_, i) => { const a = -Math.PI / 2 + (2 * Math.PI * i) / 34; return [0.5 + 0.5 * Math.cos(a), 0.5 + 0.5 * Math.sin(a)]; });

const PP = I.pairs(34);

// W: weights by pair. ref: true weights (colour by true/spurious, and draw missed edges). wscale: px per unit weight.
// marks: extra pairs drawn on top, [{ p, color, width, opacity }]
function drawNet(box, W, { ref = null, pos = POS, height = 300, wscale = 9, maxw = 4, alpha = 0.85, marks = [] } = {}) {
  const Wd = Math.max(280, box.clientWidth), H = height, pad = 14;
  const svg = el("svg", { viewBox: `0 0 ${Wd} ${H}`, width: "100%", class: "chart" }, box);
  const X = pos.map(([x, y]) => [pad + x * (Wd - 2 * pad), pad + y * (H - 2 * pad)]);
  const g = el("g", {}, svg), top = el("g", {}, svg);
  for (let p = 0; p < PP.P; p++) {
    const w = W[p], t = ref ? ref[p] : w;
    if (w === 0 && t === 0) continue;
    const a = X[PP.I[p]], b = X[PP.J[p]];
    const attrs = { x1: a[0], y1: a[1], x2: b[0], y2: b[1] };
    if (w === 0) el("line", { ...attrs, stroke: C.muted, "stroke-width": 0.8, opacity: 0.35 }, g);
    else el("line", { ...attrs, stroke: !ref || t !== 0 ? C.g1 : C.g2, "stroke-width": Math.min(maxw, 0.5 + wscale * Math.abs(w)), opacity: alpha }, !ref || t !== 0 ? g : top);
  }
  for (const mk of marks) {
    const a = X[PP.I[mk.p]], b = X[PP.J[mk.p]];
    el("line", { x1: a[0], y1: a[1], x2: b[0], y2: b[1], stroke: mk.color, "stroke-width": mk.width ?? 3, opacity: mk.opacity ?? 1, "stroke-linecap": "round" }, svg);
  }
  X.forEach(([x, y]) => el("circle", { cx: x, cy: y, r: 3.6, fill: C.ink2, stroke: C.surface, "stroke-width": 1.2 }, svg));
  return svg;
}

// Log-x frame helper: data in log10 units, ticks at powers of ten
const kfmt = (v) => (Math.abs(v) >= 10000 ? `${fmt(v / 1000, 1)}k` : fmt(v, 0));
const pow10 = (v) => { const k = Math.round(v); return Math.abs(v - k) < 1e-6 ? (k >= 0 && k <= 3 ? String(10 ** k) : `10${"⁻".repeat(k < 0)}${String(Math.abs(k)).split("").map((c) => "⁰¹²³⁴⁵⁶⁷⁸⁹"[c]).join("")}`) : ""; };
function logTicks(lo, hi) { const t = []; for (let k = Math.ceil(lo); k <= Math.floor(hi); k++) t.push(k); return t; }

// ---------------------------------------------------------------------------
// Data and recomputation
// ---------------------------------------------------------------------------

function setStatus(text, warn = false) { const s = $("status"); s.textContent = text; s.classList.toggle("warn", warn); }
const busy = (on) => ["l1", "fit", "cats"].forEach((id) => $(id).classList.toggle("busy", on));

async function recompute() {
  const gen = ++S.gen;
  stopPlay();
  resetPool();
  S.net = I.trueNetwork(S.mode, 1);
  S.data = I.simulate(S.net, S.M, 7 + 100 * S.seed);
  S.Ftrue = null;
  renderData();
  renderDensity();
  renderCompare();
  busy(true);
  setStatus("fitting…");
  const t0 = performance.now();
  S.lams = l1Grid(l1LambdaMax(S.data));
  const data = S.data, lams = S.lams;
  S.l1 = null; S.fit = null;
  // MDL first in the queue; each result is drawn as soon as it arrives
  const mdlDone = runJob({ task: "mdl", data, seed: S.seed }).then((mdl) => {
    if (gen !== S.gen) return;
    S.fit = mdl; S.frame = 0;
    ["fit", "cats"].forEach((id) => $(id).classList.remove("busy"));
    renderFit();
    renderCats();
    movesReset();
    if (fitSeen) play();
  });
  const l1Done = Promise.all([-1, 0, 1, 2, 3, 4].map((fold) => runJob({ task: "l1", data, lams, fold }))).then((l1) => {
    if (gen !== S.gen) return;
    const held = lams.map((_, i) => l1.slice(1).reduce((s, r) => s + r.held[i], 0));
    S.l1 = { Ws: l1[0].Ws, held, stats: l1[0].Ws.map((W) => ({ ...I.similarity(S.net.W, W), E: I.countEdges(W) })) };
    S.liCV = held.indexOf(Math.max(...held));
    S.liE = S.l1.stats.reduce((b, s, i) => (Math.abs(s.E - 78) < Math.abs(S.l1.stats[b].E - 78) || (Math.abs(s.E - 78) === Math.abs(S.l1.stats[b].E - 78) && s.sA > S.l1.stats[b].sA) ? i : b), 0);
    S.li = S.liCV;
    $("l1").classList.remove("busy");
    renderL1();
  });
  await Promise.all([mdlDone, l1Done]);
  if (gen !== S.gen) return;
  setStatus(`M = ${S.M} · 7 fits in parallel on ${POOL} workers · ${Math.round(performance.now() - t0)} ms`);
}

// ---------------------------------------------------------------------------
// 0. Data
// ---------------------------------------------------------------------------

function renderData() {
  const net = S.net, w = Array.from(net.W).filter((v) => v);
  tiles(clear("t-data"), [
    { label: "Nodes N", value: 34 },
    { label: "Possible edges", value: PP.P, note: "N(N−1)/2" },
    { label: "True edges E", value: 78 },
    { label: "Transitions M", value: S.M.toLocaleString() },
    { label: "Weights", value: `${fmt(Math.min(...w))} – ${fmt(Math.max(...w))}`, note: `mean ${fmt(w.reduce((a, b) => a + b, 0) / w.length, 3)}` },
  ]);
  drawNet(clear("p-true"), net.W, { height: 300 });

  const box = clear("p-raster"), wrap = document.createElement("div"); wrap.className = "raster"; box.appendChild(wrap);
  const T = Math.min(240, S.M + 1), N = 34, cv = document.createElement("canvas");
  cv.width = T; cv.height = N; cv.style.height = "300px";
  const ctx = cv.getContext("2d"), img = ctx.createImageData(T, N), X = S.data.X, TT = S.M + 1;
  for (let i = 0; i < N; i++) for (let t = 0; t < T; t++) {
    const v = X[i * TT + t] > 0 ? [214, 212, 203] : [38, 38, 36], k = 4 * (i * T + t);
    img.data[k] = v[0]; img.data[k + 1] = v[1]; img.data[k + 2] = v[2]; img.data[k + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  wrap.appendChild(cv);
  cv.addEventListener("pointermove", (e) => {
    const b = cv.getBoundingClientRect(), t = Math.floor(((e.clientX - b.left) / b.width) * T), i = Math.floor(((e.clientY - b.top) / b.height) * N);
    if (t < 0 || t >= T || i < 0 || i >= N) return hideTip();
    showTip(e, [{ value: X[i * TT + t] > 0 ? "+1" : "−1", label: "spin" }], `node ${i + 1} · t = ${t}`);
  });
  cv.addEventListener("pointerleave", hideTip);
}

// ---------------------------------------------------------------------------
// 1. L1
// ---------------------------------------------------------------------------

let l1Marks = [];

function lamFrame(box, yOpt, height = 200) {
  const L = S.lams.map(Math.log10), lo = Math.min(...L), hi = Math.max(...L);
  return frame(box, { height, x: { domain: [lo, hi], label: "λ", ticks: logTicks(lo, hi) }, xFormat: pow10, ...yOpt });
}

function addMarks(f) {
  const lx = Math.log10(S.lams[S.liCV]);
  el("line", { x1: f.xs(lx), x2: f.xs(lx), y1: f.margin.t, y2: f.margin.t + f.ih, stroke: C.axis, "stroke-width": 1 }, f.plot);
  const t = el("text", { x: f.xs(lx) + 4, y: f.margin.t + 10, class: "tick" }, f.plot); t.textContent = "CV";
  const m = el("line", { y1: f.margin.t, y2: f.margin.t + f.ih, stroke: C.model, "stroke-width": 1.5 }, f.plot);
  l1Marks.push({ f, m });
  // click or drag anywhere on the chart to choose λ
  const hit = el("rect", { x: f.margin.l, y: f.margin.t, width: f.iw, height: f.ih, fill: "transparent", style: "cursor: ew-resize" }, f.svg);
  const pick = (e) => {
    const b = f.svg.getBoundingClientRect(), v = f.xs.inv(((e.clientX - b.left) / b.width) * f.width);
    let k = 0; S.lams.forEach((l, i) => { if (Math.abs(Math.log10(l) - v) < Math.abs(Math.log10(S.lams[k]) - v)) k = i; });
    setLam(k);
  };
  hit.addEventListener("pointerdown", (e) => { pick(e); hit.setPointerCapture(e.pointerId); });
  hit.addEventListener("pointermove", (e) => { if (e.buttons) pick(e); });
}

function renderL1() {
  if (!S.l1) return;
  l1Marks = [];
  const { Ws, held, stats } = S.l1, L = S.lams.map(Math.log10), net = S.net;
  $("c-lam").max = S.lams.length - 1;

  // weight paths
  const box = clear("p-paths");
  let ymin = 0, ymax = 0;
  for (const W of Ws) for (const v of W) { ymin = Math.min(ymin, v); ymax = Math.max(ymax, v); }
  for (const v of net.W) ymax = Math.max(ymax, v);
  const f = lamFrame(box, { y: { domain: [ymin - 0.01, ymax + 0.02], label: "inferred weight Ŵ_ij" }, yFormat: (v) => fmt(v, 2) }, 300);
  const g = clipped(f), red = el("g", {}, g), blue = el("g", {}, g);
  for (let p = 0; p < PP.P; p++) {
    if (!Ws.some((W) => W[p] !== 0)) continue;
    const pts = Ws.map((W, i) => [L[i], W[p]]), tr = net.W[p] !== 0;
    const ln = line(f, pts, tr ? C.g1 : C.g2, { "stroke-width": 0.8, opacity: tr ? 0.75 : 0.55 });
    (tr ? blue : red).appendChild(ln);
  }
  for (const v of net.W) if (v) el("line", { x1: f.margin.l + f.iw + 2, x2: f.margin.l + f.iw + 9, y1: f.ys(v), y2: f.ys(v), stroke: C.ink2, "stroke-width": 1, opacity: 0.5 }, f.svg);
  addMarks(f);

  // held-out log-likelihood
  const lo = Math.min(...held), hi = Math.max(...held), cut = hi - 3 * (hi - held[held.length - 1] + 5);
  const fh = lamFrame(clear("p-held"), { y: { domain: [Math.max(lo, cut), hi + 0.06 * (hi - Math.max(lo, cut))], label: "held-out log P" }, yFormat: kfmt });
  clipped(fh).appendChild(line(fh, L.map((x, i) => [x, held[i]]), C.neutral));
  dot(fh, L[S.liCV], held[S.liCV], C.neutral);
  addMarks(fh);

  // E
  const fe = lamFrame(clear("p-E"), { y: { domain: [0, Math.log10(PP.P)], label: "E", ticks: [0, 1, 2, Math.log10(PP.P)] }, yFormat: (v) => (v === 0 ? "1" : String(Math.round(10 ** v))) });
  refLine(fe, [[L[0], Math.log10(78)], [L[L.length - 1], Math.log10(78)]]);
  line(fe, L.map((x, i) => [x, Math.log10(Math.max(1, stats[i].E))]), C.neutral);
  addMarks(fe);

  // similarity
  const fs = lamFrame(clear("p-sim"), { y: { domain: [0, 1.02], label: "similarity" }, yFormat: (v) => fmt(v, 1) });
  line(fs, L.map((x, i) => [x, stats[i].sW]), C.g1);
  line(fs, L.map((x, i) => [x, stats[i].sA]), C.model);
  const pk = (key) => stats.reduce((b, st, i) => (st[key] > stats[b][key] ? i : b), 0), pA = pk("sA"), pW = pk("sW");
  label(fs, L[pA], stats[pA].sA, "binary A", { anchor: "middle", dx: 0, dy: -8 });
  label(fs, L[pW], stats[pW].sW, "weights W", { anchor: "middle", dx: 0, dy: 16 });
  addMarks(fs);

  const cv = stats[S.liCV], ee = stats[S.liE];
  $("k-l1").innerHTML = "";
  const b = document.createElement("b");
  b.textContent = `No λ gets both right. `;
  $("k-l1").append(b, `Cross-validation picks λ̂ = ${fmt(S.lams[S.liCV], 1)}, with E = ${cv.E} edges for the true 78 (paper: 278): s(W) = ${fmt(cv.sW)}, s(A) = ${fmt(cv.sA)}. `
    + `At λ = ${fmt(S.lams[S.liE], 0)}, where E = ${ee.E}, the edges are right, s(A) = ${fmt(ee.sA)}, but their weights have shrunk to ${fmt(meanTrue(Ws[S.liE]), 3)} on average, against the true ${fmt(S.net.mu, 2)}. `
    + `And nothing in the data tells you to use that second λ: the held-out likelihood prefers the first.`);
  setLam(S.li);
}

function meanTrue(W) { let s = 0, c = 0; for (let p = 0; p < PP.P; p++) if (S.net.W[p] && W[p]) { s += W[p]; c++; } return c ? s / c : 0; }

function setLam(k) {
  if (!S.l1) return;
  S.li = k;
  $("c-lam").value = k;
  $("o-lam").textContent = fmt(S.lams[k], S.lams[k] < 10 ? 2 : 1);
  const lx = Math.log10(S.lams[k]);
  for (const { f, m } of l1Marks) { m.setAttribute("x1", f.xs(lx)); m.setAttribute("x2", f.xs(lx)); }
  const W = S.l1.Ws[k], st = S.l1.stats[k];
  drawNet(clear("p-l1net"), W, { ref: S.net.W, height: 300 });
  renderWvW();
  tiles(clear("t-l1"), [
    { label: "λ", value: fmt(S.lams[k], 1), note: k === S.liCV ? "the CV choice λ̂" : k === S.liE ? "gets E closest to 78" : "" },
    { label: "Edges E", value: st.E, note: "true 78" },
    { label: "Spurious edges", value: st.E - Math.round((st.sA * (st.E + 78)) / 2) },
    { label: "s(W, Ŵ)", value: fmt(st.sW) },
    { label: "s(A, Â)", value: fmt(st.sA) },
    { label: "Mean true-edge weight", value: fmt(meanTrue(W), 3), note: `true ${fmt(S.net.mu, 3)}` },
  ]);
}

// ---------------------------------------------------------------------------
// 2. Density vs. mass
// ---------------------------------------------------------------------------

function trueFit() {
  if (!S.Ftrue) {
    const F = new I.Fit(S.data), LL0 = F.logLik();
    F.setW(S.net.W);
    S.Ftrue = { LL0, LL1: F.logLik(), absW: S.net.W.reduce((a, b) => a + Math.abs(b), 0) };
  }
  return S.Ftrue;
}

function renderDensity() {
  const lam = +$("c-dl").value, D = 10 ** +$("c-dd").value;
  $("o-dl").textContent = lam;
  $("o-dd").textContent = D < 0.01 ? D.toExponential(0).replace("e-", "×10⁻") : fmt(D, 3);
  const dens = (w) => bits(lam * Math.abs(w) - Math.log(lam / 2));
  const mass = (w) => bits(lam * Math.abs(w) - Math.log(Math.expm1(lam * D) / 2));
  const wT = S.net ? S.net.mu : 0.22;

  tiles(clear("t-density"), [
    { label: "Density at w = 0", value: `${fmt(dens(0), 1)} bits`, note: lam > 2 ? "negative: not a code length" : "" },
    { label: `Density at w = ${fmt(wT, 2)}`, value: `${fmt(dens(wT), 1)} bits` },
    { label: `Mass at w = ${fmt(wT, 2)}`, value: `${fmt(mass(wT), 1)} bits`, note: `≈ density + log₂(1/Δ) = ${fmt(dens(wT) + Math.log2(1 / D), 1)}` },
  ]);

  const box = clear("p-cost"), ws = Array.from({ length: 201 }, (_, k) => -0.5 + k / 200);
  const ymin = Math.min(dens(0), 0) - 1, ymax = Math.max(mass(0.5), 2) + 1;
  const f = frame(box, { height: 260, x: { domain: [-0.5, 0.5], label: "weight w" }, xFormat: (v) => fmt(v, 2), y: { domain: [ymin, ymax], label: "bits" }, yFormat: (v) => fmt(v, 0) });
  el("rect", { x: f.margin.l, y: f.ys(0), width: f.iw, height: f.ys(ymin) - f.ys(0), fill: C.g2, opacity: 0.07 }, f.plot);
  refLine(f, [[-0.5, 0], [0.5, 0]], C.muted);
  line(f, ws.map((w) => [w, dens(w)]), C.g2);
  // the mass lives on the grid wΔ, w ≠ 0
  const g = clipped(f), n = Math.floor(0.5 / D);
  if (n <= 60) for (let k = -n; k <= n; k++) { if (k) g.appendChild(dot(f, k * D, mass(k * D), MDLc, 2.6)); }
  else { g.appendChild(line(f, ws.filter((w) => w < 0).map((w) => [w, mass(w)]), MDLc)); g.appendChild(line(f, ws.filter((w) => w > 0).map((w) => [w, mass(w)]), MDLc)); }
  label(f, 0.5, mass(0.5), "quantized mass", { anchor: "end", dx: -2, dy: -8, cls: "direct-label strong" });
  label(f, 0.5, dens(0.5), "density", { anchor: "end", dx: -2, dy: 14 });
  label(f, -0.5, 0, "below 0 bits", { dy: 14 });

  if (S.data) renderMapLambda(D);
}

function renderMapLambda(D) {
  const T = trueFit(), P = PP.P, E = 78, kind = S.densKind;
  const lg = Array.from({ length: 161 }, (_, k) => k / 32); // log10 λ from 0 to 5
  const sig = (l, which) => {
    const lam = 10 ** l;
    if (kind === "density") {
      const prior = which ? lam * T.absW - P * Math.log(lam / 2) : -P * Math.log(lam / 2);
      return bits(-(which ? T.LL1 : T.LL0) + prior);
    }
    const A = Math.log(P + 1) + (which ? Math.log(binom(P, E)) : 0);
    const prior = which ? A + lam * T.absW - E * Math.log(Math.expm1(lam * D) / 2) : A;
    return bits(-(which ? T.LL1 : T.LL0) + prior);
  };
  const s0 = lg.map((l) => sig(l, 0)), s1 = lg.map((l) => sig(l, 1));
  const lo = Math.min(...s0, ...s1), base = Math.min(...s1);
  const f = frame(clear("p-mapl"), {
    height: 232, x: { domain: [0, 5], label: "λ", ticks: [0, 1, 2, 3, 4, 5] }, xFormat: pow10,
    y: { domain: [lo - 800, Math.max(base + 9000, bits(-T.LL0) + 800)], label: "Σ (bits)" }, yFormat: kfmt,
  });
  const g = clipped(f);
  g.appendChild(line(f, lg.map((l, k) => [l, s0[k]]), C.muted));
  g.appendChild(line(f, lg.map((l, k) => [l, s1[k]]), C.g1));
  const k0 = s0.indexOf(Math.min(...s0)), k1 = s1.indexOf(Math.min(...s1));
  dot(f, lg[k0], s0[k0], C.muted); dot(f, lg[k1], s1[k1], C.g1);
  label(f, 0.1, s0[3], "empty network", { dy: -8 });
  label(f, lg[k1], s1[k1], "true weights", { dy: -10, anchor: "middle", dx: 0, cls: "direct-label strong" });
}

const binom = (n, k) => { let s = 1; for (let i = 1; i <= k; i++) s *= (n - k + i) / i; return s; };

// ---------------------------------------------------------------------------
// 3. Prior draws
// ---------------------------------------------------------------------------

const PLAM = 10, PDELTA = 0.02;
let priorSeed = 1;

function laplace(rng) { const u = rng.uniform() || 1e-12; return (rng.uniform() < 0.5 ? -1 : 1) * (-Math.log(u) / PLAM); }

function renderPrior() {
  const E = +$("c-pE").value, K = Math.min(+$("c-pK").value, E);
  $("o-pE").textContent = E; $("o-pK").textContent = K;
  const rng = I.makeRng(500 + priorSeed), P = PP.P;
  const pick = () => { const idx = Int32Array.from({ length: P }, (_, p) => p); for (let p = P - 1; p > 0; p--) { const q = Math.floor(rng.uniform() * (p + 1)); [idx[p], idx[q]] = [idx[q], idx[p]]; } return idx.slice(0, E); };

  // (a) Laplace density everywhere
  const W0 = new Float64Array(P).map(() => laplace(rng));
  const b0 = bits(W0.reduce((s, w) => s + PLAM * Math.abs(w) - Math.log(PLAM / 2), 0));
  // (b) uniform A, quantized Laplace on the edges (zero excluded)
  const W1 = new Float64Array(P);
  for (const p of pick()) { let v = 0; while (v === 0) v = PDELTA * Math.round(laplace(rng) / PDELTA); W1[p] = v; }
  const b1 = bits(Math.log(P + 1) + Math.log(binom(P, E))) + Array.from(W1).filter((v) => v).reduce((s, w) => s + bits(PLAM * Math.abs(w) - Math.log(Math.expm1(PLAM * PDELTA) / 2)), 0);
  // (c) the hierarchy: K values, uniform positive counts, uniform assignment
  const z = Array.from({ length: K }, () => laplace(rng));
  const cuts = new Set(); while (cuts.size < K - 1) cuts.add(1 + Math.floor(rng.uniform() * (E - 1)));
  const cs = [0, ...[...cuts].sort((a, b) => a - b), E], m = z.map((_, k) => cs[k + 1] - cs[k]);
  const W2 = new Float64Array(P), edges = pick();
  let e = 0; m.forEach((c, k) => { for (let r = 0; r < c; r++) W2[edges[e++]] = z[k]; });
  // priorParts uses the paper's λ = 1; rescaling z makes λ|z| right for λ = 10 (the Δ term is then off by log 10 per category)
  const parts = priorParts(P, E, m, z.map((v) => v * PLAM));
  const b2 = bits(sumParts(parts));

  const distinct = (W) => new Set(Array.from(W).filter((v) => v)).size;
  [[W0, b0, "pr0"], [W1, b1, "pr1"], [W2, b2, "pr2"]].forEach(([W, b, id]) => {
    const nz = I.countEdges(W);
    tiles(clear(`t-${id}`), [
      { label: "Edges", value: nz },
      { label: "Distinct values", value: distinct(W) },
      { label: "Bits for W", value: fmt(b, 0), note: `${fmt(b / Math.max(nz, 1), 1)} per edge` },
    ]).children[2].classList.toggle("bad", b < 0);
    const box = clear(`p-${id}`);
    drawNet(box, W, { pos: CIRCLE, height: 210, wscale: 6, maxw: 3, alpha: id === "pr0" ? 0.35 : 0.85 });
    weightHist(box, W, id === "pr0" ? 0 : id === "pr1" ? PDELTA : 0.004);
  });
}

function weightHist(box, W, bw) {
  const vals = Array.from(W).filter((v) => v), lim = 0.6;
  const f = frame(box, { height: 120, margin: { t: 8, r: 10, b: 30, l: 36 }, x: { domain: [-lim, lim], label: "weight", ticks: [-0.5, 0, 0.5] }, xFormat: (v) => fmt(v, 1), y: { domain: [0, 1], ticks: [] } });
  const step = bw || 0.02, bins = new Map();
  for (const v of vals) { const k = Math.round(Math.max(-lim, Math.min(lim, v)) / step); bins.set(k, (bins.get(k) ?? 0) + 1); }
  const top = Math.max(...bins.values());
  for (const [k, c] of bins) {
    const x = f.xs(k * step), w = Math.max(1.5, f.xs(step) - f.xs(0) - 1);
    el("rect", { x: x - w / 2, y: f.ys(c / top), width: w, height: f.ys(0) - f.ys(c / top), fill: bw === 0 ? C.neutral : MDLc, opacity: 0.85 }, f.plot);
  }
}

// ---------------------------------------------------------------------------
// 4. The fit, step by step
// ---------------------------------------------------------------------------

let fitSeen = false, traceMark = null;

const sigmaOf = (s) => -s.LL + sumParts(s.parts);

function renderFit() {
  if (!S.fit) return;
  const snaps = S.fit.snaps, s0 = snaps[0], P0 = sumParts(s0.parts);
  $("c-frame").max = snaps.length - 1;
  const gain = snaps.map((s) => bits(s.LL - s0.LL)), cost = snaps.map((s) => bits(sumParts(s.parts) - P0)), net = gain.map((g, k) => g - cost[k]);
  const f = frame(clear("p-trace"), {
    height: 260, x: { domain: [0, snaps.length - 1], label: "accepted move batch" }, xFormat: (v) => (Number.isInteger(v) ? String(v) : ""),
    y: { domain: [0, Math.max(...gain) * 1.06], label: "bits" }, yFormat: (v) => fmt(v, 0),
  });
  line(f, gain.map((v, k) => [k, v]), C.g1);
  line(f, cost.map((v, k) => [k, v]), C.g2);
  line(f, net.map((v, k) => [k, v]), MDLc, { "stroke-width": 2.5 });
  const n = snaps.length - 1;
  label(f, n, gain[n], "data bits saved", { anchor: "end", dx: -2, dy: -8 });
  label(f, n, net[n], "net saving", { anchor: "end", dx: -2, dy: 16, cls: "direct-label strong" });
  label(f, n, cost[n], "bits for W", { anchor: "end", dx: -2, dy: -8 });
  traceMark = { f, m: el("line", { y1: f.margin.t, y2: f.margin.t + f.ih, stroke: C.ink2, "stroke-width": 1, opacity: 0.6 }, f.plot) };
  // weight-axis range over the whole run
  let lo = 0, hi = 0;
  for (const s of snaps) for (const v of s.W) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
  for (const v of S.net.W) hi = Math.max(hi, v);
  S.fitRange = [Math.min(lo, -0.02) - 0.02, hi + 0.04];
  setFrame(S.frame);
}

function setFrame(k) {
  const snaps = S.fit.snaps;
  k = Math.max(0, Math.min(snaps.length - 1, k));
  S.frame = k;
  const s = snaps[k];
  $("c-frame").value = k;
  $("o-frame").textContent = `${k} / ${snaps.length - 1}`;
  const { f, m } = traceMark;
  m.setAttribute("x1", f.xs(k)); m.setAttribute("x2", f.xs(k));
  const W = Float64Array.from(s.W), sim = I.similarity(S.net.W, W);
  const t = tiles(clear("t-fit"), [
    { label: "Sweep · move", value: s.sweep ? `${s.sweep} · ${s.phase}` : "start", note: s.moves ? `${s.moves} accepted` : "empty network, no categories" },
    { label: "Σ", value: `${fmt(bits(sigmaOf(s)), 0)} bits`, note: `data ${fmt(bits(-s.LL), 0)} + W ${fmt(bits(sumParts(s.parts)), 0)}` },
    { label: "Edges E", value: s.E, note: "true 78" },
    { label: "Categories K", value: s.z.length },
    { label: "s(W, Ŵ)", value: fmt(sim.sW) },
    { label: "s(A, Â)", value: fmt(sim.sA) },
  ]);
  t.children[0].querySelector(".tile-value").classList.add("phase");
  drawNet(clear("p-fitnet"), W, { ref: S.net.W, height: 280 });
  renderStrip(W, s);
  renderLedger(s);
}

function renderStrip(W, s) {
  const box = clear("p-strip"), [lo, hi] = S.fitRange;
  const f = frame(box, { height: 230, margin: { t: 26, r: 14, b: 40, l: 14 }, x: { domain: [lo, hi], label: "weight" }, xFormat: (v) => fmt(v, 2), y: { domain: [0, 1], ticks: [] } });
  for (const v of S.net.W) if (v) el("line", { x1: f.xs(v), x2: f.xs(v), y1: f.ys(0), y2: f.ys(0) - 7, stroke: C.ink2, opacity: 0.5 }, f.plot);
  s.z.forEach((z, k) => {
    el("line", { x1: f.xs(z), x2: f.xs(z), y1: f.ys(0.06), y2: f.ys(1), stroke: MDLc, "stroke-width": 1.5, opacity: s.z.length > 6 ? 0.5 : 1 }, f.plot);
    if (s.z.length > 4) return;
    const t = el("text", { x: f.xs(z), y: f.margin.t - 8 - 11 * (k % 2), "text-anchor": "middle", class: "direct-label strong" }, f.svg);
    t.textContent = `${fmt(z, 3)} · m = ${s.m[k]}`;
  });
  if (s.z.length > 4) { const t = el("text", { x: f.margin.l + f.iw / 2, y: f.margin.t - 10, "text-anchor": "middle", class: "direct-label strong" }, f.svg); t.textContent = `${s.z.length} categories, most holding one or two edges`; }
  for (let p = 0; p < PP.P; p++) {
    if (!W[p]) continue;
    const jit = 0.12 + 0.82 * ((p * 0.618034) % 1);
    el("circle", { cx: f.xs(W[p]), cy: f.ys(jit), r: 2.6, fill: S.net.W[p] ? C.g1 : C.g2, opacity: 0.85 }, f.plot);
  }
  el("line", { x1: f.xs(0), x2: f.xs(0), y1: f.ys(0), y2: f.ys(1), stroke: C.axis }, f.plot);
}

const LEDGER = [
  ["edges", "which pairs are edges", "log(P+1) + log C(P, E)"],
  ["K", "how many categories", "log E"],
  ["assign", "which category each edge has", "log E! − Σ log m_k! + log C(E−1, K−1)"],
  ["values", "the category values", "Σ λ|z_k| − log(e^{λΔ}−1) + log 2"],
];
const LEDGER_C = [C.neutral, C.muted, C.g1, MDLc];

function renderLedger(s) {
  const box = clear("p-ledger"), last = S.fit.snaps[S.fit.snaps.length - 1];
  const vals = LEDGER.map(([k]) => bits(s.parts[k])), maxTot = Math.max(...S.fit.snaps.map((q) => bits(sumParts(q.parts))));
  const f = frame(box, { height: 230, margin: { t: 10, r: 70, b: 40, l: 150 }, x: { domain: [0, maxTot * 1.02], label: "bits" }, xFormat: (v) => fmt(v, 0), y: { domain: [0, LEDGER.length], ticks: [] } });
  LEDGER.forEach(([, name, formula], r) => {
    const y0 = f.ys(LEDGER.length - r - 0.2), y1 = f.ys(LEDGER.length - r - 0.8);
    const rect = el("rect", { x: f.xs(0), y: y0, width: Math.max(1, f.xs(vals[r]) - f.xs(0)), height: y1 - y0, fill: LEDGER_C[r], opacity: 0.9 }, f.plot);
    const t = el("text", { x: f.margin.l - 8, y: (y0 + y1) / 2 + 4, "text-anchor": "end", class: "direct-label" }, f.svg); t.textContent = name;
    const v = el("text", { x: f.xs(vals[r]) + 6, y: (y0 + y1) / 2 + 4, class: "direct-label strong" }, f.plot); v.textContent = fmt(vals[r], 1);
    rect.addEventListener("pointermove", (e) => showTip(e, [{ value: `${fmt(vals[r], 1)} bits`, label: formula }, { value: `${fmt(bits(last.parts[LEDGER[r][0]]), 1)} bits`, label: "at the end of the fit" }], name));
    rect.addEventListener("pointerleave", hideTip);
  });
}

function play() {
  if (!S.fit) return;
  stopPlay();
  if (S.frame >= S.fit.snaps.length - 1) setFrame(0);
  $("c-play").textContent = "Pause";
  S.playing = setInterval(() => {
    if (S.frame >= S.fit.snaps.length - 1) return stopPlay();
    setFrame(S.frame + 1);
  }, 220);
}
function stopPlay() { clearInterval(S.playing); S.playing = null; $("c-play").textContent = "Play"; }

// ---------------------------------------------------------------------------
// 5. What categories buy
// ---------------------------------------------------------------------------

function renderCats() {
  if (!S.fit) return;
  const ks = S.fit.ks.filter(Boolean);
  const tot = ks.map((r) => bits(-r.LL + sumParts(r.parts))), dat = ks.map((r) => bits(-r.LL)), mod = ks.map((r) => bits(sumParts(r.parts)));
  const b = tot.indexOf(Math.min(...tot));
  const rel = (a) => a.map((v) => v - a[b]);
  const T = rel(tot), D = rel(dat), Mo = rel(mod), all = [...T, ...D, ...Mo];
  const f = frame(clear("p-K"), {
    height: 260, x: { domain: [0.7, ks.length + 0.3], label: "categories K", ticks: ks.map((r) => r.K) }, xFormat: (v) => String(v),
    y: { domain: [Math.min(...all) - 10, Math.max(...all) + 10], label: "bits, relative to the best K" }, yFormat: (v) => fmt(v, 0),
  });
  refLine(f, [[0.7, 0], [ks.length + 0.3, 0]]);
  [[D, C.g1, "data"], [Mo, C.g2, "bits for W"], [T, MDLc, "Σ"]].forEach(([a, c, name]) => {
    line(f, a.map((v, k) => [ks[k].K, v]), c, { "stroke-width": name === "Σ" ? 2.5 : 2 });
    a.forEach((v, k) => {
      const d = dot(f, ks[k].K, v, c, k === b && name === "Σ" ? 6 : 3.5);
      d.addEventListener("pointermove", (e) => showTip(e, [{ value: `${fmt(v, 1)} bits`, label: name }, ...ks[k].z.map((z, q) => ({ value: fmt(z, 3), label: `z, ${ks[k].m[q]} edges` }))], `K = ${ks[k].K}`));
      d.addEventListener("pointerleave", hideTip);
    });
    label(f, ks[ks.length - 1].K, a[a.length - 1], name, { dx: -4, dy: name === "bits for W" ? -8 : 16, anchor: "end", cls: name === "Σ" ? "direct-label strong" : "direct-label" });
  });

  renderWvW();
}

function renderWvW() {
  if (!S.fit || !S.l1) return;
  const Wm = S.fit.snaps[S.fit.snaps.length - 1].W, Wl = S.l1.Ws[S.li], tw = Array.from(S.net.W).filter((v) => v);
  const x0 = Math.min(...tw), x1 = Math.max(...tw), pad = Math.max(0.03, 0.15 * (x1 - x0));
  let hi = x1 + pad; for (let p = 0; p < PP.P; p++) if (S.net.W[p]) hi = Math.max(hi, Wm[p], Wl[p]);
  const box = clear("p-wvw");
  legend(box, [{ label: "MDL", color: MDLc, type: "dot" }, { label: `L1 at λ = ${fmt(S.lams[S.li], 1)} (§1 slider)`, color: L1c, type: "dot" }]);
  const g = frame(box, { height: 240, x: { domain: [Math.max(0, x0 - pad), x1 + pad], label: "true weight" }, xFormat: (v) => fmt(v, 2), y: { domain: [0, hi * 1.04], label: "fitted weight" }, yFormat: (v) => fmt(v, 2) });
  refLine(g, [[Math.max(0, x0 - pad), Math.max(0, x0 - pad)], [x1 + pad, x1 + pad]], C.ink2);
  for (let p = 0; p < PP.P; p++) {
    if (!S.net.W[p]) continue;
    el("circle", { cx: g.xs(S.net.W[p]), cy: g.ys(Math.max(0, Wl[p])), r: 2.6, fill: L1c, opacity: 0.75 }, g.plot);
    el("circle", { cx: g.xs(S.net.W[p]), cy: g.ys(Math.max(0, Wm[p])), r: 2.6, fill: MDLc, opacity: 0.9 }, g.plot);
  }
}

// ---------------------------------------------------------------------------
// 6. Comparison over M (precomputed)
// ---------------------------------------------------------------------------

async function loadCompare() {
  try { S.compare = await (await fetch(new URL("../data/compare.json", import.meta.url))).json(); } catch { S.compare = null; }
  renderCompare();
}

function renderCompare() {
  const C2 = S.compare;
  if (!C2) return;
  const rows = C2.modes[S.mode], LM = C2.Ms.map(Math.log10), methods = [["mdl", "MDL", MDLc], ["normal", "true (normal) prior", NORMc], ["l1", "L1 + 5-fold CV", L1c]];
  const xopt = { domain: [LM[0] - 0.1, LM[LM.length - 1] + 0.1], label: "samples M", ticks: LM }, xf = (v) => String(Math.round(10 ** v));
  const chart = (id, get, yopt, ref = null) => {
    const box = clear(id);
    if (id === "p-cW") legend(box, methods.map(([, n, c]) => ({ label: n, color: c, type: "dot" })));
    const f = frame(box, { height: 230, x: xopt, xFormat: xf, ...yopt });
    if (ref !== null) refLine(f, [[xopt.domain[0], ref], [xopt.domain[1], ref]]);
    methods.forEach(([key, name, c], mi) => {
      const pts = rows.map((seedRows, k) => { const v = seedRows.map((r) => get(r[key])); return [LM[k] + (mi - 1) * 0.025, v.reduce((a, b) => a + b, 0) / v.length, Math.min(...v), Math.max(...v)]; });
      line(f, pts.map(([x, y]) => [x, y]), c, { "stroke-width": 1.5, opacity: 0.8 });
      pts.forEach(([x, y, lo, hi], k) => {
        el("line", { x1: f.xs(x), x2: f.xs(x), y1: f.ys(lo), y2: f.ys(hi), stroke: c, "stroke-width": 1.2 }, f.plot);
        const d = dot(f, x, y, c, 3.5);
        d.addEventListener("pointermove", (e) => showTip(e, [{ value: yopt.tip(y), label: "mean" }, { value: `${yopt.tip(lo)} – ${yopt.tip(hi)}`, label: "range over 3 data sets" }], `${name} · M = ${C2.Ms[k]}`));
        d.addEventListener("pointerleave", hideTip);
      });
    });
  };
  chart("p-cW", (r) => r.sW, { y: { domain: [0, 1.02], label: "s(W, Ŵ)" }, yFormat: (v) => fmt(v, 1), tip: (v) => fmt(v) });
  chart("p-cA", (r) => r.sA, { y: { domain: [0, 1.02], label: "s(A, Â)" }, yFormat: (v) => fmt(v, 1), tip: (v) => fmt(v) });
  chart("p-cE", (r) => Math.log10(Math.max(1, r.E)), { y: { domain: [0, Math.log10(PP.P)], label: "E", ticks: [0, 1, 2, Math.log10(PP.P)] }, yFormat: (v) => (v === 0 ? "1" : String(Math.round(10 ** v))), tip: (v) => String(Math.round(10 ** v)) }, Math.log10(78));
}

// ---------------------------------------------------------------------------
// Controls
// ---------------------------------------------------------------------------

function debounce(fn, ms) { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; }
const recomputeSoon = debounce(recompute, 180);

$("c-mode").addEventListener("click", (e) => {
  const b = e.target.closest("button"); if (!b) return;
  for (const x of $("c-mode").children) x.classList.toggle("on", x === b);
  S.mode = b.dataset.m; recompute();
});
$("c-M").addEventListener("input", () => { S.M = Ms[+$("c-M").value]; $("o-M").textContent = S.M; recomputeSoon(); });
$("c-seed").addEventListener("click", () => { S.seed++; recompute(); });
$("c-lam").addEventListener("input", () => setLam(+$("c-lam").value));
$("c-lam-cv").addEventListener("click", () => setLam(S.liCV));
$("c-lam-e").addEventListener("click", () => setLam(S.liE));
const densSoon = () => requestAnimationFrame(renderDensity);
$("c-dl").addEventListener("input", densSoon);
$("c-dd").addEventListener("input", densSoon);
$("c-dens-kind").addEventListener("click", (e) => {
  const b = e.target.closest("button"); if (!b) return;
  for (const x of $("c-dens-kind").children) x.classList.toggle("on", x === b);
  S.densKind = b.dataset.k; renderDensity();
});
const priorSoon = () => requestAnimationFrame(renderPrior);
$("c-pE").addEventListener("input", priorSoon);
$("c-pK").addEventListener("input", priorSoon);
$("c-pdraw").addEventListener("click", () => { priorSeed++; renderPrior(); });
$("c-play").addEventListener("click", () => (S.playing ? stopPlay() : play()));
$("c-back").addEventListener("click", () => { stopPlay(); setFrame(S.frame - 1); });
$("c-fwd").addEventListener("click", () => { stopPlay(); setFrame(S.frame + 1); });
$("c-frame").addEventListener("input", () => { stopPlay(); setFrame(+$("c-frame").value); });

// play the fit the first time it scrolls into view
new IntersectionObserver((entries, obs) => {
  if (entries.some((e) => e.isIntersecting)) { fitSeen = true; obs.disconnect(); if (S.fit) play(); }
}, { threshold: 0.25 }).observe($("p-trace"));

// highlight the current section in the side nav
const links = [...document.querySelectorAll("nav.side a")];
const navObs = new IntersectionObserver((entries) => {
  for (const e of entries) if (e.isIntersecting) links.forEach((a) => a.classList.toggle("active", a.getAttribute("href") === `#${e.target.id}`));
}, { rootMargin: "-40% 0px -55% 0px" });
document.querySelectorAll("section").forEach((s) => navObs.observe(s));

$("c-M").value = Ms.indexOf(S.M);
$("o-M").textContent = S.M;
renderPrior();
initReplicas();
initMoves({ fit: () => S.fit, frame: () => S.frame, data: () => S.data, net: () => S.net, drawNet });
loadCompare();
recompute();
window.addEventListener("resize", debounce(() => { renderData(); renderDensity(); renderPrior(); renderL1(); renderFit(); renderCats(); renderCompare(); movesRedraw(); }, 200));
