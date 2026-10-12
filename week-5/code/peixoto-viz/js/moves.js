// §5: every move of one sweep (§V of the paper), on a real state of the fit, with its random bisection search animated.
// The logic is in search.js; this file only draws.

import { C, el, frame, line, dot, refLine, tiles, fmt, showTip, hideTip } from "../../../../week-3/code/viz-common/charts.js";
import { clipped } from "../../../../week-4/code/viz-common/charts4.js";
import { Fit, makeRng } from "./ising.js";
import * as X from "./search.js";

const $ = (id) => document.getElementById(id);
const clear = (id) => { const e = $(id); e.replaceChildren(); return e; };
const MDLc = C.model, ADD = C.model, DEL = C.g2, SIDES = [C.model, C.edge]; // the two sides of a split

const MOVES = {
  cand: { name: "0 · candidate pairs", eq: "Eqs. (28)–(29)" },
  cat: { name: "1 · edge → existing category", eq: "option (1)" },
  new: { name: "1 · edge → new value", eq: "option (2)" },
  zero: { name: "1 · edge → 0", eq: "option (3)" },
  mv: { name: "2 · edge move", eq: "W_iu ← W_ij, W_ij ← 0" },
  swap: { name: "3 · edge swap", eq: "(i,j),(u,v) → (i,v),(u,j)" },
  val: { name: "4 · category value", eq: "z_k ← z_k′" },
  merge: { name: "5 · merge", eq: "z_k, z_l → z_m" },
  split: { name: "5 · split", eq: "z_m → z_k, z_l" },
  msplit: { name: "5 · merge-split", eq: "merge, then split" },
};

const M = { visible: false, dirty: false, ctx: null, F: null, data: null, move: "cat", which: "early", random: true, plan: null, res: null, step: 0, timer: null, rep: null, rng: makeRng(11), seq: 0 };

// ---------------------------------------------------------------------------

function snapshot() {
  const snaps = M.ctx.fit().snaps;
  if (M.which === "final") return { s: snaps[snaps.length - 1], label: "the converged fit" };
  if (M.which === "step") { const k = M.ctx.frame(); return { s: snaps[k], label: `§4 step ${k}` }; }
  let b = 0;
  snaps.forEach((s, k) => { if (s.z.length > snaps[b].z.length) b = k; });
  return { s: snaps[b], label: `early in the fit (§4 step ${b}, the most categories)` };
}

function freshState() {
  const data = M.ctx.data();
  if (M.data !== data) { M.F = new Fit(data); M.data = data; }
  const { s, label } = snapshot();
  return { mdl: X.stateFrom(M.F, s, 1), snap: s, label };
}

function pick(arr) { return arr[Math.floor(M.rng.uniform() * arr.length)]; }

// One proposal of the current move type, on a fresh copy of the chosen state
function propose(findMulti = false) {
  stop();
  M.dirty = false;
  const { mdl, snap, label } = freshState();
  M.mdl = mdl; M.snap = snap; M.rep = null; M.res = null; M.step = 0;
  const cand = X.candidates(mdl), Eset = [...new Set(cand.keep.concat(mdl.nonzero()))];
  let plan = null;
  const tries = findMulti ? 300 : 40;
  for (let t = 0; t < tries && !plan; t++) {
    switch (M.move) {
      case "cand": plan = { kind: "cand", ...cand }; break;
      case "cat": {
        plan = X.edgeToCategory(mdl, pick(Eset));
        if (plan && findMulti && countMinima(plan) < 2) plan = null;
        break;
      }
      case "new": plan = X.edgeToNewValue(mdl, pick(Eset)); break;
      case "zero": { const nz = mdl.nonzero(); plan = nz.length ? X.edgeToZero(mdl, pick(nz)) : null; break; }
      case "mv": plan = X.edgeMove(mdl, M.rng, new Set(cand.keep)); break;
      case "swap": plan = X.edgeSwap(mdl, M.rng); break;
      case "val": plan = mdl.z.length ? X.categoryValue(mdl, Math.floor(M.rng.uniform() * mdl.z.length)) : null; break;
      case "merge": plan = X.merge(mdl, M.rng); break;
      case "split": { const ks = mdl.m.map((c, k) => k).filter((k) => mdl.m[k] >= 2); plan = ks.length ? X.split(mdl, pick(ks), M.rng, { random: M.random }) : null; break; }
      case "msplit": plan = X.mergeSplit(mdl, M.rng, { random: M.random }); break;
    }
  }
  M.plan = plan; M.label = label;
  $("o-mstate").textContent = `${label}: E = ${snap.E}, K = ${snap.z.length}`;
  $("c-mmulti").hidden = M.move !== "cat";
  if (!plan) { renderEmpty(); return; }
  if (plan.kind === "continuous") {
    const n = 81;
    plan.grid = Array.from({ length: n }, (_, i) => { const v = plan.lo + ((plan.hi - plan.lo) * i) / (n - 1); return [v, plan.g(v)]; });
  } else if (plan.kind === "discrete") plan.vals = Array.from({ length: plan.n }, (_, i) => plan.g(i));
  if (plan.kind === "split") plan.pref = plan.members.map((p) => { const [g, c] = M.F.gradCurv(p); return M.F.W[p] + g / c; });
  runSearch();
}

// Local minima with a barrier of at least `bar` bits on both sides (so tiny wiggles don't count as dips)
function countMinima(plan, bar = 1) {
  const v = Array.from({ length: plan.n }, (_, i) => plan.g(i));
  let n = 0;
  for (let i = 0; i < v.length; i++) {
    if (!((i === 0 || v[i] < v[i - 1]) && (i === v.length - 1 || v[i] < v[i + 1]))) continue;
    let left = i === 0, right = i === v.length - 1;
    for (let j = i - 1; j >= 0 && !left; j--) { if (v[j] < v[i]) break; if (v[j] >= v[i] + bar) left = true; }
    for (let j = i + 1; j < v.length && !right; j++) { if (v[j] < v[i]) break; if (v[j] >= v[i] + bar) right = true; }
    if (left && right) n++;
  }
  return n;
}

function runSearch() {
  const P = M.plan;
  if (P.kind === "continuous") M.res = X.bisect(P.g, P.lo, P.hi, { rng: M.rng, random: M.random });
  else if (P.kind === "discrete") M.res = X.bisectDiscrete(P.g, P.n, { rng: M.rng, random: M.random });
  M.step = 0;
  draw();
  play();
}

function nSteps() {
  const P = M.plan;
  if (!P) return 0;
  if (P.kind === "continuous" || P.kind === "discrete") return M.res.steps.length;
  if (P.kind === "split") return P.frames.length;
  return 1;
}

function play() {
  stop();
  const n = nSteps();
  if (n <= 1) { M.step = Math.max(0, n - 1); draw(); return; }
  M.step = 0; draw();
  const dt = M.plan.kind === "split" ? 700 : 260;
  M.timer = setInterval(() => {
    if (M.step >= n - 1) return stop();
    M.step++; draw();
  }, dt);
}
function stop() { clearInterval(M.timer); M.timer = null; }

// Where 50 random searches end, against one deterministic search
function repeat() {
  const P = M.plan;
  if (!P || (P.kind !== "continuous" && P.kind !== "discrete")) return;
  const run = (random) => (P.kind === "continuous" ? X.bisect(P.g, P.lo, P.hi, { rng: M.rng, random }) : X.bisectDiscrete(P.g, P.n, { rng: M.rng, random }));
  const ends = Array.from({ length: 50 }, () => run(true));
  const det = run(false);
  let best;
  if (P.kind === "continuous") { const r = X.bisect(P.g, P.lo, P.hi, { rng: M.rng, random: false, tol: 1e-6, maxIter: 200 }); best = Math.min(r.g, ...P.grid.map((q) => q[1])); }
  else best = Math.min(...P.vals);
  const hit = (r) => r.g <= best + 0.05;
  M.rep = { ends, det, hits: ends.filter(hit).length, detHit: hit(det) };
  draw();
}

// ---------------------------------------------------------------------------
// Drawing
// ---------------------------------------------------------------------------

function verdict(dS) { return dS < -1e-9 ? "accepted" : "rejected"; }

const WHY = {
  zero: "there are no edges yet", mv: "there are no edges yet", swap: "it needs two edges with four distinct nodes",
  val: "there are no categories yet", merge: "it needs at least two categories", msplit: "it needs at least two categories",
  split: "it needs a category with at least two edges", cat: "it needs a category the edge isn't already in",
};
function renderEmpty() {
  tiles(clear("t-moves"), [{ label: "Move", value: MOVES[M.move].name, note: MOVES[M.move].eq }, { label: "Nothing to propose", value: "–", note: `in this state ${WHY[M.move] ?? "nothing applies"}; try another state` }]);
  $("h-msearch").textContent = "Nothing to propose"; $("s-msearch").textContent = "";
  clear("p-msearch"); clear("p-mnet");
  M.ctx.drawNet($("p-mnet"), stateW(), { ref: M.ctx.net().W, height: 300 });
}

function stateW() { return Float64Array.from(M.F.W); }

const CARD = {
  continuous: ["The search", "Grey: ΔΣ in bits for every value (computed only for this picture). Circles: the probes, in order. Shaded: the current bracket. Black dot: the best so far. After \"Repeat\", violet ticks mark where 50 random searches ended, and the black triangle where the deterministic one did."],
  discrete: ["The search", "The categories this edge could join, in sorted order. Grey: ΔΣ for each. Circles: the probes, in order. Shaded: the bracket. After \"Repeat\", violet ticks mark where 50 random searches ended, the black triangle where the deterministic one did."],
  single: ["The change in Σ", "No search: one proposal, kept only if the total ΔΣ is negative. Moves and swaps leave the counts, and so the prior, unchanged."],
  cand: ["Candidate scores", "Every zero pair, ranked by how much switching it on would help. Only the top κN (violet) are visited by the edge updates in this sweep, together with the current edges."],
  split: ["The split", "Each edge of the category at the weight it would pick on its own, coloured by side. Lines: the two values. Shaded: the range they start in, between the neighbouring categories."],
};

function draw() {
  const P = M.plan;
  if (!P) return;
  const [h, sub] = CARD[P.kind];
  $("h-msearch").textContent = h; $("s-msearch").textContent = sub;
  const box = clear("p-msearch");
  const info = { label: "Move", value: MOVES[M.move].name, note: MOVES[M.move].eq };
  const pr = (p) => `${M.F.I[p] + 1}–${M.F.J[p] + 1}`;
  let rows = [info], marks = [];
  if (P.kind === "cand") {
    drawCandidates(box, P);
    marks = P.keep.map((p) => ({ p, color: ADD, width: 1.6, opacity: 0.6 }));
    rows.push({ label: "Scored by", value: P.gradient ? "|∂ log P / ∂W|" : "best gain", note: P.gradient ? "Eq. (29): no categories yet" : "Eq. (28): switch on at the smallest ± category" },
      { label: "Kept", value: `${P.n} of ${P.scores.length}`, note: "κN zero pairs, κ = 1; the paper finds them with NNDescent" });
  } else if (P.kind === "continuous" || P.kind === "discrete") {
    const s = M.res.steps[M.step], done = M.step === M.res.steps.length - 1;
    P.kind === "continuous" ? drawContinuous(box, P, s) : drawDiscrete(box, P, s);
    const target = P.p !== undefined ? `pair ${pr(P.p)}` : P.pair ? `categories ${P.pair.map((k) => fmt(M.mdl.z[k], 3)).join(" + ")}` : `category ${fmt(P.from, 3)}`;
    const xval = P.kind === "continuous" ? s.b : P.xs[s.b];
    const q = P.parts(s.b);
    rows.push({ label: "Target", value: target, note: P.members ? `${P.members.length} edges move together` : `now W = ${fmt(P.from, 3)}` },
      { label: done ? "Best value found" : "Best so far", value: fmt(xval, 3), note: `probe ${M.step + 1} of ${M.res.steps.length}` },
      { label: "ΔΣ", value: `${fmt(s.gb, 1)} bits`, note: `data ${fmt(q.data, 1)} + W ${fmt(q.prior, 1)}` });
    if (done) rows.push({ label: "Verdict", value: verdict(s.gb), note: s.gb < 0 ? "Σ gets shorter" : "Σ would get longer: keep the old value" });
    if (M.rep) rows.push({ label: "50 random searches", value: `${M.rep.hits} / 50`, note: `reached the best value · deterministic: ${M.rep.detHit ? "yes" : "no"}` });
    const ps = P.p !== undefined ? [P.p] : P.members;
    marks = ps.map((p) => ({ p, color: ADD, width: P.p !== undefined ? 4.5 : 2.6 }));
  } else if (P.kind === "single") {
    drawSingle(box, P);
    marks = P.removed.map((p) => ({ p, color: DEL, width: 4.5 })).concat(P.added.map((p) => ({ p, color: ADD, width: 4.5 })));
    const what = M.move === "zero" ? `pair ${pr(P.p)} → 0` : P.node !== undefined ? `edge ${pr(P.removed[0])} → ${pr(P.added[0])}` : `${P.removed.map(pr).join(", ") || "–"} → ${P.added.map(pr).join(", ") || "–"}`;
    rows.push({ label: "Proposal", value: what, note: M.move === "zero" ? `now W = ${fmt(P.from, 3)}` : "no search: one proposal" },
      { label: "ΔΣ", value: `${fmt(P.total, 1)} bits`, note: `data ${fmt(P.data, 1)} + W ${fmt(P.prior, 1)}` }, { label: "Verdict", value: verdict(P.total) });
  } else if (P.kind === "split") {
    const f = P.frames[M.step], done = M.step === P.frames.length - 1;
    drawSplit(box, P, f);
    marks = P.members.map((p, e) => ({ p, color: SIDES[f.side[e]], width: 2.6 }));
    rows.push({ label: "Target", value: P.merged ? `merge ${P.merged.map((k) => fmt(M.mdl.z[k], 3)).join(" + ")}` : `category ${fmt(P.from, 3)}`, note: `${P.members.length} edges` },
      { label: "Step", value: f.step, note: `${M.step + 1} of ${P.frames.length}` }, { label: "ΔΣ", value: `${fmt(f.dS, 1)} bits` });
    if (done) rows.push({ label: "Verdict", value: verdict(f.dS) });
  }
  const t = tiles(clear("t-moves"), rows);
  const v = [...t.children].find((c) => c.querySelector(".tile-label").textContent === "Verdict");
  if (v) v.classList.add(v.querySelector(".tile-value").textContent === "accepted" ? "good" : "bad");
  M.ctx.drawNet(clear("p-mnet"), stateW(), { ref: M.ctx.net().W, height: 300, marks });
}

// y-range that keeps the minimum and zero in view without letting far-away values flatten everything
function yRange(vals) {
  const lo = Math.min(0, ...vals), top = Math.min(Math.max(...vals), lo + Math.max(60, 3 * Math.abs(lo) + 40));
  const pad = 0.06 * (top - lo || 1);
  return [lo - pad, Math.max(top, 0) + pad];
}

function bracket(f, a, c, y0, y1) {
  el("rect", { x: f.xs(a), y: f.ys(y1), width: Math.max(1, f.xs(c) - f.xs(a)), height: f.ys(y0) - f.ys(y1), fill: MDLc, opacity: 0.1 }, f.plot);
}

function probes(f, P, upto, X, Y, y0, y1) {
  const steps = M.res.steps.slice(0, upto + 1);
  steps.forEach((s, i) => {
    const y = Math.min(y1, Math.max(y0, s.gx));
    const c = el("circle", { cx: f.xs(X(s.x)), cy: f.ys(y), r: i === upto ? 6 : 4, fill: i === upto ? MDLc : "none", stroke: MDLc, "stroke-width": 1.6 }, f.plot);
    c.addEventListener("pointermove", (e) => showTip(e, [{ value: fmt(X(s.x), 3), label: "probe" }, { value: `${fmt(s.gx, 1)} bits`, label: "ΔΣ" }, { value: s.better ? "yes" : "no", label: "better than the best so far" }], `probe ${i + 1}`));
    c.addEventListener("pointerleave", hideTip);
    if (i < 14) { const t = el("text", { x: f.xs(X(s.x)), y: f.ys(y) - 9, "text-anchor": "middle", class: "tick" }, f.plot); t.textContent = i + 1; }
  });
  const s = steps[upto];
  dot(f, X(s.b), Math.min(y1, Math.max(y0, s.gb)), C.ink, 4);
}

function rug(f, X, y0) {
  if (!M.rep) return;
  M.rep.ends.forEach((r, i) => el("line", { x1: f.xs(X(r.x)), x2: f.xs(X(r.x)), y1: f.ys(y0) - 2 - (i % 5) * 3, y2: f.ys(y0) - 6 - (i % 5) * 3, stroke: MDLc, "stroke-width": 1.4, opacity: 0.8 }, f.plot));
  const d = M.rep.det, x = f.xs(X(d.x)), y = f.ys(y0) - 24;
  el("path", { d: `M${x - 6},${y - 9} L${x + 6},${y - 9} L${x},${y} Z`, fill: C.ink }, f.plot);
}

function nowLine(f, x, y0, y1, text = "now") {
  el("line", { x1: f.xs(x), x2: f.xs(x), y1: f.ys(y0), y2: f.ys(y1), stroke: C.muted, "stroke-width": 1 }, f.plot);
  const t = el("text", { x: f.xs(x) + 4, y: f.ys(y1) + 12, class: "tick" }, f.plot); t.textContent = text;
}

function zeroLine(f, x0, x1) {
  refLine(f, [[x0, 0], [x1, 0]], C.ink2);
  const t = el("text", { x: f.xs(x1) - 4, y: f.ys(0) - 5, "text-anchor": "end", class: "tick" }, f.plot); t.textContent = "accept below 0";
}

function drawContinuous(box, P, s) {
  const vals = P.grid.map((q) => q[1]), [y0, y1] = yRange(vals.concat(M.res.steps.map((q) => q.gx).filter((v) => v < Math.min(...vals) + 200)));
  const f = frame(box, { height: 300, x: { domain: [P.lo, P.hi], label: P.members ? "value shared by the edges" : "proposed weight W′" }, xFormat: (v) => fmt(v, 1), y: { domain: [y0, y1], label: "ΔΣ (bits)" }, yFormat: (v) => fmt(v, 0) });
  bracket(f, s.a, s.c, y0, y1);
  zeroLine(f, P.lo, P.hi);
  clipped(f).appendChild(line(f, P.grid, C.neutral, { "stroke-width": 1.6 }));
  nowLine(f, P.from, y0, y1);
  probes(f, P, M.step, (x) => x, null, y0, y1);
  rug(f, (x) => x, y0);
}

function drawDiscrete(box, P, s) {
  // the search runs over the categories in sorted order, so plot them by rank; values label the axis
  const [y0, y1] = yRange(P.vals), n = P.n, xs = P.xs;
  const every = Math.max(1, Math.ceil(n / 8)), ticks = Array.from({ length: n }, (_, i) => i).filter((i) => i % every === 0 || i === n - 1);
  const f = frame(box, {
    height: 300, x: { domain: [-0.7, n - 0.3], label: "existing categories, sorted by value z_k", ticks }, xFormat: (i) => fmt(xs[Math.round(i)], 2),
    y: { domain: [y0, y1], label: "ΔΣ (bits)" }, yFormat: (v) => fmt(v, 0),
  });
  el("rect", { x: f.xs(s.a - 0.5), y: f.ys(y1), width: f.xs(s.c + 0.5) - f.xs(s.a - 0.5), height: f.ys(y0) - f.ys(y1), fill: MDLc, opacity: 0.1 }, f.plot);
  zeroLine(f, -0.7, n - 0.3);
  clipped(f).appendChild(line(f, xs.map((x, i) => [i, P.vals[i]]), C.axis, { "stroke-width": 1 }));
  xs.forEach((x, i) => {
    const c = dot(f, i, Math.min(y1, Math.max(y0, P.vals[i])), C.neutral, 3.2);
    c.addEventListener("pointermove", (e) => showTip(e, [{ value: fmt(x, 3), label: "z_k" }, { value: P.counts[i], label: "edges in it, m_k" }, { value: `${fmt(P.vals[i], 1)} bits`, label: "ΔΣ" }], `category ${i + 1} of ${n}`));
    c.addEventListener("pointerleave", hideTip);
  });
  // where the entry is now, placed by value among the sorted categories
  const r = xs.findIndex((x) => x > P.from), at = r < 0 ? n - 0.5 : r === 0 ? -0.5 : r - 1 + (P.from - xs[r - 1]) / (xs[r] - xs[r - 1]);
  nowLine(f, at, y0, y1, P.from === 0 ? "now: no edge" : "now");
  probes(f, P, M.step, (i) => i, null, y0, y1);
  rug(f, (i) => i, y0);
}

function drawSingle(box, P) {
  const rows = [["data", P.data, C.g1], ["bits for W", P.prior, C.g2], ["ΔΣ", P.total, MDLc]];
  const m = Math.max(5, ...rows.map((r) => Math.abs(r[1]))) * 1.15;
  const f = frame(box, { height: 220, margin: { t: 14, r: 24, b: 44, l: 100 }, x: { domain: [-m, m], label: "change in bits" }, xFormat: (v) => fmt(v, 0), y: { domain: [0, 3], ticks: [] } });
  rows.forEach(([name, v, c], r) => {
    const y = f.ys(3 - r - 0.25), h = f.ys(3 - r - 0.75) - y;
    el("rect", { x: Math.min(f.xs(0), f.xs(v)), y, width: Math.abs(f.xs(v) - f.xs(0)), height: h, fill: c, opacity: 0.85 }, f.plot);
    const t = el("text", { x: f.margin.l - 8, y: y + h / 2 + 4, "text-anchor": "end", class: "direct-label" }, f.svg); t.textContent = name;
    const u = el("text", { x: f.xs(v) + (v < 0 ? -6 : 6), y: y + h / 2 + 4, "text-anchor": v < 0 ? "end" : "start", class: "direct-label strong" }, f.plot); u.textContent = fmt(v, 1);
  });
  el("line", { x1: f.xs(0), x2: f.xs(0), y1: f.margin.t, y2: f.margin.t + f.ih, stroke: C.ink2 }, f.plot);
}

function drawCandidates(box, P) {
  const s = P.scores.map((r) => r.s), lo = Math.min(...s), hi = Math.max(...s), pad = 0.05 * (hi - lo || 1);
  const f = frame(box, { height: 300, x: { domain: [0, s.length], label: "zero pairs, ranked by score" }, xFormat: (v) => fmt(v, 0), y: { domain: [lo - pad, hi + pad], label: P.gradient ? "|gradient|" : "gain if switched on (bits)" }, yFormat: (v) => fmt(v, 0) });
  if (!P.gradient) zeroLine(f, 0, s.length);
  P.scores.forEach((r, i) => el("circle", { cx: f.xs(i + 0.5), cy: f.ys(r.s), r: 2.2, fill: i < P.n ? MDLc : C.muted, opacity: i < P.n ? 1 : 0.6 }, f.plot));
  el("line", { x1: f.xs(P.n), x2: f.xs(P.n), y1: f.margin.t, y2: f.margin.t + f.ih, stroke: C.ink2 }, f.plot);
  const t = el("text", { x: f.xs(P.n) + 5, y: f.margin.t + 12, class: "tick" }, f.plot); t.textContent = `κN = ${P.n}`;
}

function drawSplit(box, P, fr) {
  const xs = P.pref.concat(fr.val, [P.lo, P.hi]), lo = Math.min(...xs) - 0.03, hi = Math.max(...xs) + 0.03;
  const f = frame(box, { height: 300, margin: { t: 30, r: 16, b: 44, l: 16 }, x: { domain: [lo, hi], label: "weight" }, xFormat: (v) => fmt(v, 2), y: { domain: [0, 1], ticks: [] } });
  el("rect", { x: f.xs(P.lo), y: f.margin.t, width: f.xs(P.hi) - f.xs(P.lo), height: f.ih, fill: MDLc, opacity: 0.07 }, f.plot);
  const cols = SIDES;
  fr.val.forEach((v, s) => {
    el("line", { x1: f.xs(v), x2: f.xs(v), y1: f.margin.t, y2: f.margin.t + f.ih, stroke: cols[s], "stroke-width": 2.5 }, f.plot);
    const t = el("text", { x: f.xs(v), y: f.margin.t - 8, "text-anchor": "middle", class: "direct-label strong" }, f.svg); t.textContent = fmt(v, 3);
  });
  P.members.forEach((p, e) => el("circle", { cx: f.xs(P.pref[e]), cy: f.ys(0.1 + 0.8 * ((e * 0.618034) % 1)), r: 3.4, fill: cols[fr.side[e]], opacity: 0.9 }, f.plot));
  nowLine(f, P.from, 0, 1, "before");
}

// ---------------------------------------------------------------------------

export function initMoves(ctx) {
  M.ctx = ctx;
  const pills = $("c-move");
  for (const [k, v] of Object.entries(MOVES)) {
    const b = document.createElement("button");
    b.textContent = v.name; b.dataset.m = k; b.classList.toggle("on", k === M.move);
    pills.appendChild(b);
  }
  pills.addEventListener("click", (e) => {
    const b = e.target.closest("button"); if (!b) return;
    for (const x of pills.children) x.classList.toggle("on", x === b);
    M.move = b.dataset.m; if (ctx.fit()) propose();
  });
  $("c-mwhich").addEventListener("click", (e) => {
    const b = e.target.closest("button"); if (!b) return;
    for (const x of $("c-mwhich").children) x.classList.toggle("on", x === b);
    M.which = b.dataset.w; if (ctx.fit()) propose();
  });
  $("c-mrand").addEventListener("click", (e) => {
    const b = e.target.closest("button"); if (!b) return;
    for (const x of $("c-mrand").children) x.classList.toggle("on", x === b);
    M.random = b.dataset.r === "1";
    if (M.plan && (M.plan.kind === "continuous" || M.plan.kind === "discrete")) { M.rep = null; runSearch(); }
    else if (ctx.fit()) propose();
  });
  $("c-mnew").addEventListener("click", () => ctx.fit() && propose());
  $("c-mmulti").addEventListener("click", () => ctx.fit() && propose(true));
  $("c-mplay").addEventListener("click", () => M.plan && play());
  $("c-mrep").addEventListener("click", repeat);
  new IntersectionObserver((es) => { M.visible = es.some((e) => e.isIntersecting); if (M.visible) refresh(); else stop(); }, { rootMargin: "200px" }).observe($("moves"));
}

// Called when a new fit arrives. The proposal is computed only once the section is on screen.
export function movesReset() {
  stop();
  M.plan = null; M.data = null; M.dirty = true;
  if (M.visible) refresh();
}

function refresh() {
  if (!M.dirty || !M.ctx.fit()) return;
  M.dirty = false;
  propose(M.move === "cat");
}

export function movesRedraw() { if (M.plan) draw(); }
