// §8: broken replica symmetry in a toy modular Ising model (the picture behind Eq. 38).
// Eight replicas of the same equilibrium Ising model run heat-bath dynamics side by side; the page shows
// replica 1's spins, every replica's module magnetizations, and the overlap distribution P(q).
// The loop runs only while the section is on screen.

import { C } from "../../../../week-3/code/viz-common/charts.js";
import { makeRng } from "./ising.js";

const NM = 5, MS = 12, N = NM * MS, R = 8, WINDOW = 600, BINS = 33, PER_FRAME = 3;
const NEG = "#fab219"; // negative inter-module coupling (the page's warning amber)

// Couplings: all pairs within a module at +1, a sparse set of ±0.6 between modules
const rng = makeRng(7);
const W = Array.from({ length: N }, () => new Float64Array(N));
for (let i = 0; i < N; i++)
  for (let j = i + 1; j < N; j++) {
    const same = Math.floor(i / MS) === Math.floor(j / MS);
    const w = same ? 1 : rng.uniform() < 0.08 ? (rng.uniform() < 0.5 ? -0.6 : 0.6) : 0;
    W[i][j] = W[j][i] = w;
  }
const NBR = W.map((row) => { const a = []; row.forEach((w, j) => { if (w) a.push(j, w); }); return a; });

const POS = [];
for (let m = 0; m < NM; m++) {
  const a = -Math.PI / 2 + (2 * Math.PI * m) / NM;
  for (let s = 0; s < MS; s++) { const b = (2 * Math.PI * s) / MS; POS.push([0.5 + 0.32 * Math.cos(a) + 0.1 * Math.cos(b), 0.5 + 0.32 * Math.sin(a) + 0.1 * Math.sin(b)]); }
}

const $ = (id) => document.getElementById(id);
const st = { beta: 0.2, X: [], sweeps: 0, hist: [], switches: [], signs: [], running: false, visible: false };

const mags = (x) => { const m = new Float64Array(NM); for (let i = 0; i < N; i++) m[Math.floor(i / MS)] += x[i]; return m.map((v) => v / MS); };
// A module's sign only changes once its magnetization is clearly on the other side (|m| ≥ 0.5),
// so jitter near zero doesn't count as a macrostate switch.
function updateSigns(x, sg) {
  const m = mags(x);
  let flips = 0;
  for (let k = 0; k < NM; k++) {
    if (Math.abs(m[k]) < 0.5) continue;
    const s = m[k] > 0 ? 1 : -1;
    if (sg[k] && s !== sg[k]) flips++;
    sg[k] = s;
  }
  return flips;
}
const pattern = (sg) => (sg[0] < 0 ? sg.map((v) => -v) : sg).join(","); // a pattern and its global flip are one

function randomize() {
  st.X = Array.from({ length: R }, () => Int8Array.from({ length: N }, () => (Math.random() < 0.5 ? -1 : 1)));
  st.sweeps = 0; st.hist = []; st.switches = []; st.signs = st.X.map(() => new Array(NM).fill(0));
}

function sweep(x) {
  for (let k = 0; k < N; k++) {
    const i = Math.floor(Math.random() * N), a = NBR[i];
    let h = 0;
    for (let q = 0; q < a.length; q += 2) h += a[q + 1] * x[a[q]];
    x[i] = Math.random() < 1 / (1 + Math.exp(-2 * st.beta * h)) ? 1 : -1;
  }
}

// The height attribute is the CSS height; read it once, because setting c.height overwrites it with device pixels
function fit(id) {
  const c = $(id), r = window.devicePixelRatio || 1, w = c.clientWidth;
  c.dataset.h ??= c.getAttribute("height");
  const h = +c.dataset.h;
  if (c.width !== Math.round(w * r) || c.height !== h * r) { c.width = Math.round(w * r); c.height = h * r; c.style.height = `${h}px`; }
  const g = c.getContext("2d");
  g.setTransform(r, 0, 0, r, 0, 0);
  g.clearRect(0, 0, w, h);
  return [g, w, h];
}

const rgba = (hex, a) => `rgba(${[1, 3, 5].map((k) => parseInt(hex.slice(k, k + 2), 16)).join(",")},${a})`;

function drawNet() {
  const [g, w, h] = fit("rs-net"), S = Math.min(w, h), ox = (w - S) / 2, P = POS.map(([x, y]) => [ox + x * S, y * S]);
  g.lineWidth = 1;
  for (let i = 0; i < N; i++)
    for (let j = i + 1; j < N; j++) {
      const v = W[i][j];
      if (!v) continue;
      const same = Math.floor(i / MS) === Math.floor(j / MS);
      g.strokeStyle = same ? C.grid : rgba(v > 0 ? C.edge : NEG, 0.75);
      g.beginPath(); g.moveTo(...P[i]); g.lineTo(...P[j]); g.stroke();
    }
  const x = st.X[0];
  for (let i = 0; i < N; i++) {
    g.beginPath(); g.arc(P[i][0], P[i][1], 5.5, 0, 7);
    g.fillStyle = x[i] > 0 ? C.g2 : C.axis; g.fill();
    g.strokeStyle = C.surface; g.lineWidth = 1.2; g.stroke();
  }
}

function drawReps() {
  const [g, w, h] = fit("rs-reps"), L = 64, cw = (w - L - 4) / NM, ch = (h - 22) / R;
  g.font = "11px system-ui, sans-serif"; g.fillStyle = C.muted; g.textAlign = "center";
  for (let m = 0; m < NM; m++) g.fillText(`module ${m + 1}`, L + cw * (m + 0.5), 12);
  st.X.forEach((x, r) => {
    const m = mags(x), y = 20 + r * ch;
    g.textAlign = "right"; g.fillStyle = C.muted; g.fillText(`replica ${r + 1}`, L - 8, y + ch / 2 + 4);
    for (let k = 0; k < NM; k++) {
      const v = m[k];
      g.fillStyle = rgba(v >= 0 ? C.g2 : C.g1, 0.12 + 0.88 * Math.abs(v));
      g.beginPath(); g.roundRect(L + k * cw + 2, y + 2, cw - 4, ch - 4, 4); g.fill();
    }
  });
}

function drawPq() {
  const [g, w, h] = fit("rs-pq"), L = 40, bot = h - 30, top = 10, cnt = new Float64Array(BINS);
  for (const qs of st.hist) for (const q of qs) cnt[Math.min(BINS - 1, Math.floor(((q + 1) / 2) * BINS))]++;
  const mx = Math.max(1e-9, ...cnt), bw = (w - L - 8) / BINS;
  g.strokeStyle = C.axis; g.lineWidth = 1; g.beginPath(); g.moveTo(L, bot); g.lineTo(w - 8, bot); g.stroke();
  g.fillStyle = C.model;
  for (let b = 0; b < BINS; b++) { const y = bot - (cnt[b] / mx) * (bot - top); g.fillRect(L + b * bw + 1, y, bw - 2, bot - y); }
  g.fillStyle = C.muted; g.font = "11px system-ui, sans-serif"; g.textAlign = "center";
  [-1, -0.5, 0, 0.5, 1].forEach((q) => g.fillText(String(q), L + ((q + 1) / 2) * (w - L - 8), bot + 15));
  g.fillText("overlap q between two replicas", L + (w - L - 8) / 2, h - 2);
  g.save(); g.translate(14, (bot + top) / 2); g.rotate(-Math.PI / 2); g.fillText("P(q)", 0, 0); g.restore();
}

function setTile(id, v) { $(id).textContent = v; }

function frameStep() {
  if (!st.visible || document.hidden) { st.running = false; return; }
  for (let s = 0; s < PER_FRAME; s++) {
    st.X.forEach(sweep);
    st.sweeps++;
    st.X.forEach((x, r) => { for (let f = updateSigns(x, st.signs[r]); f > 0; f--) st.switches.push(st.sweeps); });
    const qs = [];
    for (let a = 0; a < R; a++) for (let b = a + 1; b < R; b++) { let q = 0; for (let i = 0; i < N; i++) q += st.X[a][i] * st.X[b][i]; qs.push(q / N); }
    st.hist.push(qs);
    if (st.hist.length > WINDOW) st.hist.shift();
  }
  while (st.switches.length && st.switches[0] < st.sweeps - 1000) st.switches.shift();
  const distinct = new Set(st.signs.filter((sg) => sg.every((v) => v)).map(pattern));
  setTile("rs-distinct", `${distinct.size} of ${R}`);
  setTile("rs-switch", st.sweeps < 200 ? "…" : String(Math.round((st.switches.length * 1000) / Math.min(1000, st.sweeps))));
  setTile("rs-sweeps", st.sweeps.toLocaleString());
  drawNet(); drawReps(); drawPq();
  requestAnimationFrame(frameStep);
}

function start() { if (!st.running && st.visible && !document.hidden) { st.running = true; requestAnimationFrame(frameStep); } }

function setBeta(v) {
  st.beta = +v; $("c-rbeta").value = v; $("o-rbeta").textContent = (+v).toFixed(2);
  st.hist = []; st.switches = [];
  for (const b of document.querySelectorAll("#c-rpreset button")) b.classList.toggle("on", Math.abs(+b.dataset.b - st.beta) < 1e-9);
}

export function initReplicas() {
  randomize();
  setBeta(0.2);
  $("c-rbeta").addEventListener("input", () => setBeta($("c-rbeta").value));
  $("c-rpreset").addEventListener("click", (e) => { const b = e.target.closest("button"); if (b) setBeta(b.dataset.b); });
  $("c-rreset").addEventListener("click", randomize);
  new IntersectionObserver((es) => { st.visible = es.some((e) => e.isIntersecting); start(); }, { rootMargin: "100px" }).observe($("replicas"));
  document.addEventListener("visibilitychange", start);
}
