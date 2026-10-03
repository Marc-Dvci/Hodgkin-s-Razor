// Evidence views: the pre-registered blind tests and the chip planner.
// Every number is drawn from evidence.json, written by scripts/export_evidence.py
// from the frozen results; each AUROC is recomputed here from the per-preparation
// values, and the exporter refuses to write if its recomputation misses the
// scored primary.

const $ = (id) => document.getElementById(id);
const NS = "http://www.w3.org/2000/svg";
const C = { null: "#8b98a8", treat: "#5eb0ff", up: "#f0883e", ok: "#3fb950", line: "#232b36", bar: "#d34d4d" };
let EV = null;
const state = { method: "twin", nullg: "matched", mech: "g_nmda", chips: 6 };

// ---------- routing ----------
function show(view) {
  for (const v of ["analyse", "blind", "chips"]) {
    const el = $("view-" + v);
    if (el) el.hidden = v !== view;
  }
  document.querySelectorAll("nav.tabs a").forEach(a => a.classList.toggle("on", a.dataset.view === view));
  if (view !== "analyse" && EV) (view === "blind" ? drawBlind : drawChips)();
}
function route() {
  const h = (location.hash || "#analyse").slice(1);
  show(["analyse", "blind", "chips"].includes(h) ? h : "analyse");
}
window.addEventListener("hashchange", route);

// ---------- helpers ----------
function auroc(pos, neg) {
  let s = 0;
  for (const a of pos) for (const b of neg) s += a > b ? 1 : a === b ? 0.5 : 0;
  return s / (pos.length * neg.length);
}
const median = (xs) => { const s = [...xs].sort((a, b) => a - b), n = s.length; return n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2; };
function svg(el, w, h) { el.setAttribute("viewBox", `0 0 ${w} ${h}`); el.innerHTML = ""; return el; }
function add(parent, tag, attrs, text) {
  const e = document.createElementNS(NS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  if (text !== undefined) e.textContent = text;
  parent.appendChild(e);
  return e;
}
function tipOn(e, html) { const t = $("tip"); t.innerHTML = html; t.style.display = "block"; t.style.left = (e.clientX + 12) + "px"; t.style.top = (e.clientY + 12) + "px"; }
function tipOff() { $("tip").style.display = "none"; }
function hover(node, html) { node.addEventListener("mousemove", e => tipOn(e, html)); node.addEventListener("mouseleave", tipOff); }
function kpi(cls, big, small) { return `<div class="kpi ${cls}"><b>${big}</b><span>${small}</span></div>`; }
function seeded(i) { const x = Math.sin(i * 12.9898 + 78.233) * 43758.5453; return x - Math.floor(x); }

// ---------- blind test ----------
function groups() {
  const v = EV.v3;
  const T = v.preps.filter(p => p.kind === "treated");
  const N = v.preps.filter(p => p.kind === "null" && (state.nullg === "pooled" || p.matched));
  return { T, N };
}
function value(p, key) {
  if (state.method === "twin") return p.p[EV.mechanisms.findIndex(m => m.key === key)];
  return p[state.method];
}

function drawBlind() {
  const v = EV.v3;
  $("v3kpis").innerHTML =
    kpi("good", v.primary.auroc.toFixed(2), `AUROC of p(NMDA moved), ${v.primary.n_treated} treated vs ${v.primary.n_null} untreated preparations of the same genotypes. 95% CI ${v.primary.ci95[0].toFixed(2)} to ${v.primary.ci95[1].toFixed(2)}; pre-registered bar ${v.bar.toFixed(2)}: met`) +
    kpi("", v.comparators.unpaired.toFixed(2), "same twin with the pairing removed, same preparations") +
    kpi("", v.comparators.prior_art.toFixed(2), "Doorn et al. 2025 estimator, run unchanged on the same preparations") +
    kpi("miss", v.top1.treated, `NMDA as the single top mechanism (untreated: ${v.top1.null}). Co-primary: not met`);
  $("v3hash").textContent = `PREREGISTRATION_v3.md sha256 ${v.prereg_sha256} · frozen at commit ${v.freeze_commit} · ${v.ages}`;
  document.querySelectorAll("#v3method button").forEach(b => b.classList.toggle("on", b.dataset.m === state.method));
  document.querySelectorAll("#v3null button").forEach(b => b.classList.toggle("on", b.dataset.n === state.nullg));
  drawProfile(); drawStrip(); drawCanal(); drawDynasore();
  $("v3fail").innerHTML = `<b>What did not hold.</b> The probabilities are graded, not confident:
    no preparation in either group reached 0.5, so the twin ranks cultures rather than calling them.
    NMDA was the single top-ranked mechanism in ${v.top1.treated} treated and ${v.top1.null} untreated
    preparations. Against all ${v.pooled.n_null} untreated preparations, other genotypes included, the
    AUROC is ${v.pooled.auroc.toFixed(2)}.`;
}

function drawProfile() {
  const el = $("v3profile");
  const { T, N } = groups();
  if (state.method !== "twin") {
    const W = 520, H = 150; svg(el, W, H);
    const rows = [["Hodgkin's Razor", "twin"], ["pairing removed", "unpaired"], ["Doorn et al. 2025", "prior_art"]];
    const x0 = 150, x1 = W - 50, sx = a => x0 + (a - 0.3) / 0.7 * (x1 - x0);
    add(el, "line", { x1: sx(0.5), x2: sx(0.5), y1: 8, y2: H - 22, class: "axis", "stroke-dasharray": "3 3" });
    add(el, "line", { x1: sx(EV.v3.bar), x2: sx(EV.v3.bar), y1: 8, y2: H - 22, stroke: C.bar, "stroke-dasharray": "4 3" });
    add(el, "text", { x: sx(EV.v3.bar) + 4, y: 16 }, "bar 0.70");
    rows.forEach(([lab, m], i) => {
      const keep = state.method; state.method = m;
      const a = auroc(T.map(p => value(p, "g_nmda")), N.map(p => value(p, "g_nmda")));
      state.method = keep;
      const y = 26 + i * 38;
      add(el, "text", { x: x0 - 8, y: y + 14, "text-anchor": "end", class: m === state.method ? "sel" : "lab" }, lab);
      add(el, "rect", { x: sx(0.3), y, width: Math.max(0, sx(a) - sx(0.3)), height: 20, rx: 3, fill: m === "twin" ? C.treat : "#4a5565", opacity: m === state.method ? 1 : 0.6 });
      add(el, "text", { x: sx(a) + 6, y: y + 14, class: "lab" }, a.toFixed(2));
    });
    add(el, "text", { x: sx(0.5), y: H - 6, "text-anchor": "middle" }, "chance 0.5");
    $("profcap").textContent = "AUROC of the NMDA reading for each method, same preparations. The other two methods report the NMDA parameter only.";
    return;
  }
  const ms = EV.mechanisms.map(m => {
    const a = auroc(T.map(p => value(p, m.key)), N.map(p => value(p, m.key)));
    return { ...m, a };
  }).sort((x, y) => y.a - x.a);
  const W = 520, rowH = 26, H = ms.length * rowH + 34; svg(el, W, H);
  const x0 = 190, x1 = W - 46, sx = a => x0 + a * (x1 - x0);
  for (const g of [0, 0.25, 0.5, 0.75, 1]) {
    add(el, "line", { x1: sx(g), x2: sx(g), y1: 4, y2: H - 22, class: g === 0.5 ? "axis" : "grid", "stroke-dasharray": g === 0.5 ? "3 3" : "" });
    add(el, "text", { x: sx(g), y: H - 6, "text-anchor": "middle" }, g.toFixed(2));
  }
  ms.forEach((m, i) => {
    const y = 6 + i * rowH, on = m.key === state.mech;
    const g = add(el, "g", { class: "hit" });
    add(g, "rect", { x: 0, y: y - 2, width: W, height: rowH, fill: on ? "rgba(94,176,255,.08)" : "transparent" });
    add(g, "text", { x: x0 - 8, y: y + 14, "text-anchor": "end", class: on ? "sel" : "lab" }, m.label + (m.key === EV.v3.key ? "  (answer)" : ""));
    add(g, "rect", { x: x0, y: y + 3, width: Math.max(1, sx(m.a) - x0), height: 15, rx: 3, fill: m.key === EV.v3.key ? C.treat : "#4a5565" });
    add(g, "text", { x: sx(m.a) + 5, y: y + 15, class: "lab" }, m.a.toFixed(2));
    g.addEventListener("click", () => { state.mech = m.key; drawProfile(); drawStrip(); });
    hover(g, `${m.label} · ${m.target}<br>class: ${m.class}`);
  });
  $("profcap").textContent = "AUROC of each mechanism's probability, treated against untreated. The twin was never told which drug was applied. Click a row.";
}

function drawStrip() {
  const el = $("v3strip");
  const { T, N } = groups();
  const key = state.method === "twin" ? state.mech : "g_nmda";
  const tv = T.map(p => value(p, key)), nv = N.map(p => value(p, key));
  const all = tv.concat(nv), lo = Math.min(...all), hi = Math.max(...all), pad = (hi - lo) * 0.08 || 0.01;
  const W = 520, H = 300; svg(el, W, H);
  const y0 = H - 34, y1 = 14, sy = v => y0 - (v - (lo - pad)) / (hi - lo + 2 * pad) * (y0 - y1);
  for (let k = 0; k <= 4; k++) {
    const v = lo - pad + k * (hi - lo + 2 * pad) / 4;
    add(el, "line", { x1: 56, x2: W - 10, y1: sy(v), y2: sy(v), class: "grid" });
    add(el, "text", { x: 50, y: sy(v) + 4, "text-anchor": "end" }, v.toFixed(state.method === "prior_art" ? 2 : 3));
  }
  const cols = [[nv, N, W * 0.34, C.null, "untreated"], [tv, T, W * 0.74, C.treat, "kept on APV"]];
  for (const [vals, ps, cx, col, lab] of cols) {
    vals.forEach((v, i) => {
      const c = add(el, "circle", { cx: cx + (seeded(i + (col === C.treat ? 100 : 0)) - 0.5) * 90, cy: sy(v), r: 5, fill: col, "fill-opacity": 0.85 });
      hover(c, `${ps[i].id} · ${ps[i].genotype} · ${lab}<br>${v.toFixed(3)}`);
    });
    const m = median(vals);
    add(el, "line", { x1: cx - 58, x2: cx + 58, y1: sy(m), y2: sy(m), stroke: "#e6edf6", "stroke-width": 2.5 });
    add(el, "text", { x: cx, y: H - 12, "text-anchor": "middle", class: "lab" }, `${lab} (n = ${vals.length})`);
  }
  const a = auroc(tv, nv);
  const mlab = EV.mechanisms.find(m => m.key === key).label;
  const what = state.method === "twin" ? `p(${mlab} moved)` : state.method === "unpaired" ? `p(NMDA moved), pairing removed` : "Doorn et al. estimator, NMDA change";
  $("stripcap").textContent = `${what}, one dot per preparation. AUROC ${a.toFixed(2)}. Bars are medians.`;
}

function drawCanal() {
  const el = $("v3canal");
  const T = EV.v3.preps.filter(p => p.kind === "treated" && p.late !== null);
  const j = EV.mechanisms.findIndex(m => m.key === "g_nmda");
  const W = 520, H = 260; svg(el, W, H);
  const vals = T.flatMap(p => [p.p[j], p.late]), lo = Math.min(...vals) * 0.95, hi = Math.max(...vals) * 1.03;
  const y0 = H - 30, y1 = 12, sy = v => y0 - (v - lo) / (hi - lo) * (y0 - y1);
  const xa = 150, xb = 380;
  for (let k = 0; k <= 4; k++) {
    const v = lo + k * (hi - lo) / 4;
    add(el, "line", { x1: 60, x2: W - 20, y1: sy(v), y2: sy(v), class: "grid" });
    add(el, "text", { x: 54, y: sy(v) + 4, "text-anchor": "end" }, v.toFixed(3));
  }
  T.forEach(p => {
    const down = p.late < p.p[j];
    const l = add(el, "line", { x1: xa, x2: xb, y1: sy(p.p[j]), y2: sy(p.late), stroke: down ? C.treat : C.up, "stroke-opacity": 0.55, "stroke-width": 1.6 });
    hover(l, `${p.id} · ${p.genotype}<br>${p.p[j].toFixed(3)} → ${p.late.toFixed(3)}`);
    add(el, "circle", { cx: xa, cy: sy(p.p[j]), r: 3.5, fill: C.treat });
    add(el, "circle", { cx: xb, cy: sy(p.late), r: 3.5, fill: C.treat, "fill-opacity": 0.5 });
  });
  add(el, "text", { x: xa, y: H - 8, "text-anchor": "middle", class: "lab" }, "10 to 14 days");
  add(el, "text", { x: xb, y: H - 8, "text-anchor": "middle", class: "lab" }, "15 days and later");
  const nDown = T.filter(p => p.late < p.p[j]).length;
  $("v3canaltext").innerHTML = `Each line is one treated preparation: the twin's p(NMDA moved) early, then later
    in the same culture. It falls in ${nDown} of ${T.length} (Wilcoxon p = ${EV.v3.canalization.p.toExponential(0)}),
    and against untreated sisters the late AUROC is ${EV.v3.canalization.late_auroc.toFixed(2)}.<br><br>
    Charlesworth et al. describe exactly this: networks kept on APV compensate as they mature
    and converge on the untreated firing pattern. The pre-registration listed it as a secondary
    outcome before scoring.`;
}

function drawDynasore() {
  const el = $("dyn"), d = EV.doorn;
  const W = 520, rowH = 24, H = d.wells.length * rowH + 40; svg(el, W, H);
  const x0 = 90, x1 = W - 20, sx = v => x0 + v * (x1 - x0);
  for (const g of [0, 0.25, 0.5, 0.75, 1]) {
    add(el, "line", { x1: sx(g), x2: sx(g), y1: 4, y2: H - 26, class: g === 0.5 ? "" : "grid", stroke: g === 0.5 ? C.bar : "", "stroke-dasharray": g === 0.5 ? "4 3" : "" });
    add(el, "text", { x: sx(g), y: H - 10, "text-anchor": "middle" }, g === 0.5 ? "0.5: call" : g.toFixed(2));
  }
  d.wells.forEach((w, i) => {
    const y = 14 + i * rowH;
    add(el, "text", { x: x0 - 8, y: y + 4, "text-anchor": "end", class: "lab" }, w.well.replace("_", " "));
    add(el, "line", { x1: sx(w.null_max), x2: sx(w.treated_max), y1: y, y2: y, stroke: "#3a4554", "stroke-width": 3 });
    const a = add(el, "circle", { cx: sx(w.null_max), cy: y, r: 6, fill: C.null });
    hover(a, `${w.well}, pre-drug pair<br>top: ${w.null_top1} ${w.null_max.toFixed(2)} (no call)`);
    const b = add(el, "circle", { cx: sx(w.treated_max), cy: y, r: 6, fill: C.up, stroke: w.class_hit ? "#e6edf6" : "none", "stroke-width": 2.5 });
    hover(b, `${w.well}, after Dynasore<br>top: ${w.treated_top1} ${w.treated_max.toFixed(2)}<br>class: ${w.treated_class}${w.class_hit ? " (right class)" : ""}`);
  });
  const called = d.wells.filter(w => w.treated_called).length, ncalled = d.wells.filter(w => w.null_called).length;
  const cls = d.wells.filter(w => w.class_hit).length;
  $("dynkpis").innerHTML = kpi("good", `${called}/10`, `wells where a drug effect was called; ${ncalled}/10 on the same wells' pre-drug pairs`) +
    kpi("good", `${cls}/10`, `right mechanism class (${d.class}); white ring in the chart`) +
    kpi("miss", d.top1, "exact mechanism (u_rel or tau_d) on top; bar was 5/10: not met");
  $("dynfail").innerHTML = `<b>Why the exact mechanism missed.</b> The version 2 simulations assumed no drift between
    two recordings of one well. Measured on untreated pairs, the drift is 0.03 to 0.04 of each parameter's
    range. Version 3 adds a measured drift to every simulated pair.`;
}

// ---------- chips ----------
function drawChips() {
  const ch = EV.chips, props = Object.keys(ch.properties);
  let html = "<thead><tr><th>Readout</th>" + props.map(k => `<th style="text-align:center">${ch.properties[k]}</th>`).join("") + "</tr></thead><tbody>";
  for (const row of ch.matrix) {
    html += `<tr><td>${row.readout}</td>` + props.map(k => {
      const r = row.r[k], a = Math.max(0, Math.min(1, r / 0.75));
      return `<td style="background:rgba(94,176,255,${(a * 0.55).toFixed(2)});color:${a > 0.6 ? "#fff" : "var(--ink)"}">${r.toFixed(2)}</td>`;
    }).join("") + "</tr>";
  }
  $("heat").innerHTML = html + "</tbody>";
  const wrap = $("heat").parentElement;
  if (!wrap.classList.contains("scrollx")) {
    const d = document.createElement("div"); d.className = "scrollx";
    wrap.insertBefore(d, $("heat")); d.appendChild($("heat"));
  }
  const dsel = ch.matrix.map(r => [r.readout, r.r.direction_sel]).sort((a, b) => b[1] - a[1]);
  $("heatnote").innerHTML = `Channel directionality is invisible to chamber electrodes and calcium imaging
    (r ${ch.matrix[0].r.direction_sel.toFixed(2)} and ${ch.matrix[1].r.direction_sel.toFixed(2)}): it needs electrodes in the channels
    (best: ${dsel[0][0]}, r ${dsel[0][1].toFixed(2)}). The twin says this before a chip is built.`;
  const slider = $("chipslider"), curve = ch.power.twin;
  slider.max = curve.length - 1;
  slider.value = Math.min(state.chips, curve.length - 1);
  slider.oninput = () => { state.chips = +slider.value; drawPower(); };
  drawPower();
}

function drawPower() {
  const ch = EV.chips, curve = ch.power.twin, cur = curve[+$("chipslider").value];
  $("nchips").textContent = cur.n;
  const el = $("power"), W = 520, H = 240; svg(el, W, H);
  const x0 = 46, x1 = W - 16, y0 = H - 30, y1 = 12;
  const nmax = curve[curve.length - 1].n, sx = n => x0 + (n / nmax) * (x1 - x0), sy = p => y0 - p * (y0 - y1);
  for (const g of [0, 0.2, 0.4, 0.6, 0.8, 1]) {
    add(el, "line", { x1: x0, x2: x1, y1: sy(g), y2: sy(g), class: "grid" });
    add(el, "text", { x: x0 - 6, y: sy(g) + 4, "text-anchor": "end" }, g.toFixed(1));
  }
  add(el, "line", { x1: x0, x2: x1, y1: sy(ch.power.target), y2: sy(ch.power.target), stroke: C.bar, "stroke-dasharray": "4 3" });
  add(el, "text", { x: x1, y: sy(ch.power.target) - 5, "text-anchor": "end" }, "80% power");
  add(el, "path", { d: curve.map((c, i) => `${i ? "L" : "M"}${sx(c.n)},${sy(c.power)}`).join(""), fill: "none", stroke: C.treat, "stroke-width": 2.5 });
  curve.forEach(c => add(el, "circle", { cx: sx(c.n), cy: sy(c.power), r: c === cur ? 6 : 3, fill: c === cur ? "#fff" : C.treat }));
  const rec = curve.find(c => c.n === 8);
  if (rec) {
    add(el, "line", { x1: sx(8), x2: sx(8), y1: y1, y2: y0, stroke: C.up, "stroke-dasharray": "3 3" });
    add(el, "text", { x: sx(8) + 4, y: y1 + 10 }, "public test");
  }
  for (let n = 20; n <= nmax; n += 20) add(el, "text", { x: sx(n), y: H - 10, "text-anchor": "middle" }, n);
  add(el, "text", { x: x1, y: H - 0, "text-anchor": "end" }, "chips per design");
  const ok = cur.power >= ch.power.target;
  $("powkpis").innerHTML = kpi(ok ? "good" : "miss", `${Math.round(cur.power * 100)}%`, `chance that the channel statistic's 95% interval clears 0.5 with ${cur.n} chips per design`) +
    kpi("", `${cur.lo.toFixed(2)} to ${cur.hi.toFixed(2)}`, "90% range of the AUROC such an experiment would report");
  $("powtext").innerHTML = `The one public recorded test of chip directionality (Mateus et al. 2024) scored
    ${ch.power.recorded.chips} chips across its designs and got AUROC ${ch.power.recorded.auroc} (95% CI ${ch.power.recorded.ci95[0]} to ${ch.power.recorded.ci95[1]}):
    inconclusive. The twin puts that experiment's power at ${ch.power.power_at_recorded.toFixed(2)} and says
    about ${ch.power.needed_twin} chips per design are needed for 80%.`;
}

// ---------- wiring ----------
document.querySelectorAll("#v3method button").forEach(b => b.addEventListener("click", () => { state.method = b.dataset.m; drawBlind(); }));
document.querySelectorAll("#v3null button").forEach(b => b.addEventListener("click", () => { state.nullg = b.dataset.n; drawBlind(); }));

(async () => {
  try {
    EV = await (await fetch(new URL("evidence.json", import.meta.url))).json();
  } catch (e) {
    for (const id of ["v3kpis", "heat"]) { const el = $(id); if (el) el.innerHTML = `<p class="err">evidence.json missing: run scripts/export_evidence.py</p>`; }
  }
  route();
})();
