const $ = (id) => document.getElementById(id);
const status = (t, err) => { const s = $("status"); s.textContent = t; s.className = err ? "sub err" : "sub"; };

let gpu = false;

async function loadExamples() {
  const r = await fetch("data/index.json");
  const d = await r.json();
  gpu = d.gpu;
  $("mode").textContent = gpu
    ? "simulation on, the twin is re-run for every analysis"
    : "cached analyses, no simulation on this host";
  const sel = $("example");
  sel.innerHTML = "";
  for (const e of d.examples) {
    const o = document.createElement("option");
    o.value = e.id;
    o.textContent = `${e.label}`;
    sel.appendChild(o);
  }
  if (!d.examples.length) status("No bundled examples found. Run scripts/make_examples.py.", true);
}

function drawRaster(canvas, data, duration, nElec, colour) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  canvas.width = w * dpr; canvas.height = h * dpr;
  const g = canvas.getContext("2d");
  g.scale(dpr, dpr);
  g.clearRect(0, 0, w, h);
  g.fillStyle = "#0b0e13"; g.fillRect(0, 0, w, h);
  if (!data || !data.e || !data.e.length) {
    g.fillStyle = "#8b98a8"; g.font = "12px sans-serif";
    g.fillText("no events", 10, h / 2);
    return;
  }
  const pad = 6, rowH = (h - 2 * pad) / nElec;
  g.fillStyle = colour;
  for (let i = 0; i < data.e.length; i++) {
    const x = pad + (data.t[i] / duration) * (w - 2 * pad);
    const y = pad + data.e[i] * rowH;
    g.fillRect(x, y, 1.1, Math.max(rowH - 1.2, 1));
  }
  g.fillStyle = "#8b98a8"; g.font = "11px ui-monospace,monospace";
  g.fillText(`${data.n} events`, w - 78, 13);
}

const UNIT = {g_tonic_inh: " nS", i_drive: " pA"};
function fmtEffect(m) {
  if (m.kind === "fold") return `${m.effect.toFixed(2)}x`;
  return (m.effect >= 0 ? "+" : "") + m.effect.toFixed(2) + (UNIT[m.key] || "");
}
function fmtInterval(m) {
  if (m.kind === "fold") return `${m.lo.toFixed(2)}x to ${m.hi.toFixed(2)}x`;
  const u = UNIT[m.key] || "";
  return `${m.lo.toFixed(2)} to ${m.hi.toFixed(2)}${u}`;
}
function guardText(g) {
  if (!g) return "Guard not run on this host.";
  const parts = [];
  if (g.typicality !== undefined)
    parts.push(`typicality ${g.typicality.toFixed(1)} (threshold ${g.typicality_threshold.toFixed(1)})`);
  if (g.discrepancy !== undefined)
    parts.push(`predictive check ${g.discrepancy.toFixed(0)} statistics outside (threshold ${g.threshold.toFixed(0)})`);
  else parts.push("predictive check needs a CUDA device and was not run here");
  return "Guard: " + parts.join("; ") + ".";
}

function render(d) {
  const rep = d.report;
  $("out").hidden = false;
  let badge = '<span class="badge none">no call</span>';
  if (rep.verdict === "outside_model") badge = '<span class="badge warn">outside the model</span>';
  else if (rep.verdict === "mechanism_called") badge = '<span class="badge ok">mechanism named</span>';
  $("verdict").innerHTML = badge + rep.sentence;
  const sys_ = (rep.meta && rep.meta.recording_system) ? ` Read as ${rep.meta.recording_system}.` : "";
  $("guardline").textContent = guardText(rep.guard) + sys_;

  const tb = $("mech").querySelector("tbody");
  tb.innerHTML = "";
  for (const m of rep.mechanisms) {
    const tr = document.createElement("tr");
    if (m.p_active >= 0.5) tr.className = "called";
    tr.innerHTML = `<td>${m.label}</td><td class="sub">${m.target || "-"}</td>
      <td class="num">${m.p_active.toFixed(2)}</td>
      <td><div class="bar"><i style="width:${(m.p_active * 100).toFixed(0)}%"></i></div></td>
      <td class="num">${fmtEffect(m)}</td><td class="num">${fmtInterval(m)}</td>`;
    tb.appendChild(tr);
  }
  $("mechpanel").hidden = false;

  const kb = $("klass").querySelector("tbody");
  kb.innerHTML = "";
  for (const c of (rep.classes || [])) {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${c.name}</td><td class="num">${c.probability.toFixed(2)}</td>
      <td><div class="bar"><i style="width:${(c.probability * 100).toFixed(0)}%"></i></div></td>`;
    kb.appendChild(tr);
  }
  $("classpanel").hidden = !(rep.classes && rep.classes.length);

  const dur = rep.meta.duration_s || 60;
  const ne = (rep.meta.recording_system === "grid12") ? 12 : 16;
  $("rasterpanel").hidden = false;
  drawRaster($("c_ob"), d.observed.base, dur, ne, "#5eb0ff");
  drawRaster($("c_ot"), d.observed.treat, dur, ne, "#f0883e");
  if (d.twin) {
    drawRaster($("c_tb"), d.twin.base, dur, ne, "#7fc7ff");
    drawRaster($("c_tt"), d.twin.treat, dur, ne, "#ffb27f");
  } else {
    for (const id of ["c_tb", "c_tt"]) drawRaster($(id), null, dur, ne, "#888");
  }

  const cb = $("culture").querySelector("tbody");
  cb.innerHTML = "";
  for (const c of rep.culture) {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${c.label}</td><td class="num">${c.median.toPrecision(3)}</td>
      <td class="num">${c.lo.toPrecision(3)} to ${c.hi.toPrecision(3)}</td>
      <td class="sub">${c.unit || "-"}</td>`;
    cb.appendChild(tr);
  }
  $("culturepanel").hidden = false;

  const fb = $("feat").querySelector("tbody");
  fb.innerHTML = "";
  d.features.names.forEach((n, i) => {
    const a = d.features.base[i], b = d.features.treat[i];
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${n}</td><td class="num">${a.toPrecision(3)}</td>
      <td class="num">${b.toPrecision(3)}</td>
      <td class="num">${(b - a >= 0 ? "+" : "") + (b - a).toPrecision(3)}</td>`;
    fb.appendChild(tr);
  });
  $("md").textContent = d.markdown;
  $("featpanel").hidden = false;
  status("");
}

async function runExample() {
  const id = $("example").value;
  if (!id) return;
  $("run").disabled = true;
  status("analysing" + (gpu ? ", simulating the twin" : "") + " ...");
  try {
    const r = await fetch(`data/${id}.json`);
    if (!r.ok) throw new Error(await r.text());
    render(await r.json());
  } catch (e) { status(String(e), true); }
  $("run").disabled = false;
}

async function runUpload() {
  const fb = null, ft = null;
  if (!fb || !ft) { status("choose both files", true); return; }
  const fd = new FormData();
  fd.append("baseline", fb); fd.append("treated", ft);
  fd.append("duration", $("dur").value);
  $("upload").disabled = true;
  status("analysing upload ...");
  try {
    const r = await fetch("/api/analyse", { method: "POST", body: fd });
    if (!r.ok) throw new Error((await r.json()).detail || "failed");
    render(await r.json());
  } catch (e) { status(String(e), true); }
  $("upload").disabled = false;
}

$("run").addEventListener("click", runExample);

window.addEventListener("resize", () => { /* redraw happens on next analyse */ });
loadExamples();
