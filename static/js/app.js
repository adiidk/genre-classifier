(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const MAX_BYTES = Number(document.body.dataset.maxMb || 50) * 1024 * 1024;
  const ALLOWED = [".wav", ".mp3"];
  const KNOWN_GENRES = ["blues", "classical", "country", "disco", "hiphop",
                        "jazz", "metal", "pop", "reggae", "rock"];

  const state = { file: null, extraction: null, busy: false, previewUrl: null };

  // ------------------------------------------------------------ helpers
  function el(tag, cls, text) {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text !== undefined) n.textContent = text;
    return n;
  }
  const cap = (s) => s.charAt(0).toUpperCase() + s.slice(1);
  const pct = (x) => (x * 100).toFixed(1) + "%";

  function genreColor(g) {
    let i = KNOWN_GENRES.indexOf(g);
    if (i < 0) i = [...g].reduce((a, c) => a + c.charCodeAt(0), 0) % 10;
    return `hsl(${(i * 36 + 200) % 360}, 70%, 60%)`;
  }
  function fmt(n) {
    if (!Number.isFinite(n)) return "–";
    const a = Math.abs(n);
    if (a !== 0 && (a < 0.001 || a >= 100000)) return n.toExponential(3);
    return n.toFixed(a >= 100 ? 1 : 4);
  }

  async function api(url, options) {
    let res;
    try { res = await fetch(url, options); }
    catch { throw new Error("Could not reach the server. Is app.py still running?"); }
    let data = null;
    try { data = await res.json(); } catch { /* not JSON */ }
    if (!res.ok) throw new Error(data?.error?.message || `Request failed (${res.status}).`);
    if (data === null) throw new Error("The server sent an unreadable response.");
    return data;
  }

  // ------------------------------------------------------------ UI state
  function showError(msg) { const e = $("error"); e.textContent = msg; e.hidden = false; }
  function clearError() { $("error").hidden = true; }
  function setStatus(msg) {
    $("status").hidden = !msg;
    $("status-text").textContent = msg || "";
  }
  function hideResults() { $("results").hidden = true; }
  function hasModels() { return $("model-select").options.length > 0; }
  function updateControls() {
    $("analyze-btn").disabled = state.busy || !state.file || !hasModels();
    $("model-select").disabled = state.busy || !hasModels();
  }
  function setBusy(b) { state.busy = b; updateControls(); }

  function showIssues(issues) {
    const box = $("banners");
    box.replaceChildren();
    for (const i of issues || []) {
      box.append(el("div", "banner " + (i.level === "error" ? "banner-error" : "banner-warn"), i.message));
    }
  }

  // ------------------------------------------------------------ models
  function describeModel(m) {
    const cv = m.cv_metrics || {};
    if (cv.song_accuracy == null) return "";
    return `Cross-validated accuracy: ${pct(cv.song_accuracy)} per song, ` +
           `${pct(cv.segment_accuracy)} per 3-second segment.`;
  }
  async function loadModels() {
    const select = $("model-select");
    try {
      const data = await api("/api/models");
      showIssues(data.issues);
      select.replaceChildren();
      for (const m of data.models) {
        const o = el("option", "", m.name);
        o.value = m.key;
        o.dataset.info = describeModel(m);
        select.append(o);
      }
      $("model-info").textContent = select.selectedOptions[0]?.dataset.info || "";
    } catch (e) {
      showIssues([{ level: "error", message: e.message }]);
    }
    updateControls();
  }

  // ------------------------------------------------------------ file input
  function setFile(file) {
    clearError();
    if (!file) return;
    const dot = file.name.lastIndexOf(".");
    const ext = dot >= 0 ? file.name.slice(dot).toLowerCase() : "";
    if (!ALLOWED.includes(ext)) {
      return showError(`Unsupported file type (${ext || "unknown"}). Please choose a .wav or .mp3 file.`);
    }
    if (file.size === 0) return showError("That file is empty.");
    if (file.size > MAX_BYTES) {
      return showError(`The file is too large. The maximum size is ${MAX_BYTES / 1048576} MB.`);
    }
    state.file = file;
    state.extraction = null;
    hideResults();
    $("file-name").textContent = `${file.name} · ${(file.size / 1048576).toFixed(1)} MB`;
    if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
    state.previewUrl = URL.createObjectURL(file);
    const p = $("player");
    p.src = state.previewUrl;
    p.hidden = false;
    updateControls();
  }

  // ------------------------------------------------------------ rendering
  function renderPrediction(r) {
    $("genre").textContent = cap(r.predicted_genre);
    $("genre").style.color = genreColor(r.predicted_genre);
    $("confidence").textContent = pct(r.confidence);
    $("model-used").textContent = r.model.name;

    const ranking = $("ranking");
    ranking.replaceChildren();
    for (const item of r.ranking) {
      const row = el("div", "rank-row");
      const bar = el("div", "bar");
      const fill = el("span");
      fill.style.width = (item.probability * 100).toFixed(1) + "%";
      fill.style.background = genreColor(item.genre);
      bar.append(fill);
      row.append(el("span", "", cap(item.genre)), bar, el("span", "val", pct(item.probability)));
      ranking.append(row);
    }

    const secs = state.extraction?.analysis?.segment_seconds || 3;
    const segs = $("segments");
    segs.replaceChildren();
    for (const s of r.segment_predictions) {
      const chip = el("div", "chip");
      chip.style.borderLeftColor = genreColor(s.genre);
      chip.append(el("b", "", cap(s.genre)),
                  el("span", "", `${s.index * secs}–${(s.index + 1) * secs} s · ${pct(s.confidence)}`));
      segs.append(chip);
    }
    const votes = Object.entries(r.votes).map(([g, n]) => `${cap(g)} ${n}`).join(" · ");
    $("votes").textContent = `Segment votes: ${votes}`;
    $("results").hidden = false;
  }

  function mfccChart(values) {
    const NS = "http://www.w3.org/2000/svg";
    const W = 640, H = 180, pad = 20, y0 = (H - 18) / 2, maxH = y0 - 6;
    const svg = document.createElementNS(NS, "svg");
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    svg.setAttribute("class", "mfcc-chart");
    svg.setAttribute("role", "img");
    svg.setAttribute("aria-label", "Mean value of each MFCC coefficient");
    const maxAbs = Math.max(...values.map(Math.abs), 1e-9);
    const slot = (W - pad * 2) / values.length;

    const base = document.createElementNS(NS, "line");
    base.setAttribute("x1", pad); base.setAttribute("x2", W - pad);
    base.setAttribute("y1", y0);  base.setAttribute("y2", y0);
    base.setAttribute("stroke", "#2a3350");
    svg.append(base);

    values.forEach((v, i) => {
      const h = (Math.abs(v) / maxAbs) * maxH;
      const rect = document.createElementNS(NS, "rect");
      rect.setAttribute("x", pad + i * slot + slot * 0.15);
      rect.setAttribute("width", slot * 0.7);
      rect.setAttribute("y", v >= 0 ? y0 - h : y0);
      rect.setAttribute("height", Math.max(h, 0.5));
      rect.setAttribute("rx", 2);
      rect.setAttribute("fill", v >= 0 ? "#7c5cff" : "#22d3ee");
      const title = document.createElementNS(NS, "title");
      title.textContent = `MFCC ${i + 1}: ${v.toFixed(2)}`;
      rect.append(title);
      svg.append(rect);

      const t = document.createElementNS(NS, "text");
      t.setAttribute("x", pad + i * slot + slot / 2);
      t.setAttribute("y", H - 4);
      t.textContent = i + 1;
      svg.append(t);
    });
    return svg;
  }

  function renderExtraction(ex, info) {
    const img = $("spectrogram");
    if (ex.spectrogram) { img.src = "data:image/png;base64," + ex.spectrogram; img.hidden = false; }
    else { img.hidden = true; }

    const a = ex.analysis;
    const facts = [
      ["Sample rate", `${a.sample_rate} Hz`],
      ["Audio analysed", `${a.duration_analyzed} s`],
      ["Segments", `${a.n_segments} × ${a.segment_seconds} s (${a.samples_per_segment} samples each)`],
      ["Features per segment", a.n_features],
      ["Tempo (average)", `${ex.summary.tempo.toFixed(1)} BPM`],
    ];
    const dl = $("analysis");
    dl.replaceChildren();
    for (const [k, v] of facts) dl.append(el("dt", "", k), el("dd", "", String(v)));

    const tbody = $("scalar-rows");
    tbody.replaceChildren();
    for (const s of ex.summary.scalars) {
      const tr = el("tr");
      const name = el("td", "", s.label);
      name.append(el("small", "", s.description));
      tr.append(name, el("td", "num", fmt(s.mean)), el("td", "num", fmt(s.var)));
      tbody.append(tr);
    }
    $("mfcc").replaceChildren(mfccChart(ex.summary.mfcc_mean));
    $("truncated").hidden = !info?.will_be_truncated;
  }

  // ------------------------------------------------------------ actions
  function formWith(file) {
    const fd = new FormData();
    fd.append("file", file);
    return fd;
  }

  async function predict() {
    const result = await api("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: $("model-select").value, segments: state.extraction.segments }),
    });
    renderPrediction(result);
  }

  async function analyze() {
    if (!state.file || state.busy) return;
    clearError();
    hideResults();
    setBusy(true);
    try {
      setStatus("Checking the file…");
      const info = await api("/api/inspect", { method: "POST", body: formWith(state.file) });

      setStatus("Extracting features (STFT, MFCC, HPSS, tempo…). This can take 10–30 seconds.");
      state.extraction = await api("/api/extract", { method: "POST", body: formWith(state.file) });
      renderExtraction(state.extraction, info);

      setStatus("Predicting…");
      await predict();
    } catch (e) {
      state.extraction = null;
      hideResults();
      showError(e.message);
    } finally {
      setStatus("");
      setBusy(false);
    }
  }

  // Switching model after analysis re-predicts instantly, with no re-extraction.
  async function onModelChange() {
    $("model-info").textContent = $("model-select").selectedOptions[0]?.dataset.info || "";
    if (!state.extraction || state.busy) return;
    clearError();
    setBusy(true);
    try { await predict(); }
    catch (e) { showError(e.message); }
    finally { setBusy(false); }
  }

  // ------------------------------------------------------------ wiring
  const dz = $("dropzone");
  $("file-input").addEventListener("change", (e) => {
    const f = e.target.files[0];
    e.target.value = "";           // allows choosing the same file again
    setFile(f);
  });
  dz.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); $("file-input").click(); }
  });
  ["dragenter", "dragover"].forEach((t) =>
    dz.addEventListener(t, (e) => { e.preventDefault(); dz.classList.add("drag"); }));
  ["dragleave", "drop"].forEach((t) =>
    dz.addEventListener(t, (e) => { e.preventDefault(); dz.classList.remove("drag"); }));
  dz.addEventListener("drop", (e) => setFile(e.dataTransfer.files[0]));

  $("analyze-btn").addEventListener("click", analyze);
  $("model-select").addEventListener("change", onModelChange);

  loadModels();
})();