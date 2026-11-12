// Navigatie en interactieve onderdelen van de presentatie. Gewone JavaScript, geen bibliotheken.
// Cijfers: window.DATA (build_data.py) en window.GALLERY (make_gallery.py).
(function () {
  "use strict";
  const D = window.DATA, G = window.GALLERY;
  const $ = (id) => document.getElementById(id);
  const nl = (x, digits) => Number(x).toFixed(digits).replace(".", ",");
  const ENGINES = ["yolo26s_fp32", "yolo26s_fp16", "yolo26s_int8", "rtdetr-l_fp32", "rtdetr-l_fp16", "rtdetr-l_int8"];
  const NAME = { "yolo26s": "YOLO26s", "rtdetr-l": "RT-DETR-l" };
  const engineLabel = (k) => { const [m, p] = k.split("_"); return `${NAME[m]} ${p.toUpperCase()}`; };
  const PREC_COLOR = { fp32: "var(--fp32)", fp16: "var(--fp16)", int8: "var(--int8)" };
  const REQUIREMENT = 20;

  // ---------------------------------------------------------------- navigatie
  const chapters = [...document.querySelectorAll(".ch")];
  const progress = $("progress");
  chapters.forEach((ch, i) => {
    const b = document.createElement("button");
    b.setAttribute("aria-label", `Hoofdstuk ${i + 1}: ${ch.dataset.name}`);
    b.title = `${i + 1} · ${ch.dataset.name}`;
    b.addEventListener("click", () => go(i));
    progress.appendChild(b);
  });
  let current = 0;

  function go(i) {
    current = Math.max(0, Math.min(chapters.length - 1, i));
    chapters.forEach((ch, j) => ch.classList.toggle("active", j === current));
    [...progress.children].forEach((b, j) => b.classList.toggle("on", j <= current));
    const ch = chapters[current];
    $("hud-chapter").textContent = `${String(current + 1).padStart(2, "0")} / ${chapters.length}`;
    $("hud-alt").textContent = nl(ch.dataset.alt, 1);
    $("hud-name").textContent = ch.dataset.name;
    document.body.classList.toggle("is-demo", ch.hasAttribute("data-demo"));
    const why = ch.querySelector(".why-src");
    $("why").innerHTML = why ? why.innerHTML : "<p>Geen extra uitleg bij dit hoofdstuk.</p>";
    const notes = ch.querySelector(".notes-src");
    $("notes-text").textContent = notes ? notes.content.textContent : "";
    try { history.replaceState(null, "", `#${current + 1}`); } catch (e) { /* file:// kan dit weigeren */ }
  }

  function toggle(cls) { document.body.classList.toggle(cls); }
  function toggleWhy(force) {
    const open = force === undefined ? !$("why").classList.contains("open") : force;
    $("why").classList.toggle("open", open);
    $("why").setAttribute("aria-hidden", String(!open));
  }
  function toggleVideo(force) {
    const show = force === undefined ? !document.body.classList.contains("show-video") : force;
    document.body.classList.toggle("show-video", show);
    const v = $("video");
    if (show && v.tagName === "VIDEO") { v.currentTime = 0; v.play().catch(() => {}); } else if (v.pause) { v.pause(); }
  }

  document.addEventListener("keydown", (e) => {
    if (e.target.tagName === "INPUT" && (e.key === "ArrowLeft" || e.key === "ArrowRight")) return;  // schuifjes
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    const k = e.key.toLowerCase();
    if (k === "arrowright" || k === "pagedown" || k === " ") { e.preventDefault(); go(current + 1); }
    else if (k === "arrowleft" || k === "pageup") { e.preventDefault(); go(current - 1); }
    else if (k === "home") go(0);
    else if (k === "end") go(chapters.length - 1);
    else if (/^[1-9]$/.test(k)) go(Number(k) - 1);
    else if (k === "0") go(9);
    else if (k === "w") toggleWhy();
    else if (k === "n") toggle("show-notes");
    else if (k === "b") toggle("show-appendix");
    else if (k === "v") toggleVideo();
    else if (k === "t") startTime = Date.now();
    else if (k === "l") {
      const root = document.documentElement;
      root.dataset.theme = root.dataset.theme === "light" ? "dark" : "light";
      try { localStorage.setItem("theme", root.dataset.theme); } catch (err) { /* geen opslag */ }
    }
    else if (k === "escape") { toggleWhy(false); toggleVideo(false); document.body.classList.remove("show-appendix", "show-notes"); }
  });
  $("prev").addEventListener("click", () => go(current - 1));
  $("next").addEventListener("click", () => go(current + 1));
  try { const t = localStorage.getItem("theme"); if (t) document.documentElement.dataset.theme = t; } catch (e) { /* geen opslag */ }

  // timer bij de notities: amber vanaf 8:00 (seintje jury), rood vanaf 10:00
  let startTime = Date.now();
  setInterval(() => {
    const s = Math.floor((Date.now() - startTime) / 1000);
    const el = $("timer");
    el.textContent = `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;
    el.classList.toggle("late", s >= 600);
  }, 500);

  // ontbrekende beelden of video: duidelijk vak in plaats van een gebroken icoon
  function missing(el, text) {
    const div = document.createElement("div");
    div.className = "missing";
    div.textContent = `Nog toe te voegen: ${text}`;
    el.replaceWith(div);
  }
  document.querySelectorAll("img[data-optional]").forEach((img) => {
    const fail = () => missing(img, img.dataset.optional);
    if (img.complete && img.naturalWidth === 0) fail(); else img.addEventListener("error", fail);
  });
  $("video").addEventListener("error", () => missing($("video"), "demovideo (presentation/video/demo.mp4)"));

  // ---------------------------------------------------------------- hulpfuncties
  function barRow(label, value, max, color, text) {
    return `<div class="bar-row"><span class="lbl">${label}</span><div class="track"><div class="fill" style="width:${Math.min(100, value / max * 100)}%;background:${color}"></div></div><span class="val">${text}</span></div>`;
  }
  function chips(container, items, onPick, initial) {
    container.innerHTML = "";
    items.forEach(([key, label]) => {
      const b = document.createElement("button");
      b.className = "chip"; b.dataset.key = key; b.textContent = label;
      b.addEventListener("click", () => { mark(key); onPick(key); });
      container.appendChild(b);
    });
    function mark(key) { [...container.children].forEach((b) => b.classList.toggle("on", b.dataset.key === key)); }
    mark(initial); onPick(initial);
  }

  // ---------------------------------------------------------------- 1: openingsbeeld (5 m, met label)
  (function hero() {
    const e = D.distance_examples[0], [x1, y1, x2, y2] = e.box, s = 100 / 640;
    $("hero").innerHTML = `<img src="${e.img}" alt="Camerabeeld van de drone op ${e.distance} m: het landingsplatform met de H" style="max-height:66vh">
      <div class="bb" style="left:${x1 * s}%;top:${y1 * s}%;width:${(x2 - x1) * s}%;height:${(y2 - y1) * s}%"><span>platform</span></div>
      <span class="tagline">ALT ${e.distance},0 m · θ 0° · ${e.width_px} px</span>`;
  })();

  // ---------------------------------------------------------------- 4: afstanden
  $("distance-row").innerHTML = D.distance_examples.map((e) => {
    const [x1, y1, x2, y2] = e.box, s = 100 / 640;
    return `<figure class="panel bare" style="margin:0"><div class="boxed"><img src="${e.img}" alt="Testbeeld op ${e.distance} m">
      <div class="bb" style="left:${x1 * s}%;top:${y1 * s}%;width:${(x2 - x1) * s}%;height:${(y2 - y1) * s}%"></div>
      <span class="tagline">${e.distance} m · ${e.width_px} px</span></div></figure>`;
  }).join("");

  // ---------------------------------------------------------------- 6: bits en inferentie
  (function bits() {
    const rows = [["FP32", 32, "var(--fp32)", "0,785398185"], ["FP16", 16, "var(--fp16)", "0,785156250"], ["INT8", 8, "var(--int8)", "≈ 100 van 256 stappen"]];
    $("bits").innerHTML = `<div class="cap" style="margin:0 0 1vh">HETZELFDE GEWICHT (π/4) IN DRIE REKENPRECISIES</div>` + rows.map(([n, b, c, v]) =>
      `<div style="display:grid;grid-template-columns:4em minmax(0,1fr) 12em;gap:0.8em;align-items:center;margin:0.5vh 0;font-family:var(--mono);font-size:0.7em">
        <span>${n}</span><div style="display:flex;gap:2px">${Array.from({ length: 32 }, (_, i) => `<i style="flex:1;height:1.2em;background:${i < b ? c : "var(--bg)"};border:1px solid var(--line)"></i>`).join("")}</div>
        <span style="color:var(--muted)">${v}</span></div>`).join("");
    const max = Math.max(...ENGINES.map((k) => D.engines[k].inference_ms));
    $("inference-bars").innerHTML = ENGINES.map((k) => barRow(engineLabel(k), D.engines[k].inference_ms, max, PREC_COLOR[k.split("_")[1]], `${nl(D.engines[k].inference_ms, 1)} ms`)).join("");
  })();

  // ---------------------------------------------------------------- 7: FPS-rekenaar
  function updateRequirement() {
    const n = +$("in-n").value, v = +$("in-v").value, s = +$("in-s").value;
    const req = n * v / s;
    $("out-n").textContent = n;
    $("out-v").textContent = `${nl(v, 1)} m/s`;
    $("out-s").textContent = `${nl(s, 2)} m`;
    $("out-fps").textContent = nl(req, req % 1 ? 1 : 0);
    const max = Math.max(req, 25) * 1.05;
    $("req-bars").innerHTML = ENGINES.map((k) => {
      const fps = D.engines[k].chain_fps_mean, ok = fps >= req;
      return barRow(engineLabel(k), fps, max, ok ? "var(--hud)" : "var(--amber)", `${nl(fps, 1)} ${ok ? "✓" : "✗"}`);
    }).join("");
  }
  ["in-n", "in-v", "in-s"].forEach((id) => $(id).addEventListener("input", updateRequirement));
  updateRequirement();

  // ---------------------------------------------------------------- 8: keten-tijdlijn
  const PARTS = [["capture", "beeld ophalen · pc (AirSim)", "#56656b", "#e6f2ec"], ["rest_pc", "JPEG + beslissing · pc", "#3b4a50", "#e6f2ec"],
    ["network", "netwerk · USB-C", "var(--fg)", "var(--bg)"], ["server_rest", "JPEG + voor/nabewerking · Jetson", "#2f7d5c", "#e6f2ec"],
    ["inference", "inferentie · TensorRT · Jetson", "#5fd6a0", "#0d1417"]];
  $("chain-legend").innerHTML = PARTS.map(([, l, c]) => `<span><i style="background:${c}"></i>${l}</span>`).join("");
  function showChain(k) {
    const c = D.chain[k], MAX = 110;
    $("chain-timeline").innerHTML = PARTS.map(([key, l, bg, fg]) =>
      `<div title="${l}: ${nl(c[key], 1)} ms" style="width:${c[key] / MAX * 100}%;background:${bg};color:${fg}">${c[key] >= 6 ? nl(c[key], 0) + " ms" : ""}</div>`).join("");
    const p = 50 / MAX * 100;
    $("chain-limit").style.left = `${p}%`; $("chain-limit-lbl").style.left = `${p}%`;
    $("chain-total").textContent = `${nl(c.frame, 1)} ms per frame`;
    const fps = D.engines[k].chain_fps_mean, ok = fps >= REQUIREMENT;
    $("chain-fps").textContent = nl(fps, 1);
    $("chain-fps").style.color = ok ? "var(--hud)" : "var(--amber)";
    $("chain-badge").textContent = ok ? "EIS GEHAALD" : "TE TRAAG";
    $("chain-badge").className = `badge ${ok ? "b-hud" : "b-amber"}`;
  }
  chips($("chain-chips"), ENGINES.map((k) => [k, engineLabel(k)]), showChain, "yolo26s_fp16");

  // ---------------------------------------------------------------- 11: uitkomsten
  $("outcome-bars").innerHTML = ENGINES.map((k) => {
    const e = D.engines[k], n = e.live_attempts;
    const seg = (v, c) => v ? `<div style="width:${v / n * 100}%;background:${c};height:100%"></div>` : "";
    return `<div class="bar-row"><span class="lbl">${engineLabel(k)}</span><div class="track" style="display:flex">${seg(e.goede_landing, "var(--hud)")}${seg(e.terecht_afgebroken, "#2f7d5c")}${seg(e.foute_landing + e.slechte_landing + e.onterecht_afgebroken, "var(--red)")}</div><span class="val">${e.goede_landing + e.terecht_afgebroken}/${n}</span></div>
      <div class="mono small" style="margin:-0.6vh 0 0 calc(9em * 0.7 + 0.8em);font-size:0.58em">95 %-interval ≥ ${nl(e.live_correct_ci95_low_pct, 1)} % · live ${nl(e.live_fps_mean, 1)} FPS</div>`;
  }).join("");

  // ---------------------------------------------------------------- 12: galerij slecht zicht
  const CONDS = [["clean", "geen"], ["fog_05", "mist 0,5"], ["fog_10", "mist 1,0"], ["dust_05", "stof 0,5"], ["dust_10", "stof 1,0"], ["dark", "weinig licht"], ["blur", "onscherp"], ["noise", "ruis"]];
  const COND_TEXT = {
    clean: "Referentie: dezelfde 1000 posities zonder effect. Alle zes engines vinden alles.",
    fog_05: "Lichte mist: FP16 is gelijk aan FP32, maar YOLO26s INT8 mist 14 procentpunten meer. Kwantisatie kost pas iets als het moeilijk wordt.",
    fog_10: "Zware mist: beide modellen vinden het platform minder dan de helft van de keren.",
    dust_05: "Licht stof: bijna geen effect.",
    dust_10: "Zwaar stof: de helft van de keren gemist, in elke rekenprecisie.",
    dark: "Weinig licht: YOLO26s vindt een derde, RT-DETR-l bijna niets (0,02).",
    blur: "Bewegingsonscherpte: YOLO26s INT8 verliest het meest.",
    noise: "Ruis: beide modellen vinden de helft.",
  };
  let galModel = "yolo26s", galCond = "fog_05";
  function showGallery() {
    const entry = G.conditions[galCond], dets = entry.detections[galModel], s = 100 / 640;
    const boxes = dets.map((d) => {
      const [x1, y1, x2, y2] = d.box;
      return `<div class="bb" style="left:${x1 * s}%;top:${y1 * s}%;width:${(x2 - x1) * s}%;height:${(y2 - y1) * s}%"><span>platform ${nl(d.conf, 2)}</span></div>`;
    }).join("");
    let miss = "";
    if (!dets.length && entry.label) {
      const [x1, y1, x2, y2] = entry.label;
      miss = `<div class="bb miss" style="left:${x1 * s}%;top:${y1 * s}%;width:${(x2 - x1) * s}%;height:${(y2 - y1) * s}%"><span>GEMIST</span></div>`;
    }
    $("gal-img").innerHTML = `<img src="img/gallery/${galCond}.jpg" alt="Testbeeld in conditie ${galCond}" style="max-height:52vh;object-fit:contain">${boxes}${miss}
      <span class="tagline">${NAME[galModel]} · ${CONDS.find((c) => c[0] === galCond)[1]} · ${entry.distance} m · θ ${entry.theta}°</span>`;
    const r = D.robustness[galCond];
    $("gal-bars-title").textContent = `RECALL ${NAME[galModel].toUpperCase()} PER REKENPRECISIE · JETSON`;
    $("gal-bars").innerHTML = ["fp32", "fp16", "int8"].map((p) => {
      const v = r[`${galModel}_${p}`].recall;
      return barRow(p.toUpperCase(), v, 1, PREC_COLOR[p], nl(v, 2));
    }).join("");
    $("gal-text").textContent = COND_TEXT[galCond];
    const fa = r[`${galModel}_fp32`].false_alarm;
    $("gal-note").textContent = `Vals alarm (beelden zonder platform, FP32): ${fa === null ? "–" : nl(fa * 100, 0) + " %"} · box: ${NAME[galModel]} op de pc, drempel 0,5`;
  }
  chips($("gal-model"), [["yolo26s", "YOLO26s"], ["rtdetr-l", "RT-DETR-l"]], (k) => { galModel = k; showGallery(); }, "yolo26s");
  chips($("gal-cond"), CONDS, (k) => { galCond = k; showGallery(); }, "fog_05");

  // ---------------------------------------------------------------- 14: antwoordtabel
  (function answer() {
    const head = "<tr><th>engine</th><th>keten FPS</th><th>live</th><th>recall mist 0,5</th><th>vals alarm mist</th></tr>";
    $("answer-table").innerHTML = head + ENGINES.map((k) => {
      const e = D.engines[k], fog = D.robustness.fog_05[k], ok = e.chain_fps_mean >= REQUIREMENT;
      return `<tr class="${k === "yolo26s_fp16" ? "win" : ""}"><td>${engineLabel(k)}</td><td class="${ok ? "ok" : "no"}">${nl(e.chain_fps_mean, 1)}</td>
        <td>${e.goede_landing + e.terecht_afgebroken}/${e.live_attempts}</td><td>${nl(fog.recall, 2)}</td><td>${fog.false_alarm === null ? "–" : nl(fog.false_alarm * 100, 0) + " %"}</td></tr>`;
    }).join("");
  })();

  // start op het hoofdstuk uit de adresbalk (#5), anders bij 1
  const start = parseInt((location.hash || "").slice(1), 10);
  go(Number.isFinite(start) ? start - 1 : 0);
})();
