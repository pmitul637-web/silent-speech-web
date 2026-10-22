import {
  FaceLandmarker,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/vision_bundle.mjs";
import * as Matcher from "./matcher.js";
import * as Predict from "./predict.js";

// Same 27 jaw/mouth blendshapes as the desktop version's capture.py.
const MOUTH_BLENDSHAPES = [
  "jawOpen", "jawForward", "jawLeft", "jawRight",
  "mouthClose", "mouthFunnel", "mouthPucker",
  "mouthLeft", "mouthRight",
  "mouthRollLower", "mouthRollUpper",
  "mouthShrugLower", "mouthShrugUpper",
  "mouthUpperUpLeft", "mouthUpperUpRight",
  "mouthLowerDownLeft", "mouthLowerDownRight",
  "mouthStretchLeft", "mouthStretchRight",
  "mouthPressLeft", "mouthPressRight",
  "mouthDimpleLeft", "mouthDimpleRight",
  "mouthFrownLeft", "mouthFrownRight",
  "mouthSmileLeft", "mouthSmileRight",
];

const STORAGE_KEY = "silent_speech_clips_v1";
const SETTINGS_KEY = "silent_speech_settings_v1";
const USAGE_KEY = "silent_speech_usage_v1";
const LEARNED_KEY = "silent_speech_learned_v1";
const HABITS_KEY = "silent_speech_habits_v1";
const SILENCE_FRAMES_TO_STOP = 8;
const MIN_CLIP_FRAMES = 3;
const LEARN_CAP = 10;          // clips learned from use, per phrase (newest kept)
const LIVE_EVERY = 6;          // frames between live guesses (~5 a second)
const LIVE_MIN_FRAMES = 12;    // don't guess on the first flicker of movement
const PRIOR_WEIGHT = 0.15;     // how much "what you usually say next" can tip a close call
const SURE_MARGIN = 1.5;       // winner this many times closer than runner-up = "Sure"
const FAIRLY_MARGIN = 1.2;

let movementThreshold = 0.15;
// A guess is accepted if it's within this many times the phrase's own normal
// spread between clips. Higher = more forgiving. On recorded clips, 98% of
// correct matches came in under 1.4, so 2.5 leaves room for a new day's
// lighting while still turning away random movement.
let acceptFactor = 2.5;

// Offered on the Train tab so a new phrase costs a tap, not typing.
const SUGGESTED = [
  "Yes", "No",
  "Thank you", "Please",
  "I need help", "Wait a moment",
  "I'm fine", "Not right now",
  "Can you repeat that", "I don't understand",
  "What time is it", "Tell me a joke",
];

const $ = (id) => document.getElementById(id);

const video = $("video");
const statusEl = $("status");
const meterFill = $("meter-fill");
const recDot = $("rec-dot");
const phraseInput = $("phrase-input");
const recordBtn = $("record-btn");
const clipCountsEl = $("clip-counts");
const listenBtn = $("listen-btn");
const transcriptEl = $("transcript");
const confirmPanel = $("confirm-panel");
const confirmText = $("confirm-text");
const confirmSure = $("confirm-sure");
const confirmAlts = $("confirm-alts");
const liveGuessEl = $("live-guess");
const likelyEl = $("likely");
const completionsEl = $("completions");
const confirmSendBtn = $("confirm-send-btn");
const confirmCancelBtn = $("confirm-cancel-btn");
const movementSlider = $("movement-slider");
const distanceSlider = $("distance-slider");
const movementValue = $("movement-value");
const distanceValue = $("distance-value");
const connEl = $("conn");
const phrasesEl = $("phrases");
const suggestionsEl = $("suggestions");
const targetLine = $("target-line");

function setStatus(text, kind = "") {
  statusEl.textContent = text;
  statusEl.className = "status " + kind;
}

// ---- Tabs ------------------------------------------------------------
document.querySelectorAll("#tabs button").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll("#tabs button").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    for (const name of ["talk", "train", "test", "settings"]) {
      $("tab-" + name).hidden = name !== btn.dataset.tab;
    }
    document.body.classList.toggle("training", btn.dataset.tab === "train");
    document.body.classList.toggle("settings", btn.dataset.tab === "settings");
    document.body.classList.toggle("testing-tab", btn.dataset.tab === "test");
    if (btn.dataset.tab === "talk") renderPhraseButtons();
    if (btn.dataset.tab === "test") {
      if (listening) setListening(false);   // a test does its own listening
      renderTestTab();
      syncTests();
    } else if (testing) {
      stopTest("Test stopped because you left the Test tab.");
    }
  });
});

// ---- Speak aloud, or ask Claude -------------------------------------
let mode = "speak";
document.querySelectorAll("#mode-switch button").forEach((btn) => {
  btn.addEventListener("click", () => {
    mode = btn.dataset.mode;
    document.querySelectorAll("#mode-switch button")
      .forEach((b) => b.classList.toggle("on", b === btn));
    saveSettings();
  });
});

// ---- Settings that survive a restart ---------------------------------
function loadSettings() {
  try {
    const s = JSON.parse(localStorage.getItem(SETTINGS_KEY) || "{}");
    if (typeof s.movementThreshold === "number") movementThreshold = s.movementThreshold;
    // The old maxDistance setting was on a different scale; it's ignored.
    if (typeof s.acceptFactor === "number") acceptFactor = s.acceptFactor;
    if (s.mode === "speak" || s.mode === "ask") mode = s.mode;
  } catch { /* first run */ }
  movementSlider.value = movementThreshold;
  distanceSlider.value = acceptFactor;
  movementValue.textContent = movementThreshold.toFixed(2);
  distanceValue.textContent = acceptFactor.toFixed(1);
  document.querySelectorAll("#mode-switch button")
    .forEach((b) => b.classList.toggle("on", b.dataset.mode === mode));
}
function saveSettings() {
  try {
    localStorage.setItem(SETTINGS_KEY,
      JSON.stringify({ movementThreshold, acceptFactor, mode }));
  } catch { /* storage full or blocked */ }
}
movementSlider.addEventListener("input", () => {
  movementThreshold = parseFloat(movementSlider.value);
  movementValue.textContent = movementThreshold.toFixed(2);
  saveSettings();
});
distanceSlider.addEventListener("input", () => {
  acceptFactor = parseFloat(distanceSlider.value);
  distanceValue.textContent = acceptFactor.toFixed(1);
  saveSettings();
});
loadSettings();

// ---- Storage for recorded reference clips (stays on this device) ----
// iPhones give a web app about 5 MB here. Clips used to be stored as plain
// JSON numbers, which filled that after ~100 clips. Each value is a 0..1
// mouth-shape score, so it is now packed as one byte (1/255 steps; tested
// with no change in accuracy), about 5-12x smaller. Old-format clips are
// still read, and are repacked on the next save.
function packClip(frames) {
  if (typeof frames === "string") return frames;
  const dim = frames.length ? frames[0].length : 0;
  const bytes = new Uint8Array(frames.length * dim);
  let i = 0;
  for (const f of frames) {
    for (const v of f) bytes[i++] = Math.round(Math.min(1, Math.max(0, v)) * 255);
  }
  let s = "";
  for (let j = 0; j < bytes.length; j += 0x8000) {
    s += String.fromCharCode.apply(null, bytes.subarray(j, j + 0x8000));
  }
  return dim + ":" + btoa(s);
}
function unpackClip(c) {
  if (Array.isArray(c)) return c;
  if (typeof c !== "string") return [];
  const k = c.indexOf(":");
  const dim = parseInt(c.slice(0, k), 10);
  if (!(dim > 0)) return [];
  const s = atob(c.slice(k + 1));
  const frames = [];
  for (let i = 0; i + dim <= s.length; i += dim) {
    const f = new Array(dim);
    for (let d = 0; d < dim; d++) f[d] = s.charCodeAt(i + d) / 255;
    frames.push(f);
  }
  return frames;
}

// Unpacking is not free, so each store is decoded once and kept in memory.
// Callers get their own copy of the lists, so an edit that then fails to
// save never leaves memory and storage disagreeing.
const storeCache = {};
function readStore(key) {
  if (!storeCache[key]) {
    let raw = {};
    try { raw = JSON.parse(localStorage.getItem(key) || "{}"); } catch { raw = {}; }
    const out = {};
    for (const [name, list] of Object.entries(raw)) {
      if (Array.isArray(list)) out[name] = list.map(unpackClip).filter((c) => c.length);
    }
    storeCache[key] = out;
  }
  return Object.fromEntries(Object.entries(storeCache[key]).map(([k, v]) => [k, v.slice()]));
}
function writeStore(key, data) {
  const packed = {};
  for (const [name, list] of Object.entries(data)) packed[name] = list.map(packClip);
  localStorage.setItem(key, JSON.stringify(packed));   // throws if full
  storeCache[key] = Object.fromEntries(
    Object.entries(data).map(([k, v]) => [k, v.map(unpackClip)]));
}

function loadClips() {
  return readStore(STORAGE_KEY);
}
function saveClips(clips) {
  invalidateIndex();
  try {
    writeStore(STORAGE_KEY, clips);
    return true;
  } catch (err) {
    alert("Could not save - this phone's storage for the app is full. "
        + "Back up, then delete a phrase you no longer use.");
    return false;
  }
}

// Repack anything still in the old, larger format. Writing the smaller
// version over the old one frees space; if it fails, the old copy stays.
function shrinkStorage() {
  for (const key of [STORAGE_KEY, LEARNED_KEY]) {
    try {
      const raw = localStorage.getItem(key);
      if (!raw || !raw.includes("[[")) continue;   // already packed
      writeStore(key, readStore(key));
    } catch { /* leave as is */ }
  }
}
shrinkStorage();

// ---- Clips learned from use ------------------------------------------
// Every time you confirm a guess (or pick the right one from the list),
// that movement is kept as an extra example. The app keeps getting better
// without you going back to Train. Kept apart from trained clips so the
// oldest can be dropped and "Undo last" still means your last recording.
function loadLearned() {
  return readStore(LEARNED_KEY);
}
function saveLearned(l) {
  invalidateIndex();
  try { writeStore(LEARNED_KEY, l); } catch { /* full: skip, not vital */ }
}
function learnFrom(sequence, phrase) {
  if (!sequence || sequence.length < MIN_CLIP_FRAMES) return;
  if (!loadClips()[phrase]) return;
  const l = loadLearned();
  const list = l[phrase] || [];
  list.push(sequence);
  while (list.length > LEARN_CAP) list.shift();
  l[phrase] = list;
  saveLearned(l);
}

function allClips() {
  const clips = loadClips();
  const learned = loadLearned();
  const out = {};
  for (const [name, list] of Object.entries(clips)) {
    out[name] = list.concat(learned[name] || []);
  }
  return out;
}

// Built once, then reused until clips change.
let matchIndex = null;
function invalidateIndex() { matchIndex = null; }
function getIndex() {
  if (!matchIndex) matchIndex = Matcher.buildIndex(allClips());
  return matchIndex;
}

// ---- Habits, for predicting what you'll say next ----------------------
let habits = (() => {
  try { return { ...Predict.emptyHabits(), ...JSON.parse(localStorage.getItem(HABITS_KEY) || "{}") }; }
  catch { return Predict.emptyHabits(); }
})();
function noteHabit(phrase) {
  Predict.recordUse(habits, phrase);
  try { localStorage.setItem(HABITS_KEY, JSON.stringify(habits)); } catch {}
}

// ---- How often each phrase gets used, so the useful ones come first --
function loadUsage() {
  try { return JSON.parse(localStorage.getItem(USAGE_KEY) || "{}"); } catch { return {}; }
}
function bumpUsage(name) {
  const u = loadUsage();
  u[name] = (u[name] || 0) + 1;
  try { localStorage.setItem(USAGE_KEY, JSON.stringify(u)); } catch {}
}
function phrasesByUse() {
  const clips = loadClips();
  const usage = loadUsage();
  return Object.keys(clips).sort((a, b) => {
    const d = (usage[b] || 0) - (usage[a] || 0);
    return d !== 0 ? d : a.localeCompare(b);
  });
}

// ---- Talk tab: one big button per phrase ----------------------------
function renderPhraseButtons() {
  const names = phrasesByUse();
  renderLikely(names);
  phrasesEl.innerHTML = "";
  if (!names.length) {
    const p = document.createElement("p");
    p.className = "empty";
    p.textContent = "No phrases yet. Go to Train, or load the ones from your laptop in Settings.";
    phrasesEl.append(p);
    return;
  }
  for (const name of names) {
    const b = document.createElement("button");
    b.className = "phrase-btn";
    b.textContent = name;
    b.addEventListener("click", () => usePhrase(name));
    phrasesEl.append(b);
  }
}

// Best guesses for what you'll say next: what usually follows what you
// just said, and what you tend to say at this time of day.
function renderLikely(names) {
  // With only a few phrases the whole list is already on screen.
  const next = names.length > 3 ? Predict.likelyNext(habits, names) : [];
  likelyEl.innerHTML = "";
  likelyEl.hidden = !next.length;
  if (!next.length) return;
  const h = document.createElement("h2");
  h.textContent = "Likely next";
  const grid = document.createElement("div");
  grid.className = "likely-grid";
  for (const name of next) {
    const b = document.createElement("button");
    b.className = "phrase-btn likely-btn";
    b.textContent = name;
    b.addEventListener("click", () => usePhrase(name));
    grid.append(b);
  }
  likelyEl.append(h, grid);
}

// Tapping and mouthing both end up here.
function usePhrase(text) {
  bumpUsage(text);
  noteHabit(text);
  logLine(text, "me");
  if (mode === "ask") {
    askClaude(text);
  } else {
    speak(text);
  }
  renderPhraseButtons();
}

// ---- Train tab: suggestions instead of typing -----------------------
let target = "";

function setTarget(value) {
  target = value.trim();
  targetLine.innerHTML = target
    ? `Recording: <strong>${target}</strong>`
    : "Pick a phrase above first.";
  recordBtn.disabled = !target || !faceLandmarker;
  document.querySelectorAll(".sugg")
    .forEach((b) => b.classList.toggle("on", b.textContent === target));
}

function renderSuggestions() {
  suggestionsEl.innerHTML = "";
  for (const s of SUGGESTED) {
    const b = document.createElement("button");
    b.className = "sugg";
    b.textContent = s;
    b.addEventListener("click", () => {
      phraseInput.hidden = true;
      setTarget(s);
    });
    suggestionsEl.append(b);
  }
}
renderSuggestions();
setTarget("");

$("other-btn").addEventListener("click", () => {
  phraseInput.hidden = false;
  phraseInput.focus();
});
phraseInput.addEventListener("input", () => {
  setTarget(phraseInput.value);
  renderCompletions();
});

function renderCompletions() {
  const options = Predict.completions(
    phraseInput.value,
    phrasesByUse(),
    SUGGESTED.concat(Predict.COMMON),
    loadUsage(),
  );
  completionsEl.innerHTML = "";
  for (const text of options) {
    const b = document.createElement("button");
    b.className = "sugg completion";
    b.textContent = text;
    b.addEventListener("click", () => {
      phraseInput.value = text;
      setTarget(text);
      completionsEl.innerHTML = "";
    });
    completionsEl.append(b);
  }
}

function renderClipCounts() {
  const clips = loadClips();
  const learned = loadLearned();
  const names = phrasesByUse();
  clipCountsEl.innerHTML = "";
  if (!names.length) {
    clipCountsEl.innerHTML = '<p class="hint">Nothing trained yet.</p>';
    return;
  }
  for (const name of names) {
    const row = document.createElement("div");
    row.className = "phrase-row";

    const label = document.createElement("span");
    label.className = "name";
    label.textContent = name;

    const count = document.createElement("span");
    count.className = "count";
    const extra = (learned[name] || []).length;
    count.textContent = `${clips[name].length} clips` + (extra ? ` + ${extra} learned` : "");

    const undo = document.createElement("button");
    undo.textContent = "Undo last";
    undo.addEventListener("click", () => {
      const c = loadClips();
      c[name].pop();
      if (!c[name].length) delete c[name];
      saveClips(c);
      renderClipCounts();
      renderPhraseButtons();
    });

    const del = document.createElement("button");
    del.textContent = "Delete";
    del.addEventListener("click", () => {
      if (!confirm(`Delete "${name}" and all its clips?`)) return;
      const c = loadClips();
      delete c[name];
      saveClips(c);
      const l = loadLearned();
      if (l[name]) { delete l[name]; saveLearned(l); }
      Predict.forget(habits, name);
      try { localStorage.setItem(HABITS_KEY, JSON.stringify(habits)); } catch {}
      renderClipCounts();
      renderPhraseButtons();
    });

    row.append(label, count, undo, del);
    clipCountsEl.append(row);
  }
}
renderClipCounts();
renderPhraseButtons();

// ---- Backup and restore ---------------------------------------------
$("export-btn").addEventListener("click", () => {
  // Learned clips go in as ordinary clips, so a restore keeps them.
  const plain = {};
  for (const [name, list] of Object.entries(allClips())) {
    plain[name] = list.map((c) => c.map((f) => f.map((v) => Math.round(v * 1000) / 1000)));
  }
  const blob = new Blob([JSON.stringify(plain)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `silent-speech-phrases-${new Date().toISOString().slice(0, 10)}.json`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 2000);
});

$("import-btn").addEventListener("click", () => $("import-file").click());

$("import-file").addEventListener("change", async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  try {
    const incoming = JSON.parse(await file.text());
    if (typeof incoming !== "object" || Array.isArray(incoming)) throw new Error("wrong shape");
    const clips = loadClips();
    let added = 0;
    for (const [name, list] of Object.entries(incoming)) {
      if (!Array.isArray(list)) continue;
      const good = list.map(unpackClip).filter((c) => c.length);
      clips[name] = (clips[name] || []).concat(good);
      added += good.length;
    }
    if (saveClips(clips)) {
      renderClipCounts();
      renderPhraseButtons();
      alert(`Restored ${added} clips.`);
    }
  } catch (err) {
    alert("That file could not be read as a phrase backup.");
  }
  e.target.value = "";
});

// ---- Pull the laptop-trained phrases across ---------------------------
$("starter-btn").addEventListener("click", async () => {
  const btn = $("starter-btn");
  const label = btn.textContent;
  btn.disabled = true;
  btn.textContent = "Loading…";
  try {
    const res = await fetch("/starter-phrases.json", { cache: "no-store" });
    if (!res.ok) throw new Error(String(res.status));
    const incoming = await res.json();

    const clips = loadClips();
    let added = 0;
    let phrases = 0;
    for (const [name, list] of Object.entries(incoming)) {
      if (!Array.isArray(list)) continue;
      // Compared packed: stored clips are rounded to 1/255, so raw numbers
      // would never match and the same clips would be added again.
      const seen = new Set((clips[name] || []).map(packClip));
      const fresh = list.filter((c) => Array.isArray(c) && !seen.has(packClip(c)));
      if (!fresh.length) continue;
      clips[name] = (clips[name] || []).concat(fresh);
      added += fresh.length;
      phrases += 1;
    }

    if (!added) {
      alert("Already loaded — nothing new to add.");
    } else if (saveClips(clips)) {
      renderClipCounts();
      renderPhraseButtons();
      alert(`Added ${added} clips across ${phrases} phrases.`);
    }
  } catch (err) {
    alert("Could not load them. Check your connection and try again.");
  } finally {
    btn.disabled = false;
    btn.textContent = label;
  }
});

// ---- Matching (the method itself is in matcher.js) -------------------
// Ranks every phrase for a finished movement. In the Talk tab, what you
// usually say next can tip a close call by up to PRIOR_WEIGHT, never a
// clear one. The Test tab passes usePrior: false so scores stay honest.
function guess(sequence, { usePrior = true } = {}) {
  const idx = getIndex();
  let ranked = Matcher.rank(idx, sequence);
  if (usePrior && ranked.length > 1) {
    const s = Predict.scores(habits, ranked.map((r) => r.name));
    const top = Math.max(0, ...Object.values(s));
    if (top > 0) {
      ranked = ranked
        .map((r) => ({ ...r, distance: r.distance * (1 - PRIOR_WEIGHT * (s[r.name] || 0) / top) }))
        .sort((a, b) => a.distance - b.distance);
    }
  }
  return { ranked, ok: Matcher.accepted(idx, ranked, acceptFactor) };
}

// How sure, in words, from how far ahead the winner is of the runner-up.
function sureness(ranked) {
  const m = Matcher.margin(ranked);
  if (m >= SURE_MARGIN) return "Sure";
  if (m >= FAIRLY_MARGIN) return "Fairly sure";
  return "Not sure";
}

function vectorNorm(v) {
  let sum = 0;
  for (const x of v) sum += x * x;
  return Math.sqrt(sum);
}

// ---- Keep the screen awake while listening ---------------------------
let wakeLock = null;
async function keepAwake(on) {
  try {
    if (on && "wakeLock" in navigator) {
      wakeLock = await navigator.wakeLock.request("screen");
    } else if (wakeLock) {
      await wakeLock.release();
      wakeLock = null;
    }
  } catch { /* not supported, or denied - not important */ }
}
document.addEventListener("visibilitychange", () => {
  if (document.visibilityState === "visible" && (listening || testing) && !wakeLock) keepAwake(true);
});

// ---- Camera + MediaPipe face tracking ----
let faceLandmarker;
let lastVideoTime = -1;

async function setup() {
  if (navigator.storage && navigator.storage.persist) {
    navigator.storage.persist().catch(() => {});
  }

  setStatus("Loading face tracker…");
  const filesetResolver = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/wasm"
  );

  try {
    faceLandmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
      baseOptions: { modelAssetPath: "/models/face_landmarker.task", delegate: "GPU" },
      outputFaceBlendshapes: true,
      runningMode: "VIDEO",
      numFaces: 1,
    });
  } catch (err) {
    // Some browsers/devices don't support the GPU delegate — fall back to CPU.
    faceLandmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
      baseOptions: { modelAssetPath: "/models/face_landmarker.task", delegate: "CPU" },
      outputFaceBlendshapes: true,
      runningMode: "VIDEO",
      numFaces: 1,
    });
  }

  setStatus("Asking for the camera…");
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: "user" },
    audio: false,
  });
  video.srcObject = stream;
  await new Promise((resolve) => { video.onloadedmetadata = resolve; });
  await video.play();

  setStatus("Ready.", "ok");
  listenBtn.disabled = false;
  setTarget(target);
  renderTestTab();
  requestAnimationFrame(detectFrame);
}

function getMouthVector(result) {
  if (!result.faceBlendshapes || !result.faceBlendshapes.length) return null;
  const categories = result.faceBlendshapes[0].categories;
  const scores = {};
  for (const c of categories) scores[c.categoryName] = c.score;
  return MOUTH_BLENDSHAPES.map((name) => scores[name] ?? 0);
}

function detectFrame() {
  if (video.readyState >= 2 && video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    const result = faceLandmarker.detectForVideo(video, performance.now());
    onFrame(getMouthVector(result));
  }
  requestAnimationFrame(detectFrame);
}

// ---- Recording state ----
let recording = false;
let recordFrames = [];

recordBtn.addEventListener("click", () => {
  if (!recording) {
    if (!target) { alert("Pick a phrase first."); return; }
    recording = true;
    recordFrames = [];
    recordBtn.textContent = "Stop recording";
    recordBtn.classList.add("stop");
    recDot.hidden = false;
  } else {
    recording = false;
    recordBtn.textContent = "Record a clip";
    recordBtn.classList.remove("stop");
    recDot.hidden = true;
    if (recordFrames.length < MIN_CLIP_FRAMES) {
      alert(`Only caught ${recordFrames.length} frames with a face in them. `
          + "More light on your face, and hold still.");
      return;
    }
    const clips = loadClips();
    if (!clips[target]) clips[target] = [];
    clips[target].push(recordFrames);
    if (saveClips(clips)) {
      renderClipCounts();
      renderPhraseButtons();
      setStatus(`Saved. "${target}" now has ${clips[target].length} clips.`, "ok");
    }
  }
});

// ---- Listening state ----
let listening = false;
let mouthActive = false;
let buffer = [];
let quietStreak = 0;
let history = [];
// While a match is awaiting confirmation, or a request to Claude is in
// flight, we stop reacting to new mouth movement — otherwise repeating
// the phrase while waiting piles up multiple simultaneous requests and
// nothing seems to "go forward".
let paused = false;
let pendingPhrase = "";
let pendingSequence = null;   // the movement being confirmed, so it can be learned

function setListening(on) {
  listening = on;
  listenBtn.textContent = listening ? "Stop listening" : "Start listening";
  listenBtn.classList.toggle("stop", listening);
  keepAwake(listening);
  if (!listening) {
    mouthActive = false;
    buffer = [];
    hideConfirmPanel();
    hideLiveGuess();
    meterFill.style.width = "0%";
  }
}

// Best guess so far, shown over the camera while you're still mouthing.
function showLiveGuess(frames) {
  const idx = getIndex();
  if (!idx.entries.length) return;
  const ranked = Matcher.rank(idx, frames, { openEnd: true, keep: 2 });
  if (!ranked.length) return;
  liveGuessEl.textContent = ranked[0].name + "…";
  liveGuessEl.classList.toggle("sure", Matcher.margin(ranked) >= SURE_MARGIN);
  liveGuessEl.hidden = false;
}
function hideLiveGuess() {
  liveGuessEl.hidden = true;
}
listenBtn.addEventListener("click", () => setListening(!listening));

function onFrame(vector) {
  if (vector) {
    const level = Math.min(1, vectorNorm(vector) / 0.6);
    meterFill.style.width = (level * 100).toFixed(0) + "%";
    meterFill.classList.toggle("hot", vectorNorm(vector) > movementThreshold);
  }

  if (recording) {
    if (vector) recordFrames.push(vector);
    return;
  }
  if (!(listening || testing) || paused || !vector) return;

  const movement = vectorNorm(vector);
  if (movement > movementThreshold) {
    mouthActive = true;
    quietStreak = 0;
    buffer.push(vector);
  } else if (mouthActive) {
    quietStreak += 1;
    buffer.push(vector);
    if (quietStreak >= SILENCE_FRAMES_TO_STOP) {
      const trimmed = buffer.slice(0, -SILENCE_FRAMES_TO_STOP);
      const sequence = trimmed.length ? trimmed : buffer;
      hideLiveGuess();
      if (sequence.length >= MIN_CLIP_FRAMES && testing) {
        recordTry(sequence);
      } else if (sequence.length >= MIN_CLIP_FRAMES) {
        const { ranked, ok } = guess(sequence);
        if (ok) {
          promptConfirm(ranked, sequence);
        } else {
          logLine(ranked.length
            ? `didn't catch that — closest was "${ranked[0].name}", but not close enough`
            : "didn't catch that", "sys");
        }
      }
      mouthActive = false;
      buffer = [];
      return;
    }
  }

  if (listening && !testing && mouthActive
      && buffer.length >= LIVE_MIN_FRAMES && buffer.length % LIVE_EVERY === 0) {
    showLiveGuess(buffer);
  }
}

function logLine(text, cls = "") {
  const p = document.createElement("p");
  if (cls) p.className = cls;
  p.textContent = text;
  transcriptEl.prepend(p);
  return p;
}

function speak(text) {
  const utter = new SpeechSynthesisUtterance(text);
  speechSynthesis.speak(utter);
}

function promptConfirm(ranked, sequence) {
  paused = true;
  pendingPhrase = ranked[0].name;
  pendingSequence = sequence;
  confirmText.textContent = pendingPhrase;

  const sure = sureness(ranked);
  confirmSure.textContent = sure;
  confirmSure.className = "sure-tag " + sure.toLowerCase().replace(/\s+/g, "-");

  // The runners-up, one tap away, in case the top guess is wrong.
  confirmAlts.innerHTML = "";
  const alts = ranked.slice(1, 3);
  if (alts.length) {
    const label = document.createElement("p");
    label.className = "hint";
    label.textContent = "Or did you mean:";
    confirmAlts.append(label);
    for (const alt of alts) {
      const b = document.createElement("button");
      b.className = "alt-btn";
      b.textContent = alt.name;
      b.addEventListener("click", () => choosePhrase(alt.name));
      confirmAlts.append(b);
    }
  }

  logLine(`heard that as "${pendingPhrase}" (${sure.toLowerCase()})`, "sys");
  confirmPanel.hidden = false;
}

function hideConfirmPanel() {
  confirmPanel.hidden = true;
  pendingPhrase = "";
  pendingSequence = null;
  paused = false;
}

// You confirmed a guess or picked the right one: use it, and keep the
// movement as a new example of that phrase.
function choosePhrase(name) {
  const sequence = pendingSequence;
  hideConfirmPanel();
  learnFrom(sequence, name);
  renderClipCounts();
  usePhrase(name);
}

confirmCancelBtn.addEventListener("click", () => {
  logLine("cancelled — listening again", "sys");
  hideConfirmPanel();
});

confirmSendBtn.addEventListener("click", () => {
  if (!pendingPhrase) { hideConfirmPanel(); return; }
  choosePhrase(pendingPhrase);
});

// e.g. "Friday 2 October 2026 at 16:10 BST"
function phoneTime() {
  return new Date().toLocaleString("en-GB", {
    weekday: "long", day: "numeric", month: "long", year: "numeric",
    hour: "2-digit", minute: "2-digit", timeZoneName: "short",
  });
}

async function askClaude(phrase) {
  paused = true;
  const pending = logLine("asking Claude…", "sys");
  try {
    const res = await fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      // Claude has no clock, so send the phone's local time with every question.
      body: JSON.stringify({ phrase, history, now: phoneTime() }),
    });
    if (!res.ok) throw new Error(`server returned ${res.status}`);
    const data = await res.json();
    history.push({ role: "user", content: phrase });
    history.push({ role: "assistant", content: data.reply });
    pending.remove();
    logLine(data.reply);
    speak(data.reply);
  } catch (err) {
    pending.remove();
    logLine(navigator.onLine
      ? "Could not reach the server. Check your connection and try again."
      : "You are offline. Saying phrases out loud still works.",
      "sys");
  } finally {
    paused = false;
  }
}

// ---- Test tab: 10 tries, scored and logged automatically -------------
// Uses the same movement detection and matcher as the Talk tab, so a score
// here means the same as real use. It records a little more than the Talk
// tab shows — the distance to the right phrase and to the closest wrong
// one — because that is what tells us WHY a phrase fails.
const TESTS_KEY = "silent_speech_tests_v1";
const TEST_TRIES = 10;
const PASS_MARK = 8;
const PAUSE_BETWEEN_TRIES_MS = 1200;

let testing = false;
let testPhrase = "";
let testLight = "Normal";
let testTries = [];
let testSequences = [];   // raw mouth data per try; sent to the laptop, not kept on the phone
let testResumeTimer = null;

const round2 = (x) => Math.round(x * 100) / 100;

function loadTests() {
  try { return JSON.parse(localStorage.getItem(TESTS_KEY) || "[]"); } catch { return []; }
}
function saveTests(list) {
  try { localStorage.setItem(TESTS_KEY, JSON.stringify(list)); return true; } catch { return false; }
}

function shortDate(iso) {
  return new Date(iso).toLocaleDateString(undefined, { day: "numeric", month: "short" });
}

function renderTestTab() {
  // Phrase picker
  const holder = $("test-phrases");
  holder.innerHTML = "";
  const names = phrasesByUse();
  if (!names.length) {
    const p = document.createElement("p");
    p.className = "empty";
    p.textContent = "No phrases yet. Train one first.";
    holder.append(p);
  }
  if (testPhrase && !names.includes(testPhrase)) testPhrase = "";
  for (const name of names) {
    const b = document.createElement("button");
    b.className = "sugg" + (name === testPhrase ? " on" : "");
    b.textContent = name;
    b.disabled = testing;
    b.addEventListener("click", () => {
      testPhrase = name;
      $("test-done").hidden = true;
      renderTestTab();
    });
    holder.append(b);
  }
  $("test-start").hidden = testing;
  $("test-start").disabled = !testPhrase || !faceLandmarker;
  $("test-start").textContent = testPhrase
    ? `Start 10 tries of "${testPhrase}"`
    : "Pick a phrase first";
  document.querySelectorAll("#test-light button")
    .forEach((b) => { b.disabled = testing; });
  renderTestSummary();
}

function renderTestSummary() {
  const el = $("test-summary");
  el.innerHTML = "";
  const tests = loadTests();
  const names = phrasesByUse();
  // Show trained phrases, plus any tested phrase since deleted.
  for (const t of tests) if (!names.includes(t.phrase)) names.push(t.phrase);
  if (!names.length) {
    el.innerHTML = '<p class="hint">No results yet.</p>';
    return;
  }
  for (const name of names) {
    const mine = tests.filter((t) => t.phrase === name)
      .sort((a, b) => b.date.localeCompare(a.date));
    const row = document.createElement("div");
    row.className = "phrase-row test-row";

    const label = document.createElement("span");
    label.className = "name";
    label.textContent = name;

    const badge = document.createElement("span");
    if (!mine.length) {
      badge.className = "badge";
      badge.textContent = "Not tested yet";
    } else {
      const last = mine[0];
      badge.className = "badge " + (last.passed ? "pass" : "fail");
      badge.textContent = `${last.score}/${last.total} ${last.passed ? "✓ Working" : "✗ Not yet"}`;
    }
    row.append(label, badge);

    if (mine.length) {
      const past = document.createElement("span");
      past.className = "past";
      past.textContent = mine.slice(0, 4)
        .map((t) => `${shortDate(t.date)} ${t.score}/${t.total} (${t.light})`)
        .join(" · ");
      row.append(past);
    }
    el.append(row);
  }
}

document.querySelectorAll("#test-light button").forEach((btn) => {
  btn.addEventListener("click", () => {
    testLight = btn.dataset.light;
    document.querySelectorAll("#test-light button")
      .forEach((b) => b.classList.toggle("on", b === btn));
  });
});

function renderLive(message = "", cls = "") {
  const n = testTries.length;
  const right = testTries.filter((t) => t.correct).length;
  $("test-count").textContent = `Try ${Math.min(n + 1, TEST_TRIES)} of ${TEST_TRIES}`;
  const prompt = $("test-prompt");
  prompt.textContent = message || `Mouth "${testPhrase}" now.`;
  prompt.className = cls;
  const dots = $("test-dots");
  dots.innerHTML = "";
  for (let i = 0; i < TEST_TRIES; i++) {
    const s = document.createElement("span");
    if (i < n) s.className = testTries[i].correct ? "ok" : "no";
    dots.append(s);
  }
  $("test-tally").textContent = n ? `${right} right, ${n - right} wrong so far` : "";
  $("test-undo").disabled = !n;
}

$("test-start").addEventListener("click", () => {
  if (!testPhrase) return;
  testing = true;
  testTries = [];
  testSequences = [];
  mouthActive = false;
  buffer = [];
  paused = false;
  keepAwake(true);
  $("test-done").hidden = true;
  $("test-live").hidden = false;
  renderTestTab();
  renderLive();
});

function stopTest(message) {
  clearTimeout(testResumeTimer);
  testing = false;
  paused = false;
  mouthActive = false;
  buffer = [];
  keepAwake(false);
  $("test-live").hidden = true;
  if (message) setStatus(message);
  renderTestTab();
}

$("test-stop").addEventListener("click", () => {
  if (testTries.length && !confirm("Stop now? These tries won't be saved.")) return;
  stopTest("Test stopped. Nothing saved.");
});

$("test-undo").addEventListener("click", () => {
  testTries.pop();
  testSequences.pop();
  renderLive("Last try removed. Go again when ready.");
});

function recordTry(sequence) {
  // Every phrase scored (keep: Infinity), and no habit nudge, so the test
  // measures recognition alone.
  const idx = getIndex();
  const ranked = Matcher.rank(idx, sequence, { keep: Infinity });
  const top = ranked[0];
  const got = top && Matcher.accepted(idx, ranked, acceptFactor) ? top.name : null;
  const own = ranked.find((r) => r.name === testPhrase);
  const rival = ranked.find((r) => r.name !== testPhrase);
  const correct = got === testPhrase;

  testTries.push({
    got,
    correct,
    distance: top ? round2(top.distance) : null,
    ownDistance: own ? round2(own.distance) : null,
    closestOther: rival ? rival.name : null,
    closestOtherDistance: rival ? round2(rival.distance) : null,
    frames: sequence.length,
  });
  testSequences.push(sequence);

  if (testTries.length >= TEST_TRIES) {
    finishTest();
    return;
  }

  // Short break so the next try doesn't start mid-movement.
  const heard = got ? `Heard "${got}"` : "No match";
  renderLive(`${correct ? "✓" : "✗"} ${heard}. Get ready…`, correct ? "ok" : "no");
  paused = true;
  clearTimeout(testResumeTimer);
  testResumeTimer = setTimeout(() => {
    paused = false;
    if (testing) renderLive();
  }, PAUSE_BETWEEN_TRIES_MS);
}

function finishTest() {
  const clips = loadClips();
  const score = testTries.filter((t) => t.correct).length;
  const result = {
    id: Date.now(),
    date: new Date().toISOString(),
    phrase: testPhrase,
    score,
    total: testTries.length,
    passed: score >= PASS_MARK,
    light: testLight,
    clips: (clips[testPhrase] || []).length,
    phrasesTrained: Object.keys(clips).length,
    settings: { movementThreshold, acceptFactor, matcher: "v2" },
    tries: testTries,
    sent: false,
  };
  const sequences = testSequences;
  const tests = loadTests();
  tests.push(result);
  const stored = saveTests(tests);

  stopTest(stored ? "Test saved." : "Could not save the test — phone storage is full.");
  showDone(result);
  sendTest(result, sequences);
}

function showDone(r) {
  const done = $("test-done");
  done.innerHTML = "";
  const score = document.createElement("div");
  score.className = "score";
  score.textContent = `${r.score} / ${r.total}`;
  const verdict = document.createElement("div");
  verdict.className = "verdict " + (r.passed ? "pass" : "fail");
  verdict.textContent = r.passed ? "✓ Working" : `✗ Not yet — needs ${PASS_MARK}`;
  const detail = document.createElement("p");
  detail.textContent = `"${r.phrase}" · ${r.light} light · ${r.clips} clips trained`;
  done.append(score, verdict, detail);

  // The most common wrong answer is the most useful thing to know.
  const wrong = r.tries.filter((t) => !t.correct)
    .map((t) => t.got || "no match");
  if (wrong.length) {
    const counts = {};
    for (const w of wrong) counts[w] = (counts[w] || 0) + 1;
    const [mostName, mostCount] = Object.entries(counts).sort((a, b) => b[1] - a[1])[0];
    const p = document.createElement("p");
    p.className = "hint";
    p.textContent = mostName === "no match"
      ? `Missed ${mostCount} times with no match at all.`
      : `Mixed up with "${mostName}" ${mostCount} times.`;
    done.append(p);
  }
  done.hidden = false;
}

// ---- Send results to the laptop, so Claude can read them there -------
async function sendTest(result, sequences) {
  try {
    const res = await fetch("/test-log", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ result, sequences: sequences || null }),
    });
    if (!res.ok) throw new Error(String(res.status));
    const tests = loadTests();
    const t = tests.find((x) => x.id === result.id);
    if (t) { t.sent = true; saveTests(tests); }
    $("test-sync").textContent = "Results sent to the server too.";
    return true;
  } catch {
    $("test-sync").textContent =
      "Saved on this phone. Laptop not reachable — it will send next time, or tap Save report.";
    return false;
  }
}

// Anything the laptop missed goes across next time the Test tab opens.
async function syncTests() {
  const unsent = loadTests().filter((t) => !t.sent);
  if (!unsent.length) {
    $("test-sync").textContent = "";
    return;
  }
  for (const t of unsent) {
    if (!(await sendTest(t, null))) break;
  }
}

$("test-export").addEventListener("click", () => {
  const blob = new Blob([JSON.stringify(loadTests(), null, 1)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `silent-speech-tests-${new Date().toISOString().slice(0, 10)}.json`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 2000);
});

renderTestTab();

// ---- Is the laptop there? -------------------------------------------
async function checkConnection() {
  if (!navigator.onLine) {
    connEl.textContent = "Offline. Saying phrases out loud still works.";
    return;
  }
  try {
    const res = await fetch("/ask", { method: "POST",
      headers: { "Content-Type": "application/json" }, body: "{}" });
    connEl.textContent = (res.status === 400 || res.ok)
      ? "Connected. Claude replies will work."
      : `Server reachable, but it answered ${res.status}.`;
  } catch {
    connEl.textContent = "Cannot reach the server. Saying phrases out loud still works.";
  }
}
checkConnection();
window.addEventListener("online", checkConnection);
window.addEventListener("offline", checkConnection);

// ---- Offer to install, on iOS Safari only ---------------------------
const standalone = window.navigator.standalone === true
  || window.matchMedia("(display-mode: standalone)").matches;
const isIOS = /iPad|iPhone|iPod/.test(navigator.userAgent);
if (isIOS && !standalone && !localStorage.getItem("install_hint_dismissed")) {
  $("install-hint").hidden = false;
}
$("install-dismiss").addEventListener("click", () => {
  $("install-hint").hidden = true;
  try { localStorage.setItem("install_hint_dismissed", "1"); } catch {}
});

// ---- Service worker, so it opens instantly and works offline --------
if ("serviceWorker" in navigator) {
  navigator.serviceWorker.register("/sw.js").catch(() => {});
}

setup().catch((err) => {
  setStatus(`Could not start: ${err.message}`, "err");
});
