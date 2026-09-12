import {
  FaceLandmarker,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/vision_bundle.mjs";

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
const SILENCE_FRAMES_TO_STOP = 8;
const MIN_CLIP_FRAMES = 3;

let movementThreshold = 0.15;
let maxDistance = 15;

const video = document.getElementById("video");
const statusEl = document.getElementById("status");
const phraseInput = document.getElementById("phrase-input");
const recordBtn = document.getElementById("record-btn");
const clipCountsEl = document.getElementById("clip-counts");
const listenBtn = document.getElementById("listen-btn");
const transcriptEl = document.getElementById("transcript");
const movementSlider = document.getElementById("movement-slider");
const distanceSlider = document.getElementById("distance-slider");
const movementValue = document.getElementById("movement-value");
const distanceValue = document.getElementById("distance-value");

movementSlider.addEventListener("input", () => {
  movementThreshold = parseFloat(movementSlider.value);
  movementValue.textContent = movementThreshold.toFixed(2);
});
distanceSlider.addEventListener("input", () => {
  maxDistance = parseFloat(distanceSlider.value);
  distanceValue.textContent = maxDistance.toFixed(0);
});

// ---- Storage for recorded reference clips (stays on this device) ----
function loadClips() {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}");
  } catch {
    return {};
  }
}
function saveClips(clips) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(clips));
}
function renderClipCounts() {
  const clips = loadClips();
  const entries = Object.entries(clips).map(([name, list]) => `${name}: ${list.length}`);
  clipCountsEl.textContent = entries.length ? entries.join(" · ") : "No phrases recorded yet.";
}
renderClipCounts();

// ---- Same DTW nearest-neighbor matcher as classify.py, ported to JS ----
function dtwDistance(a, b) {
  const n = a.length;
  const m = b.length;
  const cost = Array.from({ length: n + 1 }, () => new Float64Array(m + 1).fill(Infinity));
  cost[0][0] = 0;
  for (let i = 1; i <= n; i++) {
    for (let j = 1; j <= m; j++) {
      let sq = 0;
      for (let k = 0; k < a[i - 1].length; k++) {
        const diff = a[i - 1][k] - b[j - 1][k];
        sq += diff * diff;
      }
      const stepCost = Math.sqrt(sq);
      cost[i][j] = stepCost + Math.min(cost[i - 1][j], cost[i][j - 1], cost[i - 1][j - 1]);
    }
  }
  return cost[n][m] / (n + m);
}

function classify(sequence) {
  const clips = loadClips();
  let best = null;
  let bestDist = Infinity;
  for (const [name, examples] of Object.entries(clips)) {
    for (const clip of examples) {
      const d = dtwDistance(sequence, clip);
      if (d < bestDist) {
        bestDist = d;
        best = name;
      }
    }
  }
  if (bestDist > maxDistance) return [null, bestDist];
  return [best, bestDist];
}

function vectorNorm(v) {
  let sum = 0;
  for (const x of v) sum += x * x;
  return Math.sqrt(sum);
}

// ---- Camera + MediaPipe face tracking ----
let faceLandmarker;
let lastVideoTime = -1;

async function setup() {
  statusEl.textContent = "Loading face tracker…";
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

  statusEl.textContent = "Requesting camera…";
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: "user" },
    audio: false,
  });
  video.srcObject = stream;
  await new Promise((resolve) => { video.onloadedmetadata = resolve; });
  await video.play();

  statusEl.textContent = "Ready.";
  recordBtn.disabled = false;
  listenBtn.disabled = false;
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
    recording = true;
    recordFrames = [];
    recordBtn.textContent = "Stop recording";
  } else {
    recording = false;
    recordBtn.textContent = "Record a clip";
    const phrase = phraseInput.value.trim();
    if (!phrase) {
      alert("Type a phrase first.");
      return;
    }
    if (recordFrames.length < MIN_CLIP_FRAMES) {
      alert(`Only captured ${recordFrames.length} frames with a face detected — try again with better lighting/framing.`);
      return;
    }
    const clips = loadClips();
    if (!clips[phrase]) clips[phrase] = [];
    clips[phrase].push(recordFrames);
    saveClips(clips);
    renderClipCounts();
  }
});

// ---- Listening state ----
let listening = false;
let mouthActive = false;
let buffer = [];
let quietStreak = 0;
let history = [];

listenBtn.addEventListener("click", () => {
  listening = !listening;
  listenBtn.textContent = listening ? "Stop listening" : "Start listening";
  if (!listening) {
    mouthActive = false;
    buffer = [];
  }
});

function onFrame(vector) {
  if (recording) {
    if (vector) recordFrames.push(vector);
    return;
  }
  if (!listening || !vector) return;

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
      if (sequence.length >= MIN_CLIP_FRAMES) {
        const [command, distance] = classify(sequence);
        if (command) {
          handleMatch(command, distance);
        } else {
          logLine(`(no match) — distance ${distance.toFixed(2)}`);
        }
      }
      mouthActive = false;
      buffer = [];
    }
  }
}

function logLine(text) {
  const p = document.createElement("p");
  p.textContent = text;
  transcriptEl.prepend(p);
}

function speak(text) {
  const utter = new SpeechSynthesisUtterance(text);
  speechSynthesis.speak(utter);
}

async function handleMatch(phrase, distance) {
  logLine(`→ ${phrase} (distance ${distance.toFixed(2)})`);
  logLine("…asking Claude…");
  try {
    const res = await fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ phrase, history }),
    });
    if (!res.ok) throw new Error(`server returned ${res.status}`);
    const data = await res.json();
    history.push({ role: "user", content: phrase });
    history.push({ role: "assistant", content: data.reply });
    logLine(`Claude: ${data.reply}`);
    speak(data.reply);
  } catch (err) {
    logLine(`Error talking to Claude: ${err}`);
  }
}

setup().catch((err) => {
  statusEl.textContent = `Setup failed: ${err.message}`;
});
