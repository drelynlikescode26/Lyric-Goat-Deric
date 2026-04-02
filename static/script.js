/* ── State ── */
let mediaRecorder = null;
let audioChunks = [];
let recordedBlob = null;
let uploadedFile = null;
let isRecording = false;
let timerInterval = null;
let recordingSeconds = 0;

const state = {
  tone: "melodic",
  mode: "verse",
  vibe: "introspective",
};

/* ── DOM refs ── */
const recordBtn = document.getElementById("recordBtn");
const recordLabel = document.getElementById("recordLabel");
const recordingStatus = document.getElementById("recordingStatus");
const recordTimer = document.getElementById("recordTimer");
const audioPreview = document.getElementById("audioPreview");
const fileInput = document.getElementById("fileInput");
const uploadText = document.getElementById("uploadText");
const uploadLabel = document.querySelector(".upload-label");
const generateBtn = document.getElementById("generateBtn");
const generateLabel = document.getElementById("generateLabel");
const generateSpinner = document.getElementById("generateSpinner");
const resultsSection = document.getElementById("resultsSection");
const roughText = document.getElementById("roughText");
const flowStats = document.getElementById("flowStats");
const versionsContainer = document.getElementById("versionsContainer");

/* ── Recording — click to start, click to stop ── */
recordBtn.addEventListener("click", () => {
  if (isRecording) {
    stopRecording();
  } else {
    startRecording();
  }
});

async function startRecording() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    audioChunks = [];
    mediaRecorder = new MediaRecorder(stream, { mimeType: getSupportedMimeType() });

    mediaRecorder.ondataavailable = (e) => {
      if (e.data.size > 0) audioChunks.push(e.data);
    };

    mediaRecorder.onstop = () => {
      const mimeType = getSupportedMimeType();
      recordedBlob = new Blob(audioChunks, { type: mimeType });
      uploadedFile = null;

      const url = URL.createObjectURL(recordedBlob);
      audioPreview.src = url;
      audioPreview.classList.remove("hidden");
      recordBtn.classList.remove("recording");
      recordBtn.classList.add("has-audio");
      recordLabel.textContent = "Record Again";
      recordingStatus.classList.add("hidden");
      uploadText.textContent = "Upload Audio File";
      uploadLabel.classList.remove("has-file");

      stream.getTracks().forEach((t) => t.stop());
      checkReady();
    };

    mediaRecorder.start();
    isRecording = true;
    recordedBlob = null;
    recordBtn.classList.remove("has-audio");
    recordBtn.classList.add("recording");
    recordLabel.textContent = "Stop Recording";
    recordingStatus.classList.remove("hidden");

    // Start timer
    recordingSeconds = 0;
    updateTimer();
    timerInterval = setInterval(updateTimer, 1000);

  } catch (err) {
    alert("Microphone access denied. Please allow mic access and try again.");
  }
}

function stopRecording() {
  if (!isRecording || !mediaRecorder) return;
  mediaRecorder.stop();
  isRecording = false;
  clearInterval(timerInterval);
  timerInterval = null;
}

function updateTimer() {
  const m = Math.floor(recordingSeconds / 60).toString().padStart(2, "0");
  const s = (recordingSeconds % 60).toString().padStart(2, "0");
  recordTimer.textContent = `${m}:${s}`;
  recordingSeconds++;
}

function getSupportedMimeType() {
  const types = ["audio/webm;codecs=opus", "audio/webm", "audio/ogg;codecs=opus", "audio/mp4"];
  for (const t of types) {
    if (MediaRecorder.isTypeSupported(t)) return t;
  }
  return "";
}

/* ── File Upload ── */
fileInput.addEventListener("change", () => {
  const file = fileInput.files[0];
  if (!file) return;
  uploadedFile = file;
  recordedBlob = null;
  uploadText.textContent = `✓ ${file.name}`;
  uploadLabel.classList.add("has-file");

  const url = URL.createObjectURL(file);
  audioPreview.src = url;
  audioPreview.classList.remove("hidden");
  recordBtn.classList.remove("has-audio", "recording");
  recordLabel.textContent = "Start Recording";

  checkReady();
});

/* ── Style Pills ── */
document.querySelectorAll(".pill-group").forEach((group) => {
  group.querySelectorAll(".pill").forEach((pill) => {
    pill.addEventListener("click", () => {
      group.querySelectorAll(".pill").forEach((p) => p.classList.remove("active"));
      pill.classList.add("active");
      state[group.dataset.group] = pill.dataset.value;
    });
  });
});

/* ── Ready Check ── */
function checkReady() {
  generateBtn.disabled = !(recordedBlob || uploadedFile);
}

/* ── Generate ── */
generateBtn.addEventListener("click", async () => {
  const audioSource = recordedBlob || uploadedFile;
  if (!audioSource) return;

  setGenerating(true);

  const formData = new FormData();

  if (recordedBlob) {
    const ext = getExtFromMime(recordedBlob.type);
    formData.append("audio", recordedBlob, `recording.${ext}`);
  } else {
    formData.append("audio", uploadedFile, uploadedFile.name);
  }

  formData.append("tone", state.tone);
  formData.append("mode", state.mode);
  formData.append("vibe", state.vibe);

  try {
    const res = await fetch("/process", { method: "POST", body: formData });
    const data = await res.json();

    if (!res.ok || !data.success) {
      throw new Error(data.error || "Something went wrong");
    }

    renderResults(data);
  } catch (err) {
    alert(`Error: ${err.message}`);
  } finally {
    setGenerating(false);
  }
});

function getExtFromMime(mimeType) {
  if (mimeType.includes("webm")) return "webm";
  if (mimeType.includes("ogg")) return "ogg";
  if (mimeType.includes("mp4")) return "mp4";
  return "webm";
}

function setGenerating(loading) {
  generateBtn.disabled = loading;
  generateBtn.classList.toggle("loading", loading);
  generateLabel.textContent = loading ? "Generating..." : "Generate Lyrics";
  generateSpinner.classList.toggle("hidden", !loading);
}

/* ── Render Results ── */
function renderResults(data) {
  resultsSection.classList.remove("hidden");

  roughText.textContent = data.rough_text || "(melody only — no words detected)";

  const flow = data.flow || {};
  const modeLabel = data.melody_mode ? ' · <strong style="color:#a78bfa">melody mode</strong>' : "";
  flowStats.innerHTML = [
    flow.tempo_bpm ? `<div class="stat-chip"><strong>${Math.round(flow.tempo_bpm)}</strong> BPM</div>` : "",
    flow.flow_style ? `<div class="stat-chip">Flow: <strong>${flow.flow_style}</strong></div>` : "",
    flow.syllable_count ? `<div class="stat-chip"><strong>~${flow.syllable_count}</strong> syllables</div>` : "",
  ].join("") + (data.melody_mode ? `<div class="stat-chip" style="border-color:#7c3aed;color:#a78bfa"><strong>melody mode</strong></div>` : "");

  versionsContainer.innerHTML = "";
  (data.versions || []).forEach((v, idx) => {
    const card = createVersionCard(v, idx === 0);
    versionsContainer.appendChild(card);
  });

  setTimeout(() => resultsSection.scrollIntoView({ behavior: "smooth", block: "start" }), 100);
}

function createVersionCard(version, isBest) {
  const card = document.createElement("div");
  card.className = "version-card";

  const scorePercent = Math.round((version.score || 0) * 100);

  card.innerHTML = `
    <div class="version-header">
      <span class="version-name">${version.label || version.name}</span>
      <div class="version-badge">
        ${isBest ? '<span class="best-label">⭐ Best Match</span>' : ""}
        <div class="score-bar-wrap">
          <div class="score-bar" style="width: ${scorePercent}%"></div>
        </div>
      </div>
    </div>
    <pre class="version-lyrics">${escapeHtml(version.lyrics || "")}</pre>
    <button class="copy-btn" data-lyrics="${encodeURIComponent(version.lyrics || "")}">Copy Lyrics</button>
  `;

  card.querySelector(".copy-btn").addEventListener("click", function () {
    const lyrics = decodeURIComponent(this.dataset.lyrics);
    navigator.clipboard.writeText(lyrics).then(() => {
      this.textContent = "✓ Copied!";
      this.classList.add("copied");
      setTimeout(() => {
        this.textContent = "Copy Lyrics";
        this.classList.remove("copied");
      }, 2000);
    });
  });

  return card;
}

function escapeHtml(str) {
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}
