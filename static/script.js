/* ── State ── */
let mediaRecorder = null;
let audioChunks = [];
let recordedBlob = null;
let uploadedFile = null;
let isRecording = false;

const state = {
  tone: "melodic",
  mode: "verse",
  vibe: "introspective",
};

/* ── DOM refs ── */
const recordBtn = document.getElementById("recordBtn");
const recordLabel = document.getElementById("recordLabel");
const recordingStatus = document.getElementById("recordingStatus");
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

/* ── Recording ── */
recordBtn.addEventListener("mousedown", startRecording);
recordBtn.addEventListener("touchstart", (e) => { e.preventDefault(); startRecording(); });
recordBtn.addEventListener("mouseup", stopRecording);
recordBtn.addEventListener("mouseleave", () => { if (isRecording) stopRecording(); });
recordBtn.addEventListener("touchend", stopRecording);

async function startRecording() {
  if (isRecording) return;
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
      recordBtn.classList.add("has-audio");
      recordLabel.textContent = "Re-record";
      uploadText.textContent = "Upload Audio File";
      uploadLabel.classList.remove("has-file");

      stream.getTracks().forEach((t) => t.stop());
      checkReady();
    };

    mediaRecorder.start();
    isRecording = true;
    recordBtn.classList.remove("has-audio");
    recordBtn.classList.add("recording");
    recordLabel.textContent = "Release to Stop";
    recordingStatus.classList.remove("hidden");
  } catch (err) {
    alert("Microphone access denied. Please allow mic access and try again.");
  }
}

function stopRecording() {
  if (!isRecording || !mediaRecorder) return;
  mediaRecorder.stop();
  isRecording = false;
  recordBtn.classList.remove("recording");
  recordingStatus.classList.add("hidden");
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

  // Show preview
  const url = URL.createObjectURL(file);
  audioPreview.src = url;
  audioPreview.classList.remove("hidden");
  recordBtn.classList.remove("has-audio", "recording");
  recordLabel.textContent = "Hold to Record";

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

  // Rough text
  roughText.textContent = data.rough_text || "";

  // Flow stats
  const flow = data.flow || {};
  flowStats.innerHTML = [
    flow.tempo_bpm ? `<div class="stat-chip"><strong>${Math.round(flow.tempo_bpm)}</strong> BPM</div>` : "",
    flow.flow_style ? `<div class="stat-chip">Flow: <strong>${flow.flow_style}</strong></div>` : "",
    flow.syllable_count ? `<div class="stat-chip"><strong>~${flow.syllable_count}</strong> syllables</div>` : "",
  ].join("");

  // Versions
  versionsContainer.innerHTML = "";
  (data.versions || []).forEach((v, idx) => {
    const card = createVersionCard(v, idx === 0);
    versionsContainer.appendChild(card);
  });

  // Scroll to results
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
