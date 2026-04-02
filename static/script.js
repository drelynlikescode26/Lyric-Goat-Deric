/* ── State ── */
let mediaRecorder = null;
let audioChunks = [];
let recordedBlob = null;
let uploadedFile = null;
let isRecording = false;
let timerInterval = null;
let recordingSeconds = 0;

// Result state
let currentPhraseMap = [];
let audioDuration = 0;
let syncInterval = null;
let activeSyncCard = null;

const state = {
  tone: "melodic",
  mode: "verse",
  vibe: "introspective",
};

/* ── DOM refs ── */
const recordBtn       = document.getElementById("recordBtn");
const recordLabel     = document.getElementById("recordLabel");
const recordingStatus = document.getElementById("recordingStatus");
const recordTimer     = document.getElementById("recordTimer");
const audioPreview    = document.getElementById("audioPreview");
const fileInput       = document.getElementById("fileInput");
const uploadText      = document.getElementById("uploadText");
const uploadLabel     = document.querySelector(".upload-label");
const generateBtn     = document.getElementById("generateBtn");
const generateLabel   = document.getElementById("generateLabel");
const generateSpinner = document.getElementById("generateSpinner");
const resultsSection  = document.getElementById("resultsSection");
const roughText       = document.getElementById("roughText");
const flowStats       = document.getElementById("flowStats");
const versionsContainer = document.getElementById("versionsContainer");
const timelineWrap    = document.getElementById("timelineWrap");
const timelineTrack   = document.getElementById("timelineTrack");
const timelinePlayhead = document.getElementById("timelinePlayhead");
const timelinePhrases = document.getElementById("timelinePhrases");
const melodyBadge     = document.getElementById("melodyBadge");

/* ══════════════════════════════════
   RECORDING — click to start / stop
   ══════════════════════════════════ */
recordBtn.addEventListener("click", () => {
  if (isRecording) stopRecording();
  else startRecording();
});

async function startRecording() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    audioChunks = [];
    mediaRecorder = new MediaRecorder(stream, { mimeType: getSupportedMimeType() });

    mediaRecorder.ondataavailable = (e) => { if (e.data.size > 0) audioChunks.push(e.data); };
    mediaRecorder.onstop = () => {
      const mimeType = getSupportedMimeType();
      recordedBlob = new Blob(audioChunks, { type: mimeType });
      uploadedFile = null;

      audioPreview.src = URL.createObjectURL(recordedBlob);
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
  for (const t of types) { if (MediaRecorder.isTypeSupported(t)) return t; }
  return "";
}

/* ══════════════════
   FILE UPLOAD
   ══════════════════ */
fileInput.addEventListener("change", () => {
  const file = fileInput.files[0];
  if (!file) return;
  uploadedFile = file;
  recordedBlob = null;
  uploadText.textContent = `✓ ${file.name}`;
  uploadLabel.classList.add("has-file");
  audioPreview.src = URL.createObjectURL(file);
  audioPreview.classList.remove("hidden");
  recordBtn.classList.remove("has-audio", "recording");
  recordLabel.textContent = "Start Recording";
  checkReady();
});

/* ══════════════════
   STYLE PILLS
   ══════════════════ */
document.querySelectorAll(".pill-group").forEach((group) => {
  group.querySelectorAll(".pill").forEach((pill) => {
    pill.addEventListener("click", () => {
      group.querySelectorAll(".pill").forEach((p) => p.classList.remove("active"));
      pill.classList.add("active");
      state[group.dataset.group] = pill.dataset.value;
    });
  });
});

function checkReady() {
  generateBtn.disabled = !(recordedBlob || uploadedFile);
}

/* ══════════════════
   GENERATE
   ══════════════════ */
generateBtn.addEventListener("click", async () => {
  if (!(recordedBlob || uploadedFile)) return;
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
    if (!res.ok || !data.success) throw new Error(data.error || "Something went wrong");
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

/* ══════════════════════════════════════
   RENDER RESULTS
   ══════════════════════════════════════ */
function renderResults(data) {
  // Stop any previous sync
  stopSync();

  resultsSection.classList.remove("hidden");

  // Store phrase map + duration for karaoke sync
  currentPhraseMap = data.phrase_map || [];
  audioDuration = data.flow?.duration || 0;

  // Rough text
  roughText.textContent = data.rough_text || "(melody only — no words detected)";

  // Melody badge
  if (data.melody_mode) {
    melodyBadge.classList.remove("hidden");
  } else {
    melodyBadge.classList.add("hidden");
  }

  // Phrase timeline
  renderTimeline(currentPhraseMap, audioDuration);

  // Flow stats
  const flow = data.flow || {};
  flowStats.innerHTML = [
    flow.tempo_bpm   ? `<div class="stat-chip"><strong>${Math.round(flow.tempo_bpm)}</strong> BPM</div>` : "",
    flow.flow_style  ? `<div class="stat-chip">Flow: <strong>${flow.flow_style}</strong></div>` : "",
    flow.syllable_count ? `<div class="stat-chip"><strong>~${flow.syllable_count}</strong> syllables</div>` : "",
    data.melody_mode ? `<div class="stat-chip melody-chip"><strong>melody mode</strong></div>` : "",
  ].join("");

  // Lyric version cards
  versionsContainer.innerHTML = "";
  (data.versions || []).forEach((v, idx) => {
    versionsContainer.appendChild(createVersionCard(v, idx === 0, currentPhraseMap));
  });

  setTimeout(() => resultsSection.scrollIntoView({ behavior: "smooth", block: "start" }), 100);
}

/* ══════════════════════════════════════
   PHRASE TIMELINE
   ══════════════════════════════════════ */
function renderTimeline(phraseMap, duration) {
  timelinePhrases.innerHTML = "";
  timelinePlayhead.style.left = "0%";

  if (!phraseMap.length || !duration) {
    timelineWrap.classList.add("hidden");
    return;
  }
  timelineWrap.classList.remove("hidden");

  // Draw phrase blocks on the track
  phraseMap.forEach((phrase, idx) => {
    const startPct = (phrase.start_time / duration) * 100;
    // Estimate end time: next phrase start or audio end
    const endTime = (phraseMap[idx + 1]?.start_time) || duration;
    const widthPct = Math.max(1, ((endTime - phrase.start_time) / duration) * 100);

    const block = document.createElement("div");
    block.className = "timeline-block";
    block.style.left = `${startPct}%`;
    block.style.width = `${widthPct}%`;
    block.dataset.index = idx;
    block.title = phrase.text || `phrase ${idx + 1}`;

    block.addEventListener("click", () => {
      audioPreview.currentTime = phrase.start_time;
      audioPreview.play();
    });

    timelineTrack.appendChild(block);

    // Phrase label below track
    const label = document.createElement("div");
    label.className = "timeline-phrase-label";
    label.style.left = `${startPct}%`;
    label.style.width = `${widthPct}%`;
    label.innerHTML = `<span class="tl-time">${formatTime(phrase.start_time)}</span>` +
      (phrase.text ? `<span class="tl-text">${escapeHtml(phrase.text)}</span>` : `<span class="tl-text muted">phrase ${idx + 1}</span>`);
    timelinePhrases.appendChild(label);
  });

  // Update playhead on audio timeupdate
  audioPreview.removeEventListener("timeupdate", onTimelineUpdate);
  audioPreview.addEventListener("timeupdate", onTimelineUpdate);
}

function onTimelineUpdate() {
  if (!audioDuration) return;
  const pct = (audioPreview.currentTime / audioDuration) * 100;
  timelinePlayhead.style.left = `${Math.min(pct, 100)}%`;

  // Highlight the active phrase block
  const idx = getCurrentPhraseIndex(audioPreview.currentTime);
  document.querySelectorAll(".timeline-block").forEach((b, i) => {
    b.classList.toggle("active", i === idx);
  });
}

/* ══════════════════════════════════════
   VERSION CARDS WITH KARAOKE SYNC
   ══════════════════════════════════════ */
function createVersionCard(version, isBest, phraseMap) {
  const card = document.createElement("div");
  card.className = "version-card";

  const scorePercent = Math.round((version.score || 0) * 100);
  const lines = (version.lyrics || "").split("\n").filter((l) => l.trim());

  // Build karaoke line HTML
  const lyricsHtml = lines.map((line, i) =>
    `<div class="lyric-line" data-line="${i}">${escapeHtml(line)}</div>`
  ).join("");

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
    <div class="version-lyrics karaoke-lyrics">${lyricsHtml}</div>
    <div class="card-actions">
      <button class="copy-btn" data-lyrics="${encodeURIComponent(version.lyrics || "")}">Copy</button>
      ${phraseMap.length ? '<button class="sync-btn">▶ Play Synced</button>' : ""}
    </div>
  `;

  // Copy button
  card.querySelector(".copy-btn").addEventListener("click", function () {
    navigator.clipboard.writeText(decodeURIComponent(this.dataset.lyrics)).then(() => {
      this.textContent = "✓ Copied!";
      this.classList.add("copied");
      setTimeout(() => { this.textContent = "Copy"; this.classList.remove("copied"); }, 2000);
    });
  });

  // Sync button
  const syncBtn = card.querySelector(".sync-btn");
  if (syncBtn) {
    syncBtn.addEventListener("click", () => {
      if (activeSyncCard === card) {
        stopSync();
      } else {
        startSync(card, syncBtn);
      }
    });
  }

  return card;
}

/* ══════════════════════════════════════
   KARAOKE SYNC
   ══════════════════════════════════════ */
function startSync(card, btn) {
  stopSync();
  activeSyncCard = card;
  btn.textContent = "■ Stop Sync";
  btn.classList.add("syncing");

  audioPreview.currentTime = 0;
  audioPreview.play();

  syncInterval = setInterval(() => {
    const currentTime = audioPreview.currentTime;
    const idx = getCurrentPhraseIndex(currentTime);

    card.querySelectorAll(".lyric-line").forEach((el, i) => {
      el.classList.toggle("active-line", i === idx);
      el.classList.toggle("past-line", i < idx);
    });

    // Auto scroll to active line
    const activeLine = card.querySelector(".lyric-line.active-line");
    if (activeLine) {
      activeLine.scrollIntoView({ block: "nearest", behavior: "smooth" });
    }

    // Stop when audio ends
    if (audioPreview.ended || audioPreview.paused) {
      stopSync();
    }
  }, 100);

  audioPreview.addEventListener("ended", stopSync, { once: true });
}

function stopSync() {
  clearInterval(syncInterval);
  syncInterval = null;

  if (activeSyncCard) {
    activeSyncCard.querySelectorAll(".lyric-line").forEach((el) => {
      el.classList.remove("active-line", "past-line");
    });
    const btn = activeSyncCard.querySelector(".sync-btn");
    if (btn) { btn.textContent = "▶ Play Synced"; btn.classList.remove("syncing"); }
    activeSyncCard = null;
  }

  audioPreview.pause();
}

function getCurrentPhraseIndex(currentTime) {
  if (!currentPhraseMap.length) return -1;
  let idx = 0;
  for (let i = 0; i < currentPhraseMap.length; i++) {
    if (currentTime >= currentPhraseMap[i].start_time) idx = i;
    else break;
  }
  return idx;
}

/* ══════════════════
   UTILITIES
   ══════════════════ */
function formatTime(seconds) {
  const m = Math.floor(seconds / 60).toString().padStart(2, "0");
  const s = Math.floor(seconds % 60).toString().padStart(2, "0");
  return `${m}:${s}`;
}

function escapeHtml(str) {
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}
