/* ── State ── */
let mediaRecorder = null;
let audioChunks = [];
let recordedBlob = null;
let uploadedFile = null;
let isRecording = false;
let timerInterval = null;
let recordingSeconds = 0;

let currentPhraseMap = [];
let audioDuration = 0;
let syncInterval = null;
let activeSyncCard = null;
let activeWordTimings = []; // flat array of {el, start, end} for word-level sync

const state = { tone: "melodic", mode: "verse", vibe: "introspective" };

/* ── DOM refs ── */
const recordBtn        = document.getElementById("recordBtn");
const recordLabel      = document.getElementById("recordLabel");
const recordingStatus  = document.getElementById("recordingStatus");
const recordTimer      = document.getElementById("recordTimer");
const audioPreview     = document.getElementById("audioPreview");
const fileInput        = document.getElementById("fileInput");
const uploadText       = document.getElementById("uploadText");
const uploadLabel      = document.querySelector(".upload-label");
const generateBtn      = document.getElementById("generateBtn");
const generateLabel    = document.getElementById("generateLabel");
const generateSpinner  = document.getElementById("generateSpinner");
const resultsSection   = document.getElementById("resultsSection");
const roughTextEl      = document.getElementById("roughText");
const flowStats        = document.getElementById("flowStats");
const versionsContainer = document.getElementById("versionsContainer");
const timelineWrap     = document.getElementById("timelineWrap");
const timelineTrack    = document.getElementById("timelineTrack");
const timelinePlayhead = document.getElementById("timelinePlayhead");
const timelinePhrases  = document.getElementById("timelinePhrases");
const melodyBadge      = document.getElementById("melodyBadge");

/* ══════════════════════════════════
   RECORDING
   ══════════════════════════════════ */
recordBtn.addEventListener("click", () => isRecording ? stopRecording() : startRecording());

async function startRecording() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    audioChunks = [];
    mediaRecorder = new MediaRecorder(stream, { mimeType: getSupportedMimeType() });
    mediaRecorder.ondataavailable = (e) => { if (e.data.size > 0) audioChunks.push(e.data); };
    mediaRecorder.onstop = () => {
      recordedBlob = new Blob(audioChunks, { type: getSupportedMimeType() });
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
  for (const t of ["audio/webm;codecs=opus","audio/webm","audio/ogg;codecs=opus","audio/mp4"]) {
    if (MediaRecorder.isTypeSupported(t)) return t;
  }
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

function checkReady() { generateBtn.disabled = !(recordedBlob || uploadedFile); }

/* ══════════════════
   GENERATE
   ══════════════════ */
generateBtn.addEventListener("click", async () => {
  if (!(recordedBlob || uploadedFile)) return;
  setGenerating(true);
  const formData = new FormData();
  if (recordedBlob) {
    const ext = recordedBlob.type.includes("ogg") ? "ogg" : recordedBlob.type.includes("mp4") ? "mp4" : "webm";
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
  stopSync();
  resultsSection.classList.remove("hidden");

  currentPhraseMap = data.phrase_map || [];
  audioDuration = data.flow?.duration || 0;

  // Organized transcript — bars with syllable counts
  renderOrganizedTranscript(data.rough_text || "", currentPhraseMap, data.melody_mode);

  // Melody badge
  melodyBadge.classList.toggle("hidden", !data.melody_mode);

  // Timeline
  renderTimeline(currentPhraseMap, audioDuration);

  // Flow stats
  const flow = data.flow || {};
  flowStats.innerHTML = [
    flow.tempo_bpm    ? `<div class="stat-chip"><strong>${Math.round(flow.tempo_bpm)}</strong> BPM</div>` : "",
    flow.flow_style   ? `<div class="stat-chip">Flow: <strong>${flow.flow_style}</strong></div>` : "",
    flow.syllable_count ? `<div class="stat-chip"><strong>~${flow.syllable_count}</strong> syllables</div>` : "",
    data.melody_mode  ? `<div class="stat-chip melody-chip"><strong>melody mode</strong></div>` : "",
  ].join("");

  versionsContainer.innerHTML = "";
  (data.versions || []).forEach((v, idx) => {
    versionsContainer.appendChild(createVersionCard(v, idx === 0, currentPhraseMap));
  });

  setTimeout(() => resultsSection.scrollIntoView({ behavior: "smooth", block: "start" }), 100);
}

/* ══════════════════════════════════════
   ORGANIZED TRANSCRIPT
   Shows bars as individual lines with syllable count badges
   ══════════════════════════════════════ */
function renderOrganizedTranscript(roughText, phraseMap, melodyMode) {
  if (!phraseMap.length || melodyMode) {
    roughTextEl.innerHTML = `<span class="rough-plain">${escapeHtml(roughText || "(melody — no words detected)")}</span>`;
    return;
  }

  const bars = phraseMap.map((phrase, i) => {
    const text = phrase.text || "";
    const syls = phrase.syllables || 0;
    return `<div class="transcript-bar" data-index="${i}">
      <span class="bar-num">${i + 1}</span>
      <span class="bar-text">${escapeHtml(text)}</span>
      <span class="bar-syls">${syls} syl</span>
    </div>`;
  }).join("");

  roughTextEl.innerHTML = bars;

  // Click a bar to seek audio
  roughTextEl.querySelectorAll(".transcript-bar").forEach((bar) => {
    bar.addEventListener("click", () => {
      const idx = parseInt(bar.dataset.index);
      if (currentPhraseMap[idx]) {
        audioPreview.currentTime = currentPhraseMap[idx].start_time;
        audioPreview.play();
      }
    });
  });
}

/* ══════════════════════════════════════
   PHRASE TIMELINE
   ══════════════════════════════════════ */
function renderTimeline(phraseMap, duration) {
  timelinePhrases.innerHTML = "";
  timelinePlayhead.style.left = "0%";

  // Clear old blocks from track (keep playhead)
  Array.from(timelineTrack.children).forEach((c) => {
    if (!c.classList.contains("timeline-playhead")) c.remove();
  });

  if (!phraseMap.length || !duration) {
    timelineWrap.classList.add("hidden");
    return;
  }
  timelineWrap.classList.remove("hidden");

  phraseMap.forEach((phrase, idx) => {
    const startPct = (phrase.start_time / duration) * 100;
    const endTime = phrase.end_time || duration;
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

    const label = document.createElement("div");
    label.className = "timeline-phrase-label";
    label.style.left = `${startPct}%`;
    label.style.width = `${widthPct}%`;
    label.innerHTML =
      `<span class="tl-time">${formatTime(phrase.start_time)}</span>` +
      (phrase.text
        ? `<span class="tl-text">${escapeHtml(phrase.text)}</span>`
        : `<span class="tl-text muted">phrase ${idx + 1}</span>`);
    timelinePhrases.appendChild(label);
  });

  audioPreview.removeEventListener("timeupdate", onTimelineUpdate);
  audioPreview.addEventListener("timeupdate", onTimelineUpdate);
}

function onTimelineUpdate() {
  if (!audioDuration) return;
  const pct = (audioPreview.currentTime / audioDuration) * 100;
  timelinePlayhead.style.left = `${Math.min(pct, 100)}%`;

  const idx = getCurrentPhraseIndex(audioPreview.currentTime);
  document.querySelectorAll(".timeline-block").forEach((b, i) => {
    b.classList.toggle("active", i === idx);
  });

  // Highlight corresponding transcript bar
  document.querySelectorAll(".transcript-bar").forEach((b, i) => {
    b.classList.toggle("bar-active", i === idx);
  });
}

/* ══════════════════════════════════════
   VERSION CARDS WITH WORD-LEVEL KARAOKE
   ══════════════════════════════════════ */
function createVersionCard(version, isBest, phraseMap) {
  const card = document.createElement("div");
  card.className = "version-card";

  const scorePercent = Math.round((version.score || 0) * 100);
  const lines = (version.lyrics || "").split("\n").filter((l) => l.trim());

  // Build word-span lyric lines
  const lyricsHtml = lines.map((line, lineIdx) => {
    const words = line.trim().split(/\s+/);
    const wordSpans = words.map((w, wi) =>
      `<span class="lyric-word" data-line="${lineIdx}" data-word="${wi}">${escapeHtml(w)}</span>`
    ).join(" ");
    return `<div class="lyric-line" data-line="${lineIdx}">${wordSpans}</div>`;
  }).join("");

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

  card.querySelector(".copy-btn").addEventListener("click", function () {
    navigator.clipboard.writeText(decodeURIComponent(this.dataset.lyrics)).then(() => {
      this.textContent = "✓ Copied!";
      this.classList.add("copied");
      setTimeout(() => { this.textContent = "Copy"; this.classList.remove("copied"); }, 2000);
    });
  });

  const syncBtn = card.querySelector(".sync-btn");
  if (syncBtn) {
    syncBtn.addEventListener("click", () => {
      activeSyncCard === card ? stopSync() : startSync(card, syncBtn, lines, phraseMap);
    });
  }

  return card;
}

/* ══════════════════════════════════════
   WORD-LEVEL KARAOKE SYNC
   ══════════════════════════════════════ */
function buildWordTimings(lines, phraseMap, totalDuration) {
  /**
   * For each lyric line (= one phrase), interpolate per-word timings
   * by dividing the phrase duration evenly across its words.
   */
  const timings = [];

  lines.forEach((line, lineIdx) => {
    const phrase = phraseMap[lineIdx];
    if (!phrase) return;

    const phraseStart = phrase.start_time;
    const phraseEnd = phrase.end_time || (phraseMap[lineIdx + 1]?.start_time) || totalDuration;
    const phraseDuration = Math.max(0.1, phraseEnd - phraseStart);

    const words = line.trim().split(/\s+/);
    const wordDuration = phraseDuration / words.length;

    words.forEach((_, wi) => {
      timings.push({
        lineIdx,
        wordIdx: wi,
        start: phraseStart + wi * wordDuration,
        end: phraseStart + (wi + 1) * wordDuration,
      });
    });
  });

  return timings;
}

function startSync(card, btn, lines, phraseMap) {
  stopSync();
  activeSyncCard = card;
  btn.textContent = "■ Stop Sync";
  btn.classList.add("syncing");

  // Build word-level timing map
  activeWordTimings = buildWordTimings(lines, phraseMap, audioDuration);

  audioPreview.currentTime = 0;
  audioPreview.play();

  syncInterval = setInterval(() => {
    const t = audioPreview.currentTime;

    // Find current word
    let currentTiming = null;
    for (const timing of activeWordTimings) {
      if (t >= timing.start && t < timing.end) {
        currentTiming = timing;
        break;
      }
    }

    // Apply highlights
    card.querySelectorAll(".lyric-line").forEach((lineEl) => {
      const li = parseInt(lineEl.dataset.line);
      const isCurrentLine = currentTiming && li === currentTiming.lineIdx;
      const isPastLine = currentTiming ? li < currentTiming.lineIdx : false;
      lineEl.classList.toggle("past-line", isPastLine);
      lineEl.classList.toggle("current-line", isCurrentLine);
    });

    card.querySelectorAll(".lyric-word").forEach((wordEl) => {
      const li = parseInt(wordEl.dataset.line);
      const wi = parseInt(wordEl.dataset.word);
      const isActive = currentTiming && li === currentTiming.lineIdx && wi === currentTiming.wordIdx;
      const isPast = currentTiming
        ? (li < currentTiming.lineIdx || (li === currentTiming.lineIdx && wi < currentTiming.wordIdx))
        : false;
      wordEl.classList.toggle("word-active", isActive);
      wordEl.classList.toggle("word-past", isPast);
    });

    // Auto-scroll active line into view
    const activeLine = card.querySelector(".lyric-line.current-line");
    if (activeLine) activeLine.scrollIntoView({ block: "nearest", behavior: "smooth" });

    if (audioPreview.ended || audioPreview.paused) stopSync();
  }, 50); // 50ms for smooth word tracking

  audioPreview.addEventListener("ended", stopSync, { once: true });
}

function stopSync() {
  clearInterval(syncInterval);
  syncInterval = null;
  activeWordTimings = [];

  if (activeSyncCard) {
    activeSyncCard.querySelectorAll(".lyric-line, .lyric-word").forEach((el) => {
      el.classList.remove("current-line", "past-line", "word-active", "word-past");
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
    .replace(/&/g, "&amp;").replace(/</g, "&lt;")
    .replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}
