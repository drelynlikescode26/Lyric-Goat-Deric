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
let currentRoughText = "";
let currentFlowData = {};

// Per-version lock state: { versionName: { lineIdx: lockedText } }
let lockedLines = {};

const state = {
  tone: "melodic",
  mode: "verse",
  vibe: "introspective",
  gen_mode: "cadence",
  key: "auto",
  keyQuality: "major",
};

/* ── DOM refs ── */
const recordBtn          = document.getElementById("recordBtn");
const recordLabel        = document.getElementById("recordLabel");
const recordingStatus    = document.getElementById("recordingStatus");
const recordTimer        = document.getElementById("recordTimer");
const audioPreview       = document.getElementById("audioPreview");
const fileInput          = document.getElementById("fileInput");
const uploadText         = document.getElementById("uploadText");
const uploadLabel        = document.querySelector(".upload-label");
const generateBtn        = document.getElementById("generateBtn");
const generateLabel      = document.getElementById("generateLabel");
const generateSpinner    = document.getElementById("generateSpinner");
const resultsSection     = document.getElementById("resultsSection");
const roughTextEl        = document.getElementById("roughText");
const analysisChipsEl    = document.getElementById("analysisChips");
const detectedKeyBadge   = document.getElementById("detectedKeyBadge");
const repetitiveBadge    = document.getElementById("repetitiveBadge");
const flowStats        = document.getElementById("flowStats");
const versionsContainer = document.getElementById("versionsContainer");
const timelineWrap     = document.getElementById("timelineWrap");
const timelineTrack    = document.getElementById("timelineTrack");
const timelinePlayhead = document.getElementById("timelinePlayhead");
const timelinePhrases  = document.getElementById("timelinePhrases");
const melodyBadge      = document.getElementById("melodyBadge");

/* ══════════════════════════════════
   METRONOME ENGINE (Web Audio API)
   Uses a look-ahead scheduler for
   sample-accurate click timing.
   ══════════════════════════════════ */
let metroCtx = null;
let metroRunning = false;
let metroBpm = 90;
let nextBeatTime = 0;
let beatCount = 0;
let metroSchedulerTimer = null;
const LOOKAHEAD_MS = 25;      // scheduler interval
const SCHEDULE_AHEAD_S = 0.1; // how far ahead to schedule

const metroToggle  = document.getElementById("metroToggle");
const metroIcon    = document.getElementById("metroIcon");
const metroLabel   = document.getElementById("metroLabel");
const bpmValueEl   = document.getElementById("bpmValue");
const bpmDownBtn   = document.getElementById("bpmDown");
const bpmUpBtn     = document.getElementById("bpmUp");
const beatDots     = document.querySelectorAll(".beat-dot");

// BPM +/−
bpmDownBtn.addEventListener("click", () => setBpm(metroBpm - 1));
bpmUpBtn.addEventListener("click",   () => setBpm(metroBpm + 1));

// BPM presets
document.querySelectorAll(".bpm-preset").forEach((btn) => {
  btn.addEventListener("click", () => setBpm(parseInt(btn.dataset.bpm)));
});

// Hold +/− for fast scroll
let bpmHoldTimer = null;
function startBpmHold(delta) {
  bpmHoldTimer = setInterval(() => setBpm(metroBpm + delta), 120);
}
function stopBpmHold() { clearInterval(bpmHoldTimer); bpmHoldTimer = null; }
bpmDownBtn.addEventListener("mousedown", () => startBpmHold(-1));
bpmUpBtn.addEventListener("mousedown",   () => startBpmHold(1));
["mouseup","mouseleave"].forEach((e) => {
  bpmDownBtn.addEventListener(e, stopBpmHold);
  bpmUpBtn.addEventListener(e, stopBpmHold);
});

function setBpm(val) {
  metroBpm = Math.min(220, Math.max(40, val));
  bpmValueEl.textContent = metroBpm;
  document.querySelectorAll(".bpm-preset").forEach((b) => {
    b.classList.toggle("active", parseInt(b.dataset.bpm) === metroBpm);
  });
}

// Toggle metronome on/off
metroToggle.addEventListener("click", () => {
  metroRunning ? stopMetronome() : startMetronome();
});

function startMetronome() {
  if (metroRunning) return;
  metroCtx = new (window.AudioContext || window.webkitAudioContext)();
  metroRunning = true;
  beatCount = 0;
  nextBeatTime = metroCtx.currentTime + 0.05;
  metroIcon.textContent = "■";
  metroLabel.textContent = "Stop";
  metroToggle.classList.add("metro-active");
  scheduleBeats();
}

function stopMetronome() {
  metroRunning = false;
  clearTimeout(metroSchedulerTimer);
  metroSchedulerTimer = null;
  if (metroCtx) { metroCtx.close(); metroCtx = null; }
  metroIcon.textContent = "▶";
  metroLabel.textContent = "Start";
  metroToggle.classList.remove("metro-active");
  beatDots.forEach((d) => d.classList.remove("beat-on", "beat-accent"));
}

function scheduleBeats() {
  if (!metroRunning) return;

  while (nextBeatTime < metroCtx.currentTime + SCHEDULE_AHEAD_S) {
    const isAccent = beatCount % 4 === 0;
    scheduleClick(nextBeatTime, isAccent);
    scheduleVisualBeat(nextBeatTime, beatCount % 4);
    nextBeatTime += 60 / metroBpm;
    beatCount++;
  }

  metroSchedulerTimer = setTimeout(scheduleBeats, LOOKAHEAD_MS);
}

function scheduleClick(time, isAccent) {
  const osc  = metroCtx.createOscillator();
  const gain = metroCtx.createGain();
  osc.connect(gain);
  gain.connect(metroCtx.destination);

  // Accent beat (beat 1) = higher pitch, louder
  osc.frequency.value = isAccent ? 1200 : 900;
  gain.gain.setValueAtTime(isAccent ? 0.6 : 0.35, time);
  gain.gain.exponentialRampToValueAtTime(0.0001, time + 0.04);

  osc.start(time);
  osc.stop(time + 0.04);
}

function scheduleVisualBeat(time, dotIndex) {
  // Schedule visual update using currentTime offset
  const delay = Math.max(0, (time - metroCtx.currentTime) * 1000);
  setTimeout(() => {
    if (!metroRunning) return;
    beatDots.forEach((d, i) => {
      d.classList.toggle("beat-on", i === dotIndex);
      d.classList.toggle("beat-accent", i === dotIndex && dotIndex === 0);
    });
  }, delay);
}

/* ══════════════════
   TIPS TOGGLE
   ══════════════════ */
document.getElementById("tipsToggle").addEventListener("click", () => {
  const body = document.getElementById("tipsBody");
  const arrow = document.getElementById("tipsArrow");
  const open = !body.classList.contains("hidden");
  body.classList.toggle("hidden", open);
  arrow.textContent = open ? "▼" : "▲";
});

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
  stopMetronome();
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

/* ══════════════════
   KEY SELECTOR
   ══════════════════ */
document.querySelectorAll(".key-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".key-btn").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    state.key = btn.dataset.key;
  });
});

document.querySelectorAll(".key-quality-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".key-quality-btn").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    state.keyQuality = btn.dataset.quality;
  });
});

function checkReady() { generateBtn.disabled = !(recordedBlob || uploadedFile); }

/* ══════════════════════════════════════
   RESET + REGENERATE ALL
   ══════════════════════════════════════ */
document.getElementById("resetBtn").addEventListener("click", () => {
  // Stop any playback / sync
  stopSync();
  stopMetronome();
  if (!audioPreview.paused) audioPreview.pause();

  // Clear audio state
  recordedBlob = null;
  uploadedFile = null;
  audioChunks  = [];
  currentPhraseMap = [];
  currentRoughText = "";
  currentFlowData  = {};
  lockedLines      = {};
  _lastDebugPhrases = [];

  // Reset UI controls
  audioPreview.src = "";
  audioPreview.classList.add("hidden");
  recordBtn.classList.remove("has-audio", "recording");
  recordLabel.textContent = "Start Recording";
  recordingStatus.classList.add("hidden");
  fileInput.value = "";
  uploadText.textContent = "Upload Audio File";
  uploadLabel.classList.remove("has-file");

  // Hide results
  resultsSection.classList.add("hidden");
  versionsContainer.innerHTML = "";

  // Scroll to top and focus
  window.scrollTo({ top: 0, behavior: "smooth" });
  generateBtn.disabled = true;
});

document.getElementById("regenerateAllBtn").addEventListener("click", () => {
  if (!(recordedBlob || uploadedFile)) return;
  generateBtn.click();
});

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
  formData.append("gen_mode", state.gen_mode);
  const keyValue = state.key !== "auto" ? `${state.key} ${state.keyQuality}` : "auto";
  formData.append("key", keyValue);

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
  lockedLines = {};

  currentPhraseMap = data.phrase_map || [];
  audioDuration = data.flow?.duration || 0;
  currentRoughText = data.rough_text || "";
  currentFlowData = {
    phrase_map:   currentPhraseMap,
    melody_mode:  data.melody_mode,
    vowel_family: data.vowel_family,
    detected_key: data.detected_key,
    is_repetitive: data.is_repetitive,
    tempo_bpm:    data.flow?.tempo_bpm,
    flow_style:   data.flow?.flow_style,
  };

  // Organized transcript
  renderOrganizedTranscript(currentRoughText, currentPhraseMap, data.melody_mode);

  // Badges
  melodyBadge.classList.toggle("hidden", !data.melody_mode);
  repetitiveBadge.classList.toggle("hidden", !data.is_repetitive);

  // Detected key badge on metronome
  if (data.detected_key && state.key === "auto") {
    detectedKeyBadge.textContent = `Detected: ${data.detected_key}`;
    detectedKeyBadge.classList.remove("hidden");
  } else {
    detectedKeyBadge.classList.add("hidden");
  }

  // Analysis chips
  const chips = [];
  if (data.detected_key) chips.push(`<span class="a-chip key-chip">Key: <strong>${data.detected_key}</strong></span>`);
  if (data.vowel_family)  chips.push(`<span class="a-chip">Vowel: <strong>${data.vowel_family}</strong></span>`);
  analysisChipsEl.innerHTML = chips.join("");

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

  // Debug / Audit panel
  renderDebugPanel(data.debug_phrases || []);

  setTimeout(() => resultsSection.scrollIntoView({ behavior: "smooth", block: "start" }), 100);
}

/* ══════════════════════════════════════
   ORGANIZED TRANSCRIPT
   Shows syllable placeholders per bar —
   "da da da da" — not the guessed words.
   Whisper's text shown small underneath
   as a reference only.
   ══════════════════════════════════════ */
function makeSyllablePlaceholder(count) {
  // Alternate da/dah for a more musical feel
  const syllables = ["da", "dah", "da", "da", "dah", "da", "dah", "da", "da", "dah", "da", "dah"];
  return Array.from({ length: count }, (_, i) => syllables[i % syllables.length]).join(" ");
}

function renderOrganizedTranscript(roughText, phraseMap, melodyMode) {
  if (!phraseMap.length) {
    roughTextEl.innerHTML = `<span class="rough-plain">${escapeHtml(roughText || "(no audio detected)")}</span>`;
    return;
  }

  const bars = phraseMap.map((phrase, i) => {
    const syls = phrase.syllables || 0;
    const placeholder = makeSyllablePlaceholder(syls);
    const whisperText = phrase.text || "";

    return `<div class="transcript-bar" data-index="${i}">
      <span class="bar-num">${i + 1}</span>
      <div class="bar-content">
        <span class="bar-placeholder">${escapeHtml(placeholder)}</span>
        ${whisperText ? `<span class="bar-whisper">${escapeHtml(whisperText)}</span>` : ""}
      </div>
      <span class="bar-syls">${syls} syl</span>
    </div>`;
  }).join("");

  roughTextEl.innerHTML = bars;

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
   VERSION CARDS
   ══════════════════════════════════════ */
function createVersionCard(version, isBest, phraseMap) {
  const card = document.createElement("div");
  card.className = "version-card";
  card.dataset.versionName = version.name;

  const scorePercent = Math.round((version.score || 0) * 100);
  const lines = (version.lyrics || "").split("\n").filter((l) => l.trim());

  const lyricsHtml = lines.map((line, lineIdx) => {
    const syls = phraseMap[lineIdx]?.syllables || "?";
    return `<div class="lyric-line" data-line="${lineIdx}">
      <span class="lyric-text">${escapeHtml(line)}</span>
      <span class="line-controls">
        <span class="line-syls">${syls}</span>
        <button class="line-lock-btn" data-line="${lineIdx}" title="Lock this line">🔓</button>
        <button class="line-regen-btn" data-line="${lineIdx}" title="Regenerate this line">↻</button>
      </span>
    </div>`;
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
      ${version.score_breakdown ? '<button class="score-breakdown-btn">Why this won ▾</button>' : ""}
    </div>
    ${version.score_breakdown ? buildScoreBreakdownPanel(version.score_breakdown) : ""}
  `;

  // Copy
  card.querySelector(".copy-btn").addEventListener("click", function () {
    navigator.clipboard.writeText(decodeURIComponent(this.dataset.lyrics)).then(() => {
      this.textContent = "✓ Copied!";
      this.classList.add("copied");
      setTimeout(() => { this.textContent = "Copy"; this.classList.remove("copied"); }, 2000);
    });
  });

  // Sync
  const syncBtn = card.querySelector(".sync-btn");
  if (syncBtn) {
    syncBtn.addEventListener("click", () => {
      activeSyncCard === card ? stopSync() : startSync(card, syncBtn, phraseMap);
    });
  }

  // Lock buttons
  card.querySelectorAll(".line-lock-btn").forEach((btn) => {
    btn.addEventListener("click", () => toggleLockLine(card, version.name, parseInt(btn.dataset.line)));
  });

  // Regenerate single line buttons
  card.querySelectorAll(".line-regen-btn").forEach((btn) => {
    btn.addEventListener("click", () => regenerateSingleLine(card, version.name, parseInt(btn.dataset.line)));
  });

  return card;
}

/* ══════════════════════════════════════
   LOCK LINE
   ══════════════════════════════════════ */
function toggleLockLine(card, versionName, lineIdx) {
  if (!lockedLines[versionName]) lockedLines[versionName] = {};

  const lineEl = card.querySelector(`.lyric-line[data-line="${lineIdx}"]`);
  const btn = card.querySelector(`.line-lock-btn[data-line="${lineIdx}"]`);
  const textEl = lineEl.querySelector(".lyric-text");

  if (lockedLines[versionName][lineIdx] !== undefined) {
    // Unlock
    delete lockedLines[versionName][lineIdx];
    lineEl.classList.remove("line-locked");
    btn.textContent = "🔓";
  } else {
    // Lock
    lockedLines[versionName][lineIdx] = textEl.textContent;
    lineEl.classList.add("line-locked");
    btn.textContent = "🔒";
  }
}

/* ══════════════════════════════════════
   REGENERATE SINGLE LINE
   ══════════════════════════════════════ */
async function regenerateSingleLine(card, versionName, lineIdx) {
  const regenBtn = card.querySelector(`.line-regen-btn[data-line="${lineIdx}"]`);
  const lineEl   = card.querySelector(`.lyric-line[data-line="${lineIdx}"]`);
  const textEl   = lineEl.querySelector(".lyric-text");

  if (lineEl.classList.contains("line-locked")) return; // don't regen locked lines

  regenBtn.textContent = "⏳";
  regenBtn.disabled = true;
  lineEl.classList.add("line-regenerating");

  // Collect all current lines from this card
  const allLineEls = card.querySelectorAll(".lyric-line");
  const contextLines = Array.from(allLineEls).map((el) =>
    el.querySelector(".lyric-text").textContent
  );

  const syllableCount = currentPhraseMap[lineIdx]?.syllables || 8;
  const versionLocked = lockedLines[versionName] || {};

  try {
    const res = await fetch("/regenerate-line", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        bar_index: lineIdx,
        syllable_count: syllableCount,
        context_lines: contextLines,
        locked_lines: versionLocked,
        rough_text: currentRoughText,
        flow_data: currentFlowData,
        tone: state.tone,
        mode: state.mode,
        vibe: state.vibe,
        gen_mode: state.gen_mode,
        key: state.key !== "auto" ? `${state.key} ${state.keyQuality}` : "auto",
      }),
    });
    const data = await res.json();
    if (data.success && data.line) {
      textEl.textContent = data.line;
      lineEl.classList.add("line-refreshed");
      setTimeout(() => lineEl.classList.remove("line-refreshed"), 1200);
    }
  } catch (err) {
    console.error("Regenerate line failed:", err);
  } finally {
    regenBtn.textContent = "↻";
    regenBtn.disabled = false;
    lineEl.classList.remove("line-regenerating");
  }
}

/* ══════════════════════════════════════
   LINE-LEVEL KARAOKE SYNC
   Uses real Whisper phrase timestamps —
   highlights the whole line when that
   phrase is playing. Accurate.
   ══════════════════════════════════════ */
function startSync(card, btn, phraseMap) {
  stopSync();
  activeSyncCard = card;
  btn.textContent = "■ Stop Sync";
  btn.classList.add("syncing");

  audioPreview.currentTime = 0;
  audioPreview.play();

  syncInterval = setInterval(() => {
    const t = audioPreview.currentTime;
    const activeIdx = getCurrentPhraseIndex(t);

    card.querySelectorAll(".lyric-line").forEach((lineEl) => {
      const li = parseInt(lineEl.dataset.line);
      lineEl.classList.toggle("current-line", li === activeIdx);
      lineEl.classList.toggle("past-line", li < activeIdx);
    });

    const activeLine = card.querySelector(".lyric-line.current-line");
    if (activeLine) activeLine.scrollIntoView({ block: "nearest", behavior: "smooth" });

    if (audioPreview.ended || audioPreview.paused) stopSync();
  }, 50);

  audioPreview.addEventListener("ended", stopSync, { once: true });
}

function stopSync() {
  clearInterval(syncInterval);
  syncInterval = null;

  if (activeSyncCard) {
    activeSyncCard.querySelectorAll(".lyric-line").forEach((el) => {
      el.classList.remove("current-line", "past-line");
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

/* ══════════════════════════════════════
   SCORE BREAKDOWN ("Why this won")
   Expandable panel per version card.
   Shows per-factor score contributions
   and overflow penalty if present.
   ══════════════════════════════════════ */
const SCORE_LABELS = {
  syllable_fit:   "Syllable fit",
  word_count_fit: "Word-count fit",
  stress_fit:     "Stress fit",
  rhyme:          "Rhyme density",
  vowel_affinity: "Vowel affinity",
  singability:    "Singability",
  density_fit:    "Density fit",
};
const SCORE_WEIGHTS_UI = {
  syllable_fit: 0.35, word_count_fit: 0.15, stress_fit: 0.15,
  rhyme: 0.15, vowel_affinity: 0.05, singability: 0.10, density_fit: 0.05,
};

function buildScoreBreakdownPanel(bd) {
  if (!bd) return "";

  const rows = Object.entries(SCORE_LABELS).map(([key, label]) => {
    const raw   = bd[key] ?? 0;
    const wt    = SCORE_WEIGHTS_UI[key] ?? 0;
    const contribution = raw * wt;
    const barW  = Math.round(raw * 100);
    return `<div class="sbd-row">
      <span class="sbd-label">${label}</span>
      <div class="sbd-bar-wrap"><div class="sbd-bar" style="width:${barW}%"></div></div>
      <span class="sbd-raw">${Math.round(raw * 100)}%</span>
      <span class="sbd-contrib">+${contribution.toFixed(3)}</span>
    </div>`;
  }).join("");

  const overflowWarn = (bd.overflow_bars || []).length
    ? `<div class="sbd-overflow-warn">⚠ Overflow penalty (×${0.45}) — bars: ${bd.overflow_bars.join(", ")}</div>`
    : "";

  return `<div class="score-breakdown-panel hidden">
    ${rows}
    <div class="sbd-total">
      <span>TOTAL</span>
      <span class="sbd-total-val">${Math.round((bd.total || 0) * 100)}%</span>
    </div>
    ${overflowWarn}
  </div>`;
}

// Wire "Why this won" toggle (uses event delegation on versionsContainer)
versionsContainer.addEventListener("click", (e) => {
  const btn = e.target.closest(".score-breakdown-btn");
  if (!btn) return;
  const panel = btn.closest(".version-card").querySelector(".score-breakdown-panel");
  if (!panel) return;
  const open = !panel.classList.contains("hidden");
  panel.classList.toggle("hidden", open);
  btn.textContent = open ? "Why this won ▾" : "Why this won ▴";
});

/* ══════════════════════════════════════
   DEBUG / AUDIT PANEL
   Per-phrase audio analysis table.
   Shows pitch, density, energy, confidence
   so the user can see exactly how the
   audio was interpreted before generation.
   ══════════════════════════════════════ */
const debugSection = document.getElementById("debugSection");
const debugToggle  = document.getElementById("debugToggle");
const debugArrow   = document.getElementById("debugArrow");
const debugBody    = document.getElementById("debugBody");
const debugTableBody = document.getElementById("debugTableBody");

debugToggle.addEventListener("click", () => {
  const open = !debugBody.classList.contains("hidden");
  debugBody.classList.toggle("hidden", open);
  debugArrow.textContent = open ? "▼" : "▲";
});

const CONF_CLASS = { high: "conf-high", med: "conf-med", low: "conf-low" };
const PITCH_ICON = { "↗ rising": "↗", "↘ falling": "↘", "~ held": "~", "→ flat": "→", "↕ staccato": "↕" };

// Keep a reference so the CSV export can read it
let _lastDebugPhrases = [];

function renderDebugPanel(debugPhrases) {
  if (!debugPhrases || !debugPhrases.length) {
    debugSection.classList.add("hidden");
    return;
  }

  _lastDebugPhrases = debugPhrases;
  debugSection.classList.remove("hidden");

  debugTableBody.innerHTML = debugPhrases.map((row) => {
    const confCls   = CONF_CLASS[row.confidence] || "";
    const pitchIcon = PITCH_ICON[row.pitch] || row.pitch || "—";
    const transcript = row.text
      ? escapeHtml(row.text.slice(0, 24)) + (row.text.length > 24 ? "…" : "")
      : '<em style="opacity:.4">melody</em>';
    const pauseStr  = row.pause_after > 0.15 ? `${row.pause_after.toFixed(1)}s` : "—";
    const lwPct     = typeof row.literal_weight === "number"
      ? `${Math.round(row.literal_weight * 100)}%` : "—";
    const svrPct    = typeof row.sustained_vowel_ratio === "number"
      ? `${Math.round(row.sustained_vowel_ratio * 100)}%` : "—";
    const flagBadges = (row.flags || []).map(f =>
      `<span class="debug-flag debug-flag--${f.replace(/[^a-z]/g, "-")}">${f}</span>`
    ).join("");

    // Row-level warning class
    const rowCls = (row.flags || []).length ? "debug-row-warn" : "";

    return `<tr class="${rowCls}">
      <td class="debug-bar">${row.bar}</td>
      <td class="debug-text">${transcript}</td>
      <td>${row.syllables}</td>
      <td>${row.duration}s</td>
      <td class="debug-pitch">${pitchIcon}</td>
      <td>${row.density || "—"}</td>
      <td>${row.energy || "—"}</td>
      <td class="debug-conf ${confCls}" title="score: ${row.confidence_score || ""}">${row.confidence || "—"}</td>
      <td class="debug-lw">${lwPct}</td>
      <td>${row.max_words ?? "—"}</td>
      <td>${row.max_syllables ?? "—"}</td>
      <td>${svrPct}</td>
      <td>${row.vowel_family_hint || "—"}</td>
      <td>${row.word_length_profile || "—"}</td>
      <td>${pauseStr}</td>
      <td>${flagBadges}</td>
    </tr>`;
  }).join("");
}

/* ══════════════════════════════════════
   BENCHMARK CSV EXPORT
   Downloads a pre-filled CSV template
   with audio analysis data + empty columns
   for the user to fill in manually.
   Columns: clip, bar, transcript, syl, dur,
            pitch, density, energy, confidence,
            expected_feel, feature_misread, fix_applied
   ══════════════════════════════════════ */
document.getElementById("benchmarkExportBtn").addEventListener("click", () => {
  if (!_lastDebugPhrases.length) return;

  const headers = [
    "clip", "bar", "transcript", "syllables", "duration_s",
    "pitch", "density", "energy", "confidence", "flags",
    "expected_feel", "feature_misread", "fix_applied", "notes",
  ];

  const clipName = "clip_" + new Date().toISOString().slice(0, 10);

  const rows = _lastDebugPhrases.map((row) => [
    clipName,
    row.bar,
    `"${(row.text || "").replace(/"/g, '""')}"`,
    row.syllables,
    row.duration,
    row.pitch || "—",
    row.density || "—",
    row.energy || "—",
    row.confidence || "—",
    (row.flags || []).join("+") || "—",
    "", // expected_feel — fill in manually
    "", // feature_misread — fill in manually
    "", // fix_applied — fill in manually
    "", // notes
  ]);

  const csv = [headers.join(","), ...rows.map(r => r.join(","))].join("\n");
  const blob = new Blob([csv], { type: "text/csv" });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement("a");
  a.href     = url;
  a.download = `lyric-goat-benchmark-${clipName}.csv`;
  a.click();
  URL.revokeObjectURL(url);
});
