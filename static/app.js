const API_BASE = window.location.origin;
const TRANSCRIBE_URL = `${API_BASE}/transcribe`;
const HEALTH_URL = `${API_BASE}/health`;
const WS_SCHEME = window.location.protocol === "https:" ? "wss" : "ws";
const WS_URL = `${WS_SCHEME}://${window.location.host}/ws/stream`;
const TARGET_SAMPLE_RATE = 16000;
const STREAM_CHUNK_SECONDS = 1.25;
const STREAM_CHUNK_SAMPLES = Math.floor(
  TARGET_SAMPLE_RATE * STREAM_CHUNK_SECONDS,
);
const LIVE_NOISE_GATE = 0.012;
const LIVE_TARGET_PEAK = 0.9;

const fileInput = document.getElementById("fileInput");
const dropZone = document.getElementById("dropZone");
const chooseBtn = document.getElementById("chooseBtn");
const transcribeBtn = document.getElementById("transcribeBtn");
const recordBtn = document.getElementById("recordBtn");
const timestampsToggle = document.getElementById("timestampsToggle");
const fileName = document.getElementById("fileName");
const transcriptEl = document.getElementById("transcript");
const segmentsEl = document.getElementById("segments");
const segmentCount = document.getElementById("segmentCount");
const durationChip = document.getElementById("durationChip");
const copyBtn = document.getElementById("copyBtn");
const downloadBtn = document.getElementById("downloadBtn");
const keywordsInput = document.getElementById("keywordsInput");
const keywordHitsEl = document.getElementById("keywordHits");
const keywordHitCount = document.getElementById("keywordHitCount");
const liveTranscriptEl = document.getElementById("liveTranscript");
const liveStatus = document.getElementById("liveStatus");
const toastEl = document.getElementById("toast");
const statusDot = document.getElementById("statusDot");
const statusText = document.getElementById("statusText");
const modelName = document.getElementById("modelName");
const deviceName = document.getElementById("deviceName");
const langName = document.getElementById("langName");
const refreshHealth = document.getElementById("refreshHealth");

let currentFile = null;
let websocket = null;
let isRecording = false;
let audioContext = null;
let mediaStream = null;
let sourceNode = null;
let processorNode = null;
let pendingSamples = [];
let lastTranscriptText = "";
let lastSegments = [];
let lastKeywordHits = [];

const setDownloadEnabled = (enabled) => {
  downloadBtn.disabled = !enabled;
  downloadBtn.style.opacity = enabled ? 1 : 0.55;
};
setDownloadEnabled(false);

const setToast = (msg, tone = "neutral") => {
  toastEl.textContent = msg;
  toastEl.classList.remove("hidden");
  toastEl.style.borderColor =
    tone === "error" ? "rgba(255,95,109,0.6)" : "var(--panel-border)";
  toastEl.style.color = tone === "error" ? "#ffd7dc" : "var(--text)";
  clearTimeout(setToast._timer);
  setToast._timer = setTimeout(() => toastEl.classList.add("hidden"), 3200);
};

const setLoading = (isLoading) => {
  transcribeBtn.disabled = isLoading;
  transcribeBtn.textContent = isLoading ? "Transcribing…" : "Transcribe";
  transcribeBtn.style.opacity = isLoading ? 0.7 : 1;
};

const setHealth = (status, info) => {
  const ok = status === "healthy";
  statusDot.style.background = ok ? "var(--success)" : "var(--danger)";
  statusText.textContent = ok ? "Model ready" : "Model not loaded";
  modelName.textContent = info?.model_name || "—";
  deviceName.textContent = info?.device || "—";
  langName.textContent = info?.language || "—";
};

const formatTime = (seconds) => {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60)
    .toString()
    .padStart(2, "0");
  return `${m}:${s}`;
};

const escapeHtml = (text) =>
  (text || "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");

const normalizeKeywords = (rawKeywords) => {
  if (!rawKeywords) return [];
  return rawKeywords
    .split(",")
    .map((item) => item.trim().replace(/\s+/g, " "))
    .filter(Boolean);
};

const highlightTranscript = (text, keywords) => {
  if (!text) return "";
  if (!keywords.length) return escapeHtml(text);

  const escapedKeywords = keywords
    .map((keyword) => keyword.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"))
    .sort((a, b) => b.length - a.length);

  const regex = new RegExp(`\\b(${escapedKeywords.join("|")})\\b`, "gi");
  return escapeHtml(text).replace(regex, '<mark class="kw-hit">$1</mark>');
};

const renderKeywordHits = (hits) => {
  if (!hits || !hits.length) {
    keywordHitsEl.classList.add("empty");
    keywordHitCount.textContent = "0 hits";
    keywordHitsEl.innerHTML =
      '<p class="placeholder">No keyword hits detected in this transcript.</p>';
    return;
  }

  keywordHitsEl.classList.remove("empty");
  keywordHitCount.textContent = `${hits.length} hit${hits.length === 1 ? "" : "s"}`;
  keywordHitsEl.innerHTML = hits
    .map((hit, idx) => {
      const start = formatTime(hit.start || 0);
      const end = formatTime(hit.end || 0);
      const match = escapeHtml((hit.match || "").trim());
      const keyword = escapeHtml((hit.keyword || "").trim());
      return `
        <div class="segment">
          <div class="segment-meta">
            <span class="badge-idx">#${idx + 1}</span>
            <span class="badge-time">${start} – ${end}</span>
            <span class="badge-keyword">${keyword}</span>
          </div>
          <div>${match}</div>
        </div>
      `;
    })
    .join("");
};

const renderSegments = (segments) => {
  if (!segments || !segments.length) {
    segmentsEl.innerHTML =
      '<p class="placeholder">Enable timestamps to see segment-level timing.</p>';
    segmentsEl.classList.add("empty");
    segmentCount.textContent = "0 segments";
    return;
  }

  segmentsEl.classList.remove("empty");
  segmentCount.textContent = `${segments.length} segment${
    segments.length === 1 ? "" : "s"
  }`;
  segmentsEl.innerHTML = segments
    .map((seg, idx) => {
      const start = formatTime(seg.start || 0);
      const end = formatTime(seg.end || 0);
      const safeText = (seg.text || "").trim();
      return `
        <div class="segment">
          <div class="segment-meta">
            <span class="badge-idx">#${idx + 1}</span>
            <span class="badge-time">${start} – ${end}</span>
          </div>
          <div>${safeText}</div>
        </div>
      `;
    })
    .join("");
};

const updateFile = (file) => {
  if (!file) return;
  currentFile = file;
  fileName.textContent = `${file.name} • ${(file.size / (1024 * 1024)).toFixed(
    2,
  )} MB`;
  setToast("File ready to transcribe");
};

const handleFileInput = (files) => {
  if (!files || !files.length) return;
  updateFile(files[0]);
};

const fetchHealth = async () => {
  try {
    statusText.textContent = "Checking health…";
    const res = await fetch(HEALTH_URL);
    if (!res.ok) throw new Error("Health check failed");
    const data = await res.json();
    setHealth(data.status, data.model_info);
  } catch (err) {
    setHealth("not_ready", {});
    setToast("Health check failed. Is the server running?", "error");
  }
};

const transcribe = async () => {
  if (!currentFile) {
    setToast("Select an audio file first", "error");
    return;
  }

  const includeTimestamps = timestampsToggle.checked;
  const rawKeywords = keywordsInput.value || "";
  const normalizedKeywords = normalizeKeywords(rawKeywords);
  const form = new FormData();
  form.append("file", currentFile);
  form.append("include_timestamps", includeTimestamps ? "true" : "false");
  if (normalizedKeywords.length) {
    form.append("keywords", normalizedKeywords.join(", "));
  }

  setLoading(true);
  durationChip.textContent = "Processing…";
  transcriptEl.textContent = "Transcribing…";
  renderSegments([]);
  renderKeywordHits([]);
  lastTranscriptText = "";
  lastSegments = [];
  lastKeywordHits = [];
  setDownloadEnabled(false);

  const started = performance.now();

  try {
    const res = await fetch(TRANSCRIBE_URL, { method: "POST", body: form });
    const elapsed = (performance.now() - started) / 1000;

    if (!res.ok) {
      const errText = await res.text();
      throw new Error(errText || "Transcription failed");
    }

    const data = await res.json();
    transcriptEl.innerHTML = highlightTranscript(
      data.text || "",
      normalizedKeywords,
    );
    lastTranscriptText = data.text || "";
    lastSegments = data.segments || [];
    lastKeywordHits = data.keyword_hits || [];
    durationChip.textContent = `Done in ${elapsed.toFixed(1)}s`;
    renderSegments(data.segments || []);
    renderKeywordHits(data.keyword_hits || []);
    setDownloadEnabled(Boolean(lastTranscriptText.trim()));
    setToast("Transcription complete");
  } catch (err) {
    durationChip.textContent = "Error";
    transcriptEl.textContent = "Transcription failed. See toast for details.";
    renderKeywordHits([]);
    setDownloadEnabled(false);
    setToast(err.message, "error");
  } finally {
    setLoading(false);
  }
};

const copyTranscript = async () => {
  const text = transcriptEl.textContent || "";
  if (!text.trim()) {
    setToast("Nothing to copy", "error");
    return;
  }
  try {
    await navigator.clipboard.writeText(text);
    setToast("Copied to clipboard");
  } catch (err) {
    setToast("Clipboard unavailable", "error");
  }
};

const buildDownloadText = () => {
  const lines = [];
  const fileLabel = currentFile ? currentFile.name : "unknown";
  lines.push(`File: ${fileLabel}`);
  lines.push(`Generated: ${new Date().toISOString()}`);
  lines.push("");
  lines.push("Transcript:");
  lines.push(lastTranscriptText || transcriptEl.textContent || "");

  if (lastSegments && lastSegments.length) {
    lines.push("");
    lines.push("Segments:");
    lastSegments.forEach((seg, idx) => {
      const start = formatTime(seg.start || 0);
      const end = formatTime(seg.end || 0);
      lines.push(`#${idx + 1} [${start} - ${end}] ${seg.text || ""}`);
    });
  }

  if (lastKeywordHits && lastKeywordHits.length) {
    lines.push("");
    lines.push("Keyword hits:");
    lastKeywordHits.forEach((hit, idx) => {
      const start = formatTime(hit.start || 0);
      const end = formatTime(hit.end || 0);
      lines.push(`#${idx + 1} [${start} - ${end}] ${hit.keyword}: ${hit.match}`);
    });
  }

  return lines.join("\n");
};

const downloadTranscript = () => {
  const text = lastTranscriptText || transcriptEl.textContent || "";
  if (!text.trim()) {
    setToast("Nothing to download", "error");
    return;
  }

  const blob = new Blob([buildDownloadText()], { type: "text/plain" });
  const url = URL.createObjectURL(blob);
  const baseName = currentFile
    ? currentFile.name.replace(/\.[^.]+$/, "")
    : "transcript";
  const link = document.createElement("a");
  link.href = url;
  link.download = `${baseName || "transcript"}.txt`;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
  setToast("Download started");
};

const floatTo16BitPCM = (float32Array) => {
  const pcm = new Int16Array(float32Array.length);
  for (let i = 0; i < float32Array.length; i++) {
    const s = Math.max(-1, Math.min(1, float32Array[i]));
    pcm[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return pcm;
};

const downsampleBuffer = (channelData, sourceRate, targetRate) => {
  if (sourceRate === targetRate) {
    return channelData;
  }

  const ratio = sourceRate / targetRate;
  const newLength = Math.max(1, Math.round(channelData.length / ratio));
  const result = new Float32Array(newLength);

  let offset = 0;
  for (let i = 0; i < newLength; i++) {
    const nextOffset = Math.min(
      channelData.length,
      Math.round((i + 1) * ratio),
    );
    let sum = 0;
    let count = 0;

    for (let j = offset; j < nextOffset; j++) {
      sum += channelData[j];
      count += 1;
    }

    result[i] = count > 0 ? sum / count : 0;
    offset = nextOffset;
  }

  return result;
};

const preprocessLiveChunk = (samples) => {
  if (!samples.length) {
    return samples;
  }

  let sum = 0;
  for (let i = 0; i < samples.length; i++) {
    sum += samples[i];
  }

  const mean = sum / samples.length;
  const centered = new Float32Array(samples.length);
  let peak = 0;

  for (let i = 0; i < samples.length; i++) {
    const value = samples[i] - mean;
    centered[i] = Math.abs(value) < LIVE_NOISE_GATE ? 0 : value;
    peak = Math.max(peak, Math.abs(centered[i]));
  }

  if (peak === 0) {
    return centered;
  }

  const scale = Math.min(1, LIVE_TARGET_PEAK / peak);
  for (let i = 0; i < centered.length; i++) {
    centered[i] *= scale;
  }

  return centered;
};

const queueSamples = (samples) => {
  for (let i = 0; i < samples.length; i++) {
    pendingSamples.push(samples[i]);
  }
};

const flushPendingSamples = () => {
  if (!websocket || websocket.readyState !== WebSocket.OPEN) {
    return;
  }

  while (pendingSamples.length >= STREAM_CHUNK_SAMPLES) {
    const chunk = pendingSamples.slice(0, STREAM_CHUNK_SAMPLES);
    pendingSamples = pendingSamples.slice(STREAM_CHUNK_SAMPLES);
    const pcmData = floatTo16BitPCM(Float32Array.from(chunk));
    websocket.send(pcmData.buffer);
  }
};

const closeAudioPipeline = async () => {
  if (processorNode) {
    processorNode.disconnect();
    processorNode.onaudioprocess = null;
    processorNode = null;
  }

  if (sourceNode) {
    sourceNode.disconnect();
    sourceNode = null;
  }

  if (mediaStream) {
    mediaStream.getTracks().forEach((track) => track.stop());
    mediaStream = null;
  }

  if (audioContext) {
    await audioContext.close();
    audioContext = null;
  }

  pendingSamples = [];
};

const startRecording = async () => {
  if (isRecording) return;

  if (!navigator.mediaDevices?.getUserMedia) {
    setToast("Microphone is not available in this browser", "error");
    return;
  }

  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    });

    audioContext = new AudioContext();
    sourceNode = audioContext.createMediaStreamSource(mediaStream);
    processorNode = audioContext.createScriptProcessor(4096, 1, 1);

    processorNode.onaudioprocess = (event) => {
      if (!websocket || websocket.readyState !== WebSocket.OPEN) {
        return;
      }

      const inputData = event.inputBuffer.getChannelData(0);
      const downsampled = downsampleBuffer(
        inputData,
        audioContext.sampleRate,
        TARGET_SAMPLE_RATE,
      );

      queueSamples(preprocessLiveChunk(downsampled));
      flushPendingSamples();
    };

    sourceNode.connect(processorNode);
    processorNode.connect(audioContext.destination);

    // Connect to WebSocket
    websocket = new WebSocket(WS_URL);

    websocket.onopen = () => {
      console.log("WebSocket connected");
      liveStatus.textContent = "Recording…";
      liveStatus.style.color = "var(--accent)";
      recordBtn.textContent = "⏹️ Stop Recording";
      isRecording = true;
      setToast("Recording started");
    };

    websocket.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === "transcription") {
        liveTranscriptEl.textContent = data.accumulated_text || "Listening…";
      } else if (data.type === "error") {
        setToast(`Transcription error: ${data.message}`, "error");
      }
    };

    websocket.onclose = () => {
      console.log("WebSocket closed");
      stopRecording();
    };

    websocket.onerror = (error) => {
      console.error("WebSocket error:", error);
      setToast("Connection error", "error");
      stopRecording();
    };
  } catch (err) {
    console.error("Recording error:", err);
    setToast("Could not access microphone", "error");
    stopRecording({ announce: false });
  }
};

const stopRecording = async ({ announce = true } = {}) => {
  const activeSocket = websocket;
  websocket = null;

  await closeAudioPipeline();

  if (activeSocket && activeSocket.readyState < WebSocket.CLOSING) {
    activeSocket.onclose = null;
    activeSocket.close();
  }

  isRecording = false;
  liveStatus.textContent = "Ready";
  liveStatus.style.color = "var(--muted)";
  recordBtn.textContent = "🎤 Start Recording";

  if (announce) {
    setToast("Recording stopped");
  }
};

const toggleRecording = () => {
  if (isRecording) {
    stopRecording();
  } else {
    startRecording();
  }
};

chooseBtn.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", (e) => handleFileInput(e.target.files));

transcribeBtn.addEventListener("click", transcribe);
copyBtn.addEventListener("click", copyTranscript);
downloadBtn.addEventListener("click", downloadTranscript);
recordBtn.addEventListener("click", toggleRecording);
refreshHealth.addEventListener("click", fetchHealth);

["dragenter", "dragover"].forEach((evt) => {
  dropZone.addEventListener(evt, (e) => {
    e.preventDefault();
    e.stopPropagation();
    dropZone.classList.add("dragging");
  });
});

["dragleave", "drop"].forEach((evt) => {
  dropZone.addEventListener(evt, (e) => {
    e.preventDefault();
    e.stopPropagation();
    dropZone.classList.remove("dragging");
  });
});

dropZone.addEventListener("drop", (e) => {
  handleFileInput(e.dataTransfer.files);
});

dropZone.addEventListener("click", () => fileInput.click());

window.addEventListener("load", fetchHealth);
