const API_BASE = window.location.origin;
const TRANSCRIBE_URL = `${API_BASE}/transcribe`;
const HEALTH_URL = `${API_BASE}/health`;

const fileInput = document.getElementById("fileInput");
const dropZone = document.getElementById("dropZone");
const chooseBtn = document.getElementById("chooseBtn");
const transcribeBtn = document.getElementById("transcribeBtn");
const timestampsToggle = document.getElementById("timestampsToggle");
const fileName = document.getElementById("fileName");
const transcriptEl = document.getElementById("transcript");
const segmentsEl = document.getElementById("segments");
const segmentCount = document.getElementById("segmentCount");
const durationChip = document.getElementById("durationChip");
const copyBtn = document.getElementById("copyBtn");
const toastEl = document.getElementById("toast");
const statusDot = document.getElementById("statusDot");
const statusText = document.getElementById("statusText");
const modelName = document.getElementById("modelName");
const deviceName = document.getElementById("deviceName");
const langName = document.getElementById("langName");
const refreshHealth = document.getElementById("refreshHealth");

let currentFile = null;

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
    2
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
  const form = new FormData();
  form.append("file", currentFile);
  form.append("include_timestamps", includeTimestamps ? "true" : "false");

  setLoading(true);
  durationChip.textContent = "Processing…";
  transcriptEl.textContent = "Transcribing…";
  renderSegments([]);

  const started = performance.now();

  try {
    const res = await fetch(TRANSCRIBE_URL, { method: "POST", body: form });
    const elapsed = (performance.now() - started) / 1000;

    if (!res.ok) {
      const errText = await res.text();
      throw new Error(errText || "Transcription failed");
    }

    const data = await res.json();
    transcriptEl.textContent = data.text || "(No text returned)";
    durationChip.textContent = `Done in ${elapsed.toFixed(1)}s`;
    renderSegments(data.segments || []);
    setToast("Transcription complete");
  } catch (err) {
    durationChip.textContent = "Error";
    transcriptEl.textContent = "Transcription failed. See toast for details.";
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

chooseBtn.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", (e) => handleFileInput(e.target.files));

transcribeBtn.addEventListener("click", transcribe);
copyBtn.addEventListener("click", copyTranscript);
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
