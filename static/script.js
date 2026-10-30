const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("fileInput");
const uploadStage = document.getElementById("uploadStage");
const previewStage = document.getElementById("previewStage");
const previewImg = document.getElementById("previewImg");
const errorMsg = document.getElementById("errorMsg");
const loadingRow = document.getElementById("loadingRow");
const resultRow = document.getElementById("resultRow");
const resultLabel = document.getElementById("resultLabel");
const resultHindi = document.getElementById("resultHindi");
const dietDot = document.getElementById("dietDot");
const confidenceFill = document.getElementById("confidenceFill");
const confidenceValue = document.getElementById("confidenceValue");
const altRow = document.getElementById("altRow");
const demoNotice = document.getElementById("demoNotice");
const resetBtn = document.getElementById("resetBtn");
const nutCalories = document.getElementById("nutCalories");
const nutProtein = document.getElementById("nutProtein");
const nutCarbs = document.getElementById("nutCarbs");
const nutFat = document.getElementById("nutFat");
const nutCalcium = document.getElementById("nutCalcium");
const walkNote = document.getElementById("walkNote");
const walkMinutes = document.getElementById("walkMinutes");
const listenBtn = document.getElementById("listenBtn");
const logBtn = document.getElementById("logBtn");
const printBtn = document.getElementById("printBtn");
const todayTotal = document.getElementById("todayTotal");
const todayCount = document.getElementById("todayCount");
const clearLogBtn = document.getElementById("clearLogBtn");

let lastResult = null; // holds the most recent prediction for Listen/Log/Print

// ---------- Upload flow ----------

function showError(message) {
  errorMsg.textContent = message;
  errorMsg.hidden = false;
}

function clearError() {
  errorMsg.hidden = true;
  errorMsg.textContent = "";
}

function openFileDialog() {
  fileInput.click();
}

dropzone.addEventListener("click", openFileDialog);
dropzone.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") {
    e.preventDefault();
    openFileDialog();
  }
});

["dragenter", "dragover"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  })
);

["dragleave", "drop"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
  })
);

dropzone.addEventListener("drop", (e) => {
  const file = e.dataTransfer.files[0];
  if (file) handleFile(file);
});

fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) handleFile(fileInput.files[0]);
});

function handleFile(file) {
  clearError();

  const allowedTypes = ["image/jpeg", "image/png", "image/jpg"];
  if (!allowedTypes.includes(file.type)) {
    showError("Please upload a JPG or PNG image.");
    return;
  }
  if (file.size > 8 * 1024 * 1024) {
    showError("Image is too large — please use a file under 8MB.");
    return;
  }

  const reader = new FileReader();
  reader.onload = (e) => {
    previewImg.src = e.target.result;
    goToPreviewStage();
    submitForPrediction(file);
  };
  reader.readAsDataURL(file);
}

function goToPreviewStage() {
  uploadStage.hidden = true;
  previewStage.hidden = false;
  loadingRow.hidden = false;
  resultRow.hidden = true;
  demoNotice.hidden = true;
  confidenceFill.style.width = "0%";
  logBtn.classList.remove("active");
  logBtn.textContent = "+ Add to today's log";
}

async function submitForPrediction(file) {
  const formData = new FormData();
  formData.append("image", file);

  try {
    const response = await fetch("/predict", { method: "POST", body: formData });
    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || "Something went wrong.");
    }

    renderResult(data);
  } catch (err) {
    loadingRow.hidden = true;
    showError(err.message || "Could not reach the server.");
    resetToUpload();
  }
}

function renderResult(data) {
  loadingRow.hidden = true;
  resultRow.hidden = false;
  lastResult = data;

  resultLabel.textContent = data.label;
  resultHindi.textContent = data.label_hindi || "";
  confidenceValue.textContent = `${data.confidence}%`;

  requestAnimationFrame(() => {
    confidenceFill.style.width = `${data.confidence}%`;
  });

  altRow.innerHTML = "";
  data.top3.slice(1).forEach((item) => {
    const span = document.createElement("span");
    span.textContent = `${item.label} ${item.confidence}%`;
    altRow.appendChild(span);
  });

  demoNotice.hidden = !data.demo_mode;

  const n = data.nutrition || {};
  const fmt = (val, unit) => (val === null || val === undefined) ? "—" : `${val}${unit}`;
  nutCalories.textContent = fmt(n.calories, "");
  nutProtein.textContent = fmt(n.protein_g, "g");
  nutCarbs.textContent = fmt(n.carbs_g, "g");
  nutFat.textContent = fmt(n.fat_g, "g");
  nutCalcium.textContent = fmt(n.calcium_mg, "mg");

  // Veg / Non-Veg badge
  dietDot.className = "dietDot";
  if (data.diet === "Vegetarian") {
    dietDot.classList.add("veg");
    dietDot.title = "Vegetarian";
  } else if (data.diet === "Non-Vegetarian") {
    dietDot.classList.add("nonveg");
    dietDot.title = "Non-Vegetarian";
  } else {
    dietDot.title = "";
  }

  // Calorie burn estimate
  if (data.walk_minutes !== null && data.walk_minutes !== undefined) {
    walkMinutes.textContent = `${data.walk_minutes} min`;
    walkNote.hidden = false;
  } else {
    walkNote.hidden = true;
  }
}

function resetToUpload() {
  uploadStage.hidden = false;
  previewStage.hidden = true;
  fileInput.value = "";
  lastResult = null;
}

resetBtn.addEventListener("click", resetToUpload);

// ---------- Listen (Web Speech API, client-side only) ----------

listenBtn.addEventListener("click", () => {
  if (!lastResult || !("speechSynthesis" in window)) return;
  const n = lastResult.nutrition || {};
  const parts = [
    `This looks like ${lastResult.label}.`,
    `Confidence ${lastResult.confidence} percent.`,
  ];
  if (n.calories) parts.push(`Around ${n.calories} calories per 100 grams.`);
  if (lastResult.diet) parts.push(lastResult.diet + ".");
  const utterance = new SpeechSynthesisUtterance(parts.join(" "));
  window.speechSynthesis.cancel();
  window.speechSynthesis.speak(utterance);
});

// ---------- Today's log (localStorage, per-browser) ----------

const LOG_KEY = "foodlens_daily_log";

function todayKey() {
  return new Date().toISOString().slice(0, 10); // YYYY-MM-DD
}

function loadTodayLog() {
  try {
    const raw = localStorage.getItem(LOG_KEY);
    const all = raw ? JSON.parse(raw) : {};
    return all[todayKey()] || [];
  } catch (e) {
    return [];
  }
}

function saveTodayLog(entries) {
  try {
    const raw = localStorage.getItem(LOG_KEY);
    const all = raw ? JSON.parse(raw) : {};
    all[todayKey()] = entries;
    localStorage.setItem(LOG_KEY, JSON.stringify(all));
  } catch (e) {
    // localStorage unavailable — log just won't persist, app still works
  }
}

function renderTodaySummary() {
  const entries = loadTodayLog();
  const total = entries.reduce((sum, e) => sum + (e.calories || 0), 0);
  todayTotal.textContent = total;
  todayCount.textContent = entries.length;
}

logBtn.addEventListener("click", () => {
  if (!lastResult) return;
  const n = lastResult.nutrition || {};
  const entries = loadTodayLog();
  entries.push({
    label: lastResult.label,
    calories: n.calories || 0,
    time: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
  });
  saveTodayLog(entries);
  renderTodaySummary();
  logBtn.classList.add("active");
  logBtn.textContent = "✓ Added to log";
});

clearLogBtn.addEventListener("click", () => {
  saveTodayLog([]);
  renderTodaySummary();
});

// ---------- Save as PDF (browser print dialog, "Save as PDF" destination) ----------

printBtn.addEventListener("click", () => {
  window.print();
});

// Initial paint of today's log on page load
renderTodaySummary();
