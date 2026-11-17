const $ = (id) => document.getElementById(id);
const fileInput = $("file"), dz = $("dropzone");
let currentFile = null, beforeURL = null, timer = null;

const ALLOWED = ["image/jpeg", "image/png", "image/webp"];
const MAX_MB = 10;

function showError(msg) { $("error").textContent = msg; $("error").hidden = !msg; }

function setFile(f) {
  showError("");
  if (!f) return;
  if (!ALLOWED.includes(f.type)) return showError("Only JPG, JPEG, PNG and WEBP files are allowed.");
  if (f.size > MAX_MB * 1024 * 1024) return showError(`File too large. Maximum is ${MAX_MB} MB.`);
  currentFile = f;
  if (beforeURL) URL.revokeObjectURL(beforeURL);
  beforeURL = URL.createObjectURL(f);
  $("preview").src = beforeURL;
  $("preview-wrap").hidden = false;
  $("drop-text").hidden = true;
  $("generate").disabled = false;
}

function clearFile() {
  currentFile = null; fileInput.value = "";
  $("preview-wrap").hidden = true; $("drop-text").hidden = false;
  $("generate").disabled = true;
}

dz.addEventListener("click", (e) => { if (!currentFile && e.target.tagName !== "BUTTON") fileInput.click(); });
fileInput.addEventListener("change", () => setFile(fileInput.files[0]));
$("replace").addEventListener("click", (e) => { e.stopPropagation(); fileInput.click(); });
$("remove").addEventListener("click", (e) => { e.stopPropagation(); clearFile(); });
["dragover", "dragenter"].forEach(ev => dz.addEventListener(ev, e => { e.preventDefault(); dz.classList.add("over"); }));
["dragleave", "drop"].forEach(ev => dz.addEventListener(ev, e => { e.preventDefault(); dz.classList.remove("over"); }));
dz.addEventListener("drop", e => setFile(e.dataTransfer.files[0]));

const steps = ["Analyzing your photo…", "Building the historical style…", "Generating your photograph…", "Finishing touches…"];

$("generate").addEventListener("click", async () => {
  if (!currentFile) return;
  showError("");
  $("form-card").hidden = true; $("result").hidden = true; $("progress").hidden = false;
  let i = 0; $("progress-text").textContent = steps[0];
  timer = setInterval(() => { i = Math.min(i + 1, steps.length - 1); $("progress-text").textContent = steps[i]; }, 12000);

  const fd = new FormData();
  fd.append("photo", currentFile);
  fd.append("year", $("year").value);
  try {
    const res = await fetch("/generate", { method: "POST", body: fd });
    const data = await res.json().catch(() => ({ error: "Unexpected server response." }));
    if (!res.ok) throw new Error(data.error || "Generation failed.");

    $("img-before").src = beforeURL;
    $("img-after").src = data.result_url + "?t=" + Date.now();
    $("result-year").textContent = data.year; $("cap-year").textContent = data.year;
    $("download").href = data.result_url + "?download=1";
    if (data.classifier_used && data.generated_1980s_probability != null) {
      $("classifier-note").textContent = `Classifier 1980s score — original: ${(data.original_1980s_probability * 100).toFixed(0)}%, generated: ${(data.generated_1980s_probability * 100).toFixed(0)}%.`;
    } else if (data.classifier_used) {
      $("classifier-note").textContent = "Classifier guided the style strength (it only scores 1980s photos).";
    } else {
      $("classifier-note").textContent = "";
    }
    $("progress").hidden = true; $("result").hidden = false;
  } catch (err) {
    $("progress").hidden = true; $("form-card").hidden = false;
    showError(err.message);
  } finally { clearInterval(timer); }
});

$("again").addEventListener("click", () => {
  $("result").hidden = true; $("form-card").hidden = false; clearFile();
});
