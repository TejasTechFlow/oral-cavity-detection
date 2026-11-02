const input = document.getElementById("image");
const zone = document.getElementById("dropzone");
const preview = document.getElementById("preview");
const placeholder = document.getElementById("placeholder");
const btn = document.getElementById("submit-btn");
const err = document.getElementById("file-error");
const form = document.getElementById("upload-form");
const MAX = 10 * 1024 * 1024;

function showError(msg) { err.textContent = msg; err.hidden = !msg; }

function handle(file) {
  showError("");
  if (!file) return;
  if (!file.type.startsWith("image/")) { showError("Please select an image file."); btn.disabled = true; return; }
  if (file.size > MAX) { showError("File too large (max 10 MB)."); btn.disabled = true; return; }
  preview.src = URL.createObjectURL(file);
  preview.hidden = false;
  placeholder.hidden = true;
  btn.disabled = false;
}

input.addEventListener("change", () => handle(input.files[0]));
["dragenter", "dragover"].forEach(e => zone.addEventListener(e, ev => { ev.preventDefault(); zone.classList.add("drag"); }));
["dragleave", "drop"].forEach(e => zone.addEventListener(e, ev => { ev.preventDefault(); zone.classList.remove("drag"); }));
zone.addEventListener("drop", ev => {
  if (ev.dataTransfer.files.length) { input.files = ev.dataTransfer.files; handle(input.files[0]); }
});
form.addEventListener("submit", () => { btn.disabled = true; btn.classList.add("loading"); btn.textContent = "Analysing"; });
