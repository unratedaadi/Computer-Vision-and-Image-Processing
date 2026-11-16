"use strict";

const byId = (id) => document.getElementById(id);
const samples = byId("samples");
const upload = byId("upload");
const model = byId("model");
const invert = byId("invert");
const preview = byId("preview");
const button = byId("predict");
const result = byId("result");
const status = byId("status");
let previewUrl;
const percentage = (score) => (score * 100).toFixed(1) + "%";

async function getJson(url, options) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error ?? "Request failed.");
  return data;
}

function updateInput() {
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = upload.files[0] ? URL.createObjectURL(upload.files[0]) : undefined;
  const source = previewUrl ?? (samples.value ? "/samples/" + encodeURIComponent(samples.value) : "");
  if (source) preview.src = source;
  else preview.removeAttribute("src");
  preview.hidden = !source;
  preview.classList.toggle("sample", !previewUrl);
  preview.classList.toggle("inverted", invert.checked);
  byId("placeholder").hidden = Boolean(source);
  button.disabled = !source;
  result.hidden = true;
  status.textContent = source ? "Ready to classify." : "Select a test sample or upload an image.";
}

samples.addEventListener("change", () => {
  upload.value = "";
  invert.checked = false;
  updateInput();
});
upload.addEventListener("change", () => {
  if (upload.files[0]) samples.value = "";
  updateInput();
});
model.addEventListener("change", updateInput);
invert.addEventListener("change", updateInput);

function encode(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result.split(",")[1]);
    reader.onerror = () => reject(new Error("Could not read this image."));
    reader.readAsDataURL(file);
  });
}

function renderPrediction(data) {
  byId("digit").textContent = data.scores[0].digit;
  byId("score").textContent = percentage(data.scores[0].score) + " model score";
  byId("actual").textContent = data.actual === null ? data.model : "Actual label: " + data.actual;
  const rows = data.scores.slice(0, 3).map(({digit, score}) => {
    const row = document.createElement("div");
    row.className = "score-row";
    const label = document.createElement("span");
    label.textContent = digit;
    const bar = document.createElement("progress");
    bar.max = 1;
    bar.value = score;
    bar.setAttribute("aria-label", "Score for digit " + digit);
    const value = document.createElement("span");
    value.textContent = percentage(score);
    row.append(label, bar, value);
    return row;
  });
  byId("scores").replaceChildren(...rows);
  result.hidden = false;
}

button.addEventListener("click", async () => {
  const controls = [samples, upload, model, invert, button];
  const file = upload.files[0];
  const payload = {model: model.value, invert: invert.checked};
  controls.forEach((control) => { control.disabled = true; });
  result.hidden = true;
  status.textContent = "Classifying…";
  try {
    if (file && file.size > 5 * 1024 * 1024) throw new Error("Choose an image under 5 MB.");
    if (file) payload.image = await encode(file);
    else payload.sample = samples.value;
    const data = await getJson("/api/predict", {
      method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(payload)
    });
    renderPrediction(data);
    status.textContent = data.model + " prediction";
  } catch (error) {
    status.textContent = error.message;
  } finally {
    controls.forEach((control) => { control.disabled = false; });
  }
});

getJson("/api/samples").then((items) => {
  for (const item of items) samples.add(new Option("Digit " + item.actual + " · #" + item.index, item.file));
}).catch((error) => { status.textContent = error.message; });

getJson("/api/metrics").then((data) => {
  const table = document.createElement("table");
  const header = table.createTHead().insertRow();
  for (const label of ["Model", "Accuracy", "Macro precision", "Macro recall", "Macro F1"]) {
    const cell = document.createElement("th");
    cell.scope = "col";
    cell.textContent = label;
    header.append(cell);
  }
  const body = table.createTBody();
  for (const [key, name] of [["cnn", "Basic CNN"], ["resnet", "ResNet18"]]) {
    const report = data.models[key].classification_report;
    const macro = report["macro avg"];
    const row = body.insertRow();
    for (const value of [name, ...[report.accuracy, macro.precision, macro.recall, macro["f1-score"]].map(percentage)]) {
      row.insertCell().textContent = value;
    }
  }
  byId("metrics").append(table);
}).catch((error) => { byId("metrics").textContent = error.message; });
