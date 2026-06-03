const state = {
  features: [],
  featureByKey: new Map(),
  selectedFeature: null,
  originalBlob: null,
  baseBlob: null,
  previewBlob: null,
  previewFeature: null,
  history: [],
  redo: [],
  objectUrls: [],
  cropStart: null,
  cropEnd: null,
  cropSelecting: false,
};

const el = (id) => document.getElementById(id);
const featureList = el('featureList');
const controlsContainer = el('controlsContainer');
const statusText = el('statusText');
const beforeImage = el('beforeImage');
const afterImage = el('afterImage');
const cropBox = el('cropBox');
const afterStage = el('afterStage');
let debounceTimer = null;

function setStatus(message) {
  statusText.textContent = message;
}

function rememberUrl(url) {
  state.objectUrls.push(url);
  if (state.objectUrls.length > 40) {
    URL.revokeObjectURL(state.objectUrls.shift());
  }
  return url;
}

function blobUrl(blob) {
  return rememberUrl(URL.createObjectURL(blob));
}

function currentBlob() {
  return state.previewBlob || state.baseBlob;
}

function setImageInfoFromImage(img, prefix = '') {
  if (!img || !img.naturalWidth) return;
  el('imageInfo').textContent = `${prefix}${img.naturalWidth} x ${img.naturalHeight} px`;
}

function updateButtons() {
  const hasImage = Boolean(state.baseBlob);
  ['saveBtn', 'previewBtn', 'applyBtn', 'cancelBtn', 'resetBtn', 'histBtn', 'cnnBtn', 'cropBtn'].forEach((id) => {
    el(id).disabled = !hasImage;
  });
  el('undoBtn').disabled = state.history.length === 0;
  el('redoBtn').disabled = state.redo.length === 0;
}

function updateImages() {
  if (!state.baseBlob) {
    beforeImage.removeAttribute('src');
    afterImage.removeAttribute('src');
    updateButtons();
    return;
  }
  const beforeBlob = el('showOriginalBefore').checked ? state.originalBlob : state.baseBlob;
  beforeImage.src = blobUrl(beforeBlob);
  afterImage.src = blobUrl(currentBlob());
  afterImage.onload = () => setImageInfoFromImage(afterImage);
  hideCropBox();
  updateButtons();
}

// Theme helpers: apply theme and initialize from localStorage / prefers-color-scheme
function applyTheme(theme) {
  if (theme === 'light') document.documentElement.setAttribute('data-theme', 'light');
  else document.documentElement.removeAttribute('data-theme');
  const toggle = el('themeToggle');
  if (toggle) toggle.checked = (theme === 'light');
  try { localStorage.setItem('theme', theme); } catch (e) { /* ignore */ }
}

function initTheme() {
  // safe read from localStorage
  let saved = null;
  try { saved = localStorage.getItem('theme'); } catch (e) { /* ignore */ }
  let theme;
  if (saved === 'light' || saved === 'dark') theme = saved;
  else theme = (window.matchMedia && window.matchMedia('(prefers-color-scheme: light)').matches) ? 'light' : 'dark';
  applyTheme(theme);
  // attach listener to toggle if it exists now
  const toggle = el('themeToggle');
  if (toggle) toggle.addEventListener('change', (e) => applyTheme(e.target.checked ? 'light' : 'dark'));
}

// initialize theme early, before wiring other UI listeners
initTheme();

function groupByCategory(features) {
  return features.reduce((groups, feature) => {
    if (!groups.has(feature.category)) groups.set(feature.category, []);
    groups.get(feature.category).push(feature);
    return groups;
  }, new Map());
}

function renderFeatures() {
  featureList.innerHTML = '';
  for (const [category, features] of groupByCategory(state.features)) {
    const title = document.createElement('div');
    title.className = 'category-title';
    title.textContent = category;
    featureList.appendChild(title);
    for (const feature of features) {
      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'feature-button';
      button.dataset.key = feature.key;
      button.innerHTML = `<strong>${feature.name}</strong><small>${feature.description}</small>`;
      button.addEventListener('click', () => selectFeature(feature.key));
      featureList.appendChild(button);
    }
  }
}

function selectFeature(key) {
  const feature = state.featureByKey.get(key);
  if (!feature) return;
  state.selectedFeature = feature;
  document.querySelectorAll('.feature-button').forEach((btn) => btn.classList.toggle('active', btn.dataset.key === key));
  el('featureTitle').textContent = feature.name;
  el('featureDesc').textContent = feature.description;
  renderPresets(feature);
  renderControls(feature);
  state.previewBlob = null;
  state.previewFeature = null;
  updateImages();
  scheduleLivePreview();
}

function renderPresets(feature) {
  const preset = el('presetSelect');
  preset.innerHTML = '<option value="Manual">Manual</option>';
  Object.keys(feature.presets || {}).forEach((name) => {
    const option = document.createElement('option');
    option.value = name;
    option.textContent = name;
    preset.appendChild(option);
  });
  preset.value = 'Manual';
}

function renderControls(feature) {
  controlsContainer.innerHTML = '';
  if (!feature.controls.length) {
    const empty = document.createElement('p');
    empty.className = 'panel-heading';
    empty.textContent = 'Fitur ini tidak membutuhkan parameter tambahan.';
    controlsContainer.appendChild(empty);
    return;
  }
  for (const control of feature.controls) {
    const wrap = document.createElement('div');
    wrap.className = 'control';
    wrap.dataset.key = control.key;
    if (control.kind === 'slider') {
      const label = document.createElement('label');
      label.innerHTML = `<span>${control.label}</span><span class="value-pill">${control.default}</span>`;
      const input = document.createElement('input');
      input.type = 'range';
      input.min = control.minimum;
      input.max = control.maximum;
      input.value = control.default;
      input.step = control.integer ? '1' : '0.1';
      input.dataset.param = control.key;
      input.addEventListener('input', () => {
        label.querySelector('.value-pill').textContent = input.value;
        el('presetSelect').value = 'Manual';
        scheduleLivePreview();
      });
      wrap.append(label, input);
    } else if (control.kind === 'combo') {
      const label = document.createElement('label');
      label.textContent = control.label;
      const select = document.createElement('select');
      select.dataset.param = control.key;
      for (const optionText of control.options) {
        const option = document.createElement('option');
        option.value = optionText;
        option.textContent = optionText;
        select.appendChild(option);
      }
      select.value = control.default;
      select.addEventListener('change', () => {
        el('presetSelect').value = 'Manual';
        scheduleLivePreview();
      });
      wrap.append(label, select);
    } else if (control.kind === 'check') {
      const label = document.createElement('label');
      label.className = 'check-row';
      const input = document.createElement('input');
      input.type = 'checkbox';
      input.checked = Boolean(control.default);
      input.dataset.param = control.key;
      input.addEventListener('change', () => {
        el('presetSelect').value = 'Manual';
        scheduleLivePreview();
      });
      label.append(input, document.createTextNode(control.label));
      wrap.appendChild(label);
    }
    controlsContainer.appendChild(wrap);
  }
}

function getParams() {
  const params = { interpolation: el('interpolationSelect').value };
  controlsContainer.querySelectorAll('[data-param]').forEach((input) => {
    if (input.type === 'checkbox') {
      params[input.dataset.param] = input.checked;
    } else if (input.type === 'range' || input.type === 'number') {
      params[input.dataset.param] = Number(input.value);
    } else {
      params[input.dataset.param] = input.value;
    }
  });
  return params;
}

function applyParams(values) {
  controlsContainer.querySelectorAll('[data-param]').forEach((input) => {
    if (!(input.dataset.param in values)) return;
    if (input.type === 'checkbox') input.checked = Boolean(values[input.dataset.param]);
    else input.value = values[input.dataset.param];
    const pill = input.parentElement?.querySelector?.('.value-pill');
    if (pill) pill.textContent = input.value;
  });
}

function scheduleLivePreview() {
  clearTimeout(debounceTimer);
  if (!state.baseBlob || !state.selectedFeature || !el('livePreview').checked || !state.selectedFeature.live) return;
  debounceTimer = setTimeout(() => processSelected(false), 280);
}

async function processFeature(featureKey, params, sourceBlob = state.baseBlob) {
  if (!sourceBlob) throw new Error('Belum ada gambar.');
  const form = new FormData();
  form.append('image', sourceBlob, 'image.png');
  form.append('feature', featureKey);
  form.append('params', JSON.stringify(params));
  const response = await fetch('/api/process', { method: 'POST', body: form });
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(error.detail || response.statusText);
  }
  const blob = await response.blob();
  const encodedMessage = response.headers.get('X-Image-Message') || '';
  const message = encodedMessage ? decodeURIComponent(encodedMessage) : 'Selesai.';
  return { blob, message };
}

async function processSelected(commit) {
  if (!state.selectedFeature || !state.baseBlob) return;
  try {
    setStatus(`Memproses ${state.selectedFeature.name}...`);
    const { blob, message } = await processFeature(state.selectedFeature.key, getParams());
    state.previewBlob = blob;
    state.previewFeature = state.selectedFeature.key;
    updateImages();
    setStatus(message);
    if (commit) commitPreview();
  } catch (error) {
    setStatus(error.message);
    alert(error.message);
  }
}

function commitPreview() {
  if (!state.previewBlob || !state.baseBlob) return;
  state.history.push(state.baseBlob);
  state.redo = [];
  state.baseBlob = state.previewBlob;
  state.previewBlob = null;
  state.previewFeature = null;
  updateImages();
  setStatus('Perubahan diterapkan ke state edit.');
}

function cancelPreview() {
  state.previewBlob = null;
  state.previewFeature = null;
  updateImages();
  setStatus('Preview dibatalkan.');
}

async function loadImage(file) {
  if (!file) return;
  state.originalBlob = file;
  state.baseBlob = file;
  state.previewBlob = null;
  state.history = [];
  state.redo = [];
  updateImages();
  setStatus(`Loaded: ${file.name}`);
}

async function exportCurrent() {
  const blob = currentBlob();
  if (!blob) return;
  const form = new FormData();
  const format = el('saveFormat').value;
  form.append('image', blob, 'result.png');
  form.append('image_format', format);
  form.append('quality', el('jpegQuality').value || '90');
  const response = await fetch('/api/export', { method: 'POST', body: form });
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(error.detail || response.statusText);
  }
  const outBlob = await response.blob();
  const suffix = format.toLowerCase().replace('jpeg', 'jpg');
  const a = document.createElement('a');
  a.href = blobUrl(outBlob);
  a.download = `mini-photoshop-result.${suffix}`;
  a.click();
  setStatus(`File diekspor sebagai ${format}.`);
}

function undo() {
  if (!state.history.length || !state.baseBlob) return;
  state.redo.push(state.baseBlob);
  state.baseBlob = state.history.pop();
  cancelPreview();
  setStatus('Undo berhasil.');
}

function redo() {
  if (!state.redo.length || !state.baseBlob) return;
  state.history.push(state.baseBlob);
  state.baseBlob = state.redo.pop();
  cancelPreview();
  setStatus('Redo berhasil.');
}

function resetImage() {
  if (!state.originalBlob || !state.baseBlob) return;
  state.history.push(state.baseBlob);
  state.redo = [];
  state.baseBlob = state.originalBlob;
  cancelPreview();
  setStatus('Gambar dikembalikan ke kondisi awal.');
}

function resetParams() {
  if (!state.selectedFeature) return;
  const values = {};
  for (const control of state.selectedFeature.controls) values[control.key] = control.default;
  applyParams(values);
  el('presetSelect').value = 'Manual';
  scheduleLivePreview();
  setStatus('Parameter direset.');
}

function pointOnImage(event) {
  const rect = afterImage.getBoundingClientRect();
  const x = Math.max(0, Math.min(rect.width, event.clientX - rect.left));
  const y = Math.max(0, Math.min(rect.height, event.clientY - rect.top));
  return {
    x: Math.round((x / rect.width) * afterImage.naturalWidth),
    y: Math.round((y / rect.height) * afterImage.naturalHeight),
    screenX: x + rect.left - afterStage.getBoundingClientRect().left,
    screenY: y + rect.top - afterStage.getBoundingClientRect().top,
  };
}

function drawCropBox() {
  if (!state.cropStart || !state.cropEnd) return;
  const left = Math.min(state.cropStart.screenX, state.cropEnd.screenX);
  const top = Math.min(state.cropStart.screenY, state.cropEnd.screenY);
  const width = Math.abs(state.cropStart.screenX - state.cropEnd.screenX);
  const height = Math.abs(state.cropStart.screenY - state.cropEnd.screenY);
  Object.assign(cropBox.style, { left: `${left}px`, top: `${top}px`, width: `${width}px`, height: `${height}px` });
  cropBox.classList.remove('hidden');
}

function hideCropBox() {
  state.cropStart = null;
  state.cropEnd = null;
  state.cropSelecting = false;
  cropBox.classList.add('hidden');
}

async function cropSelection() {
  if (!state.cropStart || !state.cropEnd || !currentBlob()) {
    alert('Drag area pada panel After terlebih dahulu.');
    return;
  }
  const params = {
    x1: state.cropStart.x,
    y1: state.cropStart.y,
    x2: state.cropEnd.x,
    y2: state.cropEnd.y,
  };
  try {
    setStatus('Memproses crop...');
    const { blob, message } = await processFeature('crop', params, currentBlob());
    state.history.push(state.baseBlob);
    state.redo = [];
    state.baseBlob = blob;
    state.previewBlob = null;
    hideCropBox();
    updateImages();
    setStatus(message);
  } catch (error) {
    setStatus(error.message);
    alert(error.message);
  }
}

async function histogramFor(blob) {
  const form = new FormData();
  form.append('image', blob, 'image.png');
  const response = await fetch('/api/histogram', { method: 'POST', body: form });
  if (!response.ok) throw new Error('Gagal mengambil histogram.');
  return response.json();
}

function drawHistogram(before, after) {
  const canvas = el('histCanvas');
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = '#0b1220';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = '#334155';
  ctx.lineWidth = 1;
  for (let i = 0; i <= 8; i++) {
    const y = 30 + i * 55;
    ctx.beginPath();
    ctx.moveTo(40, y);
    ctx.lineTo(canvas.width - 20, y);
    ctx.stroke();
  }
  const keys = ['gray', 'R', 'G', 'B'];
  const colors = { gray: '#e5e7eb', R: '#ef4444', G: '#22c55e', B: '#38bdf8' };
  const areas = [
    { data: before.histograms, label: 'Before', top: 35, height: 205 },
    { data: after.histograms, label: 'After', top: 290, height: 205 },
  ];
  for (const area of areas) {
    ctx.fillStyle = '#94a3b8';
    ctx.fillText(area.label, 46, area.top - 12);
    const max = Math.max(...Object.values(area.data).flat());
    for (const key of keys) {
      if (!area.data[key]) continue;
      ctx.strokeStyle = colors[key];
      ctx.beginPath();
      area.data[key].forEach((value, i) => {
        const x = 40 + (i / 255) * (canvas.width - 70);
        const y = area.top + area.height - (value / max) * area.height;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
    }
  }
}

async function showHistogram() {
  if (!state.originalBlob || !currentBlob()) return;
  try {
    setStatus('Mengambil histogram...');
    const [before, after] = await Promise.all([histogramFor(state.originalBlob), histogramFor(currentBlob())]);
    drawHistogram(before, after);
    el('histDialog').showModal();
    setStatus('Histogram siap.');
  } catch (error) {
    setStatus(error.message);
    alert(error.message);
  }
}

async function runCnn() {
  if (!currentBlob()) return;
  const form = new FormData();
  form.append('image', currentBlob(), 'image.png');
  form.append('model_name', el('cnnModel').value);
  form.append('top_k', el('cnnTopK').value || '5');
  try {
    setStatus('Menjalankan CNN...');
    const response = await fetch('/api/cnn', { method: 'POST', body: form });
    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: response.statusText }));
      throw new Error(error.detail || response.statusText);
    }
    const result = await response.json();
    el('cnnOutput').innerHTML = `<p>Model: <strong>${result.model}</strong></p><ol>${result.predictions.map((p) => `<li>${p.label}: ${(p.confidence * 100).toFixed(2)}%</li>`).join('')}</ol>`;
    el('cnnDialog').showModal();
    setStatus('CNN selesai.');
  } catch (error) {
    el('cnnOutput').textContent = error.message;
    el('cnnDialog').showModal();
    setStatus('CNN belum aktif atau gagal dijalankan.');
  }
}

async function init() {
  const response = await fetch('/api/features');
  const data = await response.json();
  state.features = data.features;
  state.featureByKey = new Map(state.features.map((feature) => [feature.key, feature]));
  renderFeatures();
  selectFeature(state.features[0].key);
  el('cnnModel').innerHTML = data.models.map((name) => `<option value="${name}">${name}</option>`).join('');
  el('cnnModel').value = data.defaultModel;
  updateButtons();
}

el('fileInput').addEventListener('change', (event) => loadImage(event.target.files[0]));
el('saveBtn').addEventListener('click', () => exportCurrent().catch((error) => alert(error.message)));
el('undoBtn').addEventListener('click', undo);
el('redoBtn').addEventListener('click', redo);
el('resetBtn').addEventListener('click', resetImage);
el('previewBtn').addEventListener('click', () => processSelected(false));
el('applyBtn').addEventListener('click', async () => {
  if (state.previewBlob && state.previewFeature === state.selectedFeature?.key) commitPreview();
  else await processSelected(true);
});
el('cancelBtn').addEventListener('click', cancelPreview);
el('resetParamBtn').addEventListener('click', resetParams);
el('cropBtn').addEventListener('click', cropSelection);
el('histBtn').addEventListener('click', showHistogram);
el('cnnBtn').addEventListener('click', runCnn);
el('showOriginalBefore').addEventListener('change', updateImages);
el('livePreview').addEventListener('change', scheduleLivePreview);
el('interpolationSelect').addEventListener('change', scheduleLivePreview);
el('presetSelect').addEventListener('change', (event) => {
  if (!state.selectedFeature || event.target.value === 'Manual') return;
  applyParams(state.selectedFeature.presets[event.target.value] || {});
  scheduleLivePreview();
});
el('closeHist').addEventListener('click', () => el('histDialog').close());
el('closeCnn').addEventListener('click', () => el('cnnDialog').close());

afterStage.addEventListener('pointerdown', (event) => {
  if (!currentBlob() || !afterImage.src) return;
  const point = pointOnImage(event);
  state.cropSelecting = true;
  state.cropStart = point;
  state.cropEnd = point;
  drawCropBox();
});
afterStage.addEventListener('pointermove', (event) => {
  if (!state.cropSelecting) return;
  state.cropEnd = pointOnImage(event);
  drawCropBox();
});
window.addEventListener('pointerup', (event) => {
  if (!state.cropSelecting) return;
  state.cropEnd = pointOnImage(event);
  state.cropSelecting = false;
  drawCropBox();
});

init().catch((error) => {
  setStatus('Gagal memuat fitur: ' + error.message);
  console.error(error);
});
