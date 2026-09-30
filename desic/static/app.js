'use strict';
/* Desic dashboard — dependency-free. All user/LLM-provided strings are
   inserted as text nodes (never innerHTML) so datasets cannot inject markup. */

// ------------------------------------------------------------------ helpers
const $ = (sel, el = document) => el.querySelector(sel);

function h(tag, attrs, ...kids) {
  const el = document.createElement(tag);
  setAttrs(el, attrs);
  append(el, kids);
  return el;
}
function sv(tag, attrs, ...kids) {
  const el = document.createElementNS('http://www.w3.org/2000/svg', tag);
  setAttrs(el, attrs);
  append(el, kids);
  return el;
}
function setAttrs(el, attrs) {
  if (!attrs) return;
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null || v === false) continue;
    if (k === 'class') el.setAttribute('class', v);
    else if (k === 'style' && typeof v === 'object') Object.assign(el.style, v);
    else if (k.startsWith('on') && typeof v === 'function') el.addEventListener(k.slice(2), v);
    else if (k === 'value') el.value = v;
    else if (k === 'checked' || k === 'selected' || k === 'disabled' || k === 'open') el[k] = !!v;
    else el.setAttribute(k, v === true ? '' : v);
  }
}
function append(el, kids) {
  for (const kid of kids.flat(Infinity)) {
    if (kid == null || kid === false) continue;
    el.append(kid instanceof Node ? kid : document.createTextNode(String(kid)));
  }
}

async function api(method, path, body, isForm) {
  const opts = { method, headers: {} };
  if (body !== undefined) {
    if (isForm) opts.body = body;
    else { opts.headers['Content-Type'] = 'application/json'; opts.body = JSON.stringify(body); }
  }
  const r = await fetch(path, opts);
  if (r.status === 204) return null;
  const data = await r.json().catch(() => null);
  if (!r.ok) {
    let msg = data && data.detail;
    if (Array.isArray(msg)) msg = msg.map(d => `${(d.loc || []).slice(1).join('.')}: ${d.msg}`).join('; ');
    throw new Error(msg || r.statusText);
  }
  return data;
}

const pct = v => (v == null ? '—' : (v * 100).toFixed(1) + '%');
const num = v => (v == null ? '—' : Number(v).toLocaleString());
const enc = encodeURIComponent;
function fmtVal(v) {
  if (v == null) return '∅';
  if (typeof v === 'number') return Number.isInteger(v) ? String(v) : String(+v.toFixed(3));
  if (Array.isArray(v)) return v.join(', ');
  return String(v);
}
function ago(t) {
  const s = Math.max(0, Date.now() / 1000 - t);
  if (s < 60) return `${Math.floor(s)}s ago`;
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  return new Date(t * 1000).toLocaleDateString();
}
function debounce(fn, ms) {
  let t;
  return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); };
}
function toast(msg, kind = 'info', ms = 4500) {
  const el = h('div', { class: `toast ${kind}` }, msg);
  $('#dock').append(el);
  setTimeout(() => el.remove(), ms);
}
async function guard(btn, fn) {
  if (btn) btn.disabled = true;
  try { return await fn(); }
  catch (e) { toast(e.message, 'error', 7000); }
  finally { if (btn) btn.disabled = false; }
}
function condText(c) {
  const v = Array.isArray(c.value) ? `[${c.value.join(', ')}]` : fmtVal(c.value);
  const op = { '<=': '≤', '>=': '≥', '!=': '≠', not_in: 'not in', is_missing: 'is missing' }[c.op] || c.op;
  return c.op === 'is_missing' ? `${c.feature} ${op}` : `${c.feature} ${op} ${v}`;
}
function barRows(entries, fmt = pct) {
  const max = Math.max(...entries.map(e => e[1]), 1e-9);
  return entries.map(([name, v]) => h('div', { class: 'bar-row', title: `${name}: ${fmt(v)}` },
    h('span', { class: 'name' }, name),
    h('div', { class: 'track' }, h('div', { class: 'fill', style: { width: `${(v / max) * 100}%` } })),
    h('span', { class: 'val' }, fmt(v))));
}

// ------------------------------------------------------------------ state
const state = {
  view: null,           // { kind: 'model', name } | { kind: 'models' } | ...
  model: null,          // current model detail
  tab: 'feed',
  jobWatchers: new Map(),
  memberSel: null,
};

// ------------------------------------------------------------------ realtime
function connect() {
  const ws = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`);
  const live = $('#live');
  ws.onopen = () => { live.classList.add('on'); $('.label', live).textContent = 'live'; };
  ws.onclose = () => {
    live.classList.remove('on'); $('.label', live).textContent = 'offline — retrying';
    setTimeout(connect, 2000);
  };
  ws.onmessage = e => onEvent(JSON.parse(e.data));
}

const isCurrent = name => state.view && state.view.kind === 'model' && state.view.name === name;
const refreshModelSoon = debounce(() => refreshModel(), 500);
const refreshListSoon = debounce(() => { if (state.view && state.view.kind === 'models') viewModels(); }, 800);

function onEvent(ev) {
  switch (ev.type) {
    case 'job': onJob(ev.job); break;
    case 'decision':
      if (isCurrent(ev.model)) { feedPrepend(ev.decision); refreshModelSoon(); }
      refreshListSoon();
      break;
    case 'feedback':
      if (isCurrent(ev.model)) { markFeedItem(ev.decision_id, ev.label, ev.correct); refreshModelSoon(); }
      break;
    case 'metrics':
      if (isCurrent(ev.model)) refreshModelSoon();
      refreshListSoon();
      break;
    case 'drift':
      if (isCurrent(ev.model) && ev.event.type === 'drift')
        toast(`Concept drift detected in ${ev.model} — a fresher tree took over.`, 'warn');
      break;
    case 'model_created': case 'model_deleted': case 'model_reset':
      refreshListSoon();
      if (isCurrent(ev.model)) { if (ev.type === 'model_deleted') location.hash = '#/models'; else refreshModelSoon(); }
      break;
    case 'dataset_created':
      if (state.view && state.view.kind === 'datasets') viewDatasets();
      break;
  }
}

function onJob(job) {
  const w = state.jobWatchers.get(job.id);
  if (w) w(job);
  if (job.status === 'done' && !w) toast(`${job.kind === 'train' ? 'Training' : 'Generation'} finished: ${job.message}`, 'good');
  if (job.status === 'error' && !w) toast(`${job.kind} failed: ${job.error}`, 'error', 8000);
  if (job.status !== 'running') setTimeout(() => state.jobWatchers.delete(job.id), 1000);
}
function jobPanel(onDone) {
  const bar = h('div', { style: { width: '0%' } });
  const msg = h('div', { class: 'small muted' }, 'starting…');
  const out = h('div');
  const el = h('div', {}, h('div', { class: 'progress' }, bar), msg, out);
  return {
    el,
    watch(job) {
      let finished = false;
      const update = j => {
        if (finished) return;
        bar.style.width = `${Math.round((j.progress || 0) * 100)}%`;
        msg.textContent = j.error ? `Error: ${j.error}` : j.message;
        if (j.status === 'error') { msg.style.color = 'var(--critical)'; finished = true; }
        if (j.status === 'done') { finished = true; onDone(j, out); }
      };
      state.jobWatchers.set(job.id, update);
      // the job may have progressed before we started listening
      api('GET', `/api/jobs/${job.id}`).then(update).catch(() => {});
    },
  };
}

// ------------------------------------------------------------------ router
function route() {
  const parts = location.hash.replace(/^#\/?/, '').split('/').filter(Boolean).map(decodeURIComponent);
  const section = parts[0] || 'models';
  document.querySelectorAll('[data-nav]').forEach(a => a.classList.toggle('active', a.dataset.nav === section));
  if (section === 'models' && parts[1]) return viewModel(parts[1]);
  if (section === 'datasets' && parts[1]) return viewDataset(parts[1]);
  if (section === 'datasets') return viewDatasets();
  if (section === 'generate') return viewGenerate();
  return viewModels();
}
function mount(...kids) {
  const main = $('#main');
  main.replaceChildren(...kids.flat().filter(k => k != null && k !== false));
  return main;
}

// ================================================================== models
async function viewModels() {
  const first = !state.view || state.view.kind !== 'models';
  state.view = { kind: 'models' };
  const models = await api('GET', '/api/models').catch(e => { toast(e.message, 'error'); return []; });
  if (!state.view || state.view.kind !== 'models') return;
  const newBtn = h('button', { class: 'primary', onclick: () => showCreateForm() }, '+ New model');
  const cards = models.length ? h('div', { class: 'grid-cards' }, models.map(modelCard)) : h('div', { class: 'empty' },
    h('p', {}, 'No models yet.'),
    h('p', {}, 'Create one from scratch, ', h('a', { href: '#/datasets' }, 'train one from your dataset'),
      ' or ', h('a', { href: '#/generate' }, 'generate data with your own AI key'), '.'));
  const formSlot = h('div', { id: 'create-slot' });
  if (first || !$('#create-slot')) {
    mount(
      h('div', { class: 'page-head' },
        h('div', {}, h('h1', {}, 'Decision models'),
          h('div', { class: 'sub' }, 'Explainable models that keep learning from every piece of feedback.')),
        newBtn),
      formSlot, cards);
  } else {
    // live refresh: keep an open create form, swap the cards only
    const main = $('#main');
    main.replaceChild(cards, main.lastElementChild);
  }
}

function modelCard(m) {
  return h('div', { class: 'panel model-card', onclick: () => { location.hash = `#/models/${enc(m.name)}`; } },
    h('div', { class: 'row', style: { justifyContent: 'space-between' } },
      h('div', { class: 'title' }, m.name), h('span', { class: 'badge accent' }, m.kind)),
    h('div', { class: 'muted small' }, `decides “${m.target}” · ${m.n_features} features · ${m.classes.length} classes`),
    m.description ? h('div', { class: 'small', style: { marginTop: '6px' } }, m.description) : null,
    h('div', { class: 'meta' },
      h('div', {}, h('b', {}, pct(m.rolling_accuracy)), h('span', { class: 'small muted' }, 'rolling acc.')),
      h('div', {}, h('b', {}, num(m.learned)), h('span', { class: 'small muted' }, 'learned')),
      h('div', {}, h('b', {}, num(m.decisions)), h('span', { class: 'small muted' }, 'decisions'))));
}

function showCreateForm() {
  const slot = $('#create-slot');
  if (!slot || slot.firstChild) return;
  const feats = h('div');
  const addFeat = (name = '', type = 'numeric', values = '') => {
    const row = h('div', { class: 'rule-card' },
      h('div', { class: 'cond-row', style: { gridTemplateColumns: '2fr 1fr 3fr auto' } },
        h('input', { placeholder: 'feature name', value: name, 'data-k': 'name' }),
        h('select', { 'data-k': 'type' }, h('option', { value: 'numeric', selected: type === 'numeric' }, 'numeric'),
          h('option', { value: 'categorical', selected: type === 'categorical' }, 'categorical')),
        h('input', { placeholder: 'categories (comma separated, optional)', value: values, 'data-k': 'values' }),
        h('button', { class: 'ghost sm', title: 'Remove', onclick: () => row.remove() }, '✕')));
    feats.append(row);
  };
  addFeat('income', 'numeric'); addFeat('age', 'numeric'); addFeat('city', 'categorical', 'istanbul, ankara, izmir');
  const name = h('input', { placeholder: 'e.g. loan_approval' });
  const target = h('input', { placeholder: 'e.g. decision', value: 'decision' });
  const classes = h('input', { placeholder: 'e.g. approve, reject', value: 'approve, reject' });
  const kind = h('select', {}, h('option', { value: 'tree' }, 'Adaptive tree (most explainable)'),
    h('option', { value: 'forest' }, 'Adaptive random forest (more accurate)'));
  const desc = h('input', { placeholder: 'optional' });
  const create = h('button', { class: 'primary' }, 'Create model');
  create.onclick = () => guard(create, async () => {
    const features = [...feats.children].map(r => ({
      name: $('[data-k=name]', r).value.trim(),
      type: $('[data-k=type]', r).value,
      values: $('[data-k=values]', r).value.split(',').map(s => s.trim()).filter(Boolean),
    })).filter(f => f.name);
    const m = await api('POST', '/api/models', {
      name: name.value.trim(), target: target.value.trim(), features, kind: kind.value, description: desc.value,
      classes: classes.value.split(',').map(s => s.trim()).filter(Boolean),
    });
    location.hash = `#/models/${enc(m.name)}`;
  });
  slot.append(h('div', { class: 'panel', style: { marginBottom: '20px' } },
    h('div', { class: 'panel-head' }, h('h2', {}, 'New model'), h('button', { class: 'ghost sm', onclick: () => slot.replaceChildren() }, 'Cancel')),
    h('div', { class: 'form-grid' },
      h('label', { class: 'field' }, h('span', {}, 'Name'), name),
      h('label', { class: 'field' }, h('span', {}, 'Decision (target) name'), target),
      h('label', { class: 'field' }, h('span', {}, 'Known decisions'), classes),
      h('label', { class: 'field' }, h('span', {}, 'Model type'), kind),
      h('label', { class: 'field' }, h('span', {}, 'Description'), desc)),
    h('h3', {}, 'Features'), feats,
    h('div', { class: 'row' }, h('button', { onclick: () => addFeat() }, '+ Add feature'), create),
    h('p', { class: 'note' }, 'New decisions (classes) can also appear later — just give feedback with a new label.')));
}

// ================================================================== model detail
async function viewModel(name) {
  const changed = !state.view || state.view.kind !== 'model' || state.view.name !== name;
  state.view = { kind: 'model', name };
  if (changed) { state.tab = 'feed'; state.memberSel = null; state.lastDecision = null; }
  let m;
  try { m = await api('GET', `/api/models/${enc(name)}`); }
  catch (e) { mount(h('div', { class: 'empty' }, e.message, ' — ', h('a', { href: '#/models' }, 'back to models'))); return; }
  state.model = m;

  const kpis = h('div', { class: 'kpis', id: 'kpis' });
  const chart = h('div', { id: 'chart-slot' });
  const importance = h('div', { id: 'imp-slot' });
  const tabBody = h('div', { id: 'tab-body' });
  const tabs = h('div', { class: 'tabs', role: 'tablist' });
  const TABS = [['feed', 'Live feed'], ['review', 'Review queue'], ['learned', 'Learned rules'], ['rules', 'Hard rules'],
    ['tree', 'Tree'], ['perf', 'Performance'], ['api', 'Integrate']];
  for (const [id, label] of TABS) {
    tabs.append(h('button', { role: 'tab', class: state.tab === id ? 'active' : '', 'data-tab': id, onclick: () => {
      state.tab = id;
      tabs.querySelectorAll('button').forEach(b => b.classList.toggle('active', b.dataset.tab === id));
      renderTab();
    } }, label));
  }

  mount(
    h('div', { class: 'page-head' },
      h('div', {},
        h('div', { class: 'row' }, h('h1', {}, m.name), h('span', { class: 'badge accent' }, m.kind),
          h('span', { class: 'badge' }, `target: ${m.target}`)),
        h('div', { class: 'sub' }, m.description || 'Make decisions, correct them, and watch the model adapt in real time.')),
      h('div', { class: 'row' },
        h('button', { onclick: e => guard(e.currentTarget, async () => {
          if (!confirm(`Forget everything ${m.name} has learned? Rules and schema are kept.`)) return;
          await api('POST', `/api/models/${enc(m.name)}/reset`); toast('Model reset', 'good'); refreshModel();
        }) }, 'Reset learning'),
        h('button', { class: 'danger', onclick: e => guard(e.currentTarget, async () => {
          if (!confirm(`Delete model ${m.name} and its decision log?`)) return;
          await api('DELETE', `/api/models/${enc(m.name)}`); location.hash = '#/models';
        }) }, 'Delete'))),
    kpis,
    h('div', { class: 'grid-2' },
      decidePanel(m),
      h('div', { class: 'stack' },
        h('div', { class: 'panel' }, h('div', { class: 'panel-head' }, h('h2', {}, 'Accuracy over time'),
          h('span', { class: 'small muted' }, 'prequential: each label is scored before it is learned')), chart),
        h('div', { class: 'panel' }, h('div', { class: 'panel-head' }, h('h2', {}, 'What drives decisions'),
          h('span', { class: 'small muted' }, 'share of information gain')), importance))),
    tabs, tabBody);
  renderLive(m);
  renderTab();
}

async function refreshModel() {
  if (!state.view || state.view.kind !== 'model') return;
  try { state.model = await api('GET', `/api/models/${enc(state.view.name)}`); }
  catch { return; }
  renderLive(state.model);
  if (state.tab === 'perf') renderTab();
}

function renderLive(m) {
  const kpis = $('#kpis');
  if (!kpis) return;
  const s = m.structure, c = m.counters;
  const kpi = (k, v, d) => h('div', { class: 'kpi' }, h('div', { class: 'k' }, k), h('div', { class: 'v' }, v), h('div', { class: 'd' }, d));
  kpis.replaceChildren(
    kpi('Rolling accuracy', pct(m.metrics.rolling_accuracy), `last ${num(m.metrics.window)} labels`),
    kpi('Overall accuracy', pct(m.metrics.accuracy), `${num(m.metrics.evaluated)} labels scored`),
    kpi('Decision accuracy', pct(c.decision_accuracy), 'incl. hard rules, from feedback'),
    kpi('Awaiting feedback', num(c.pending), `${num(c.decisions)} decisions total`),
    kpi(m.kind === 'forest' ? 'Forest size' : 'Tree size', `${num(s.nodes)} nodes`, `${num(s.leaves)} leaves · depth ${s.depth}`),
    kpi('Drifts adapted', num(s.drifts), `${num(s.warnings)} warnings`));
  $('#chart-slot').replaceChildren(accuracyChart(m.history));
  const imp = Object.entries(m.importance);
  $('#imp-slot').replaceChildren(imp.length ? h('div', {}, barRows(imp))
    : h('div', { class: 'empty' }, 'The model has not split on any feature yet — it needs more labelled examples.'));
}

// ------------------------------------------------------------------ decide panel
function featureInput(f, value) {
  if (f.type === 'categorical') {
    const id = `dl-${f.name.replace(/\W/g, '_')}`;
    return [h('input', { name: f.name, list: id, value: value ?? '', placeholder: f.values.slice(0, 3).join(' / ') || 'text' }),
      h('datalist', { id }, f.values.map(v => h('option', { value: v })))];
  }
  return h('input', { name: f.name, type: 'number', step: 'any', value: value ?? '', placeholder: 'number' });
}

function decidePanel(m) {
  const form = h('form', { class: 'form-grid' },
    m.schema.features.map(f => h('label', { class: 'field' }, h('span', { title: f.description || '' }, f.name), featureInput(f))));
  const result = h('div');
  const submit = h('button', { class: 'primary', type: 'submit' }, 'Decide');
  const outer = h('form', { onsubmit: e => {
    e.preventDefault();
    guard(submit, async () => {
      const features = {};
      for (const el of form.querySelectorAll('input[name]')) {
        if (el.value === '') continue;
        features[el.name] = el.type === 'number' ? Number(el.value) : el.value;
      }
      const d = await api('POST', `/api/models/${enc(m.name)}/decide`, { features });
      state.lastDecision = d;
      result.replaceChildren(decisionResult(d));
    });
  } }, form, h('div', { class: 'row' }, submit,
    h('button', { type: 'button', onclick: () => { form.querySelectorAll('input').forEach(i => { i.value = ''; }); result.replaceChildren(); } }, 'Clear')));
  if (state.lastDecision) result.append(decisionResult(state.lastDecision));
  return h('div', { class: 'panel' },
    h('div', { class: 'panel-head' }, h('h2', {}, 'Make a decision'), h('span', { class: 'small muted' }, 'empty fields = missing')),
    outer, result);
}

function decisionResult(d) {
  const e = d.explanation || {};
  const probs = Object.entries(d.probabilities || {});
  const verdict = d.prediction == null
    ? h('div', { class: 'verdict' }, h('span', { class: 'label muted' }, 'No idea yet'),
      h('span', { class: 'muted small' }, 'Tell Desic the right answer below and it will start learning.'))
    : h('div', { class: 'verdict' }, h('span', { class: 'label' }, d.prediction),
      h('span', { class: 'badge' }, `${pct(d.confidence)} confident`),
      d.source === 'rule' ? h('span', { class: 'badge warn' }, `hard rule: ${d.rule.name || d.rule.id}`) : h('span', { class: 'badge accent' }, 'learned model'));
  return h('div', { class: 'result' }, verdict,
    d.source === 'rule' && d.model_prediction != null && d.model_prediction !== d.prediction
      ? h('p', { class: 'small muted' }, `The learned model alone would have said “${d.model_prediction}”.`) : null,
    probs.length ? h('div', { class: 'probs' }, barRows(probs)) : null,
    e.path && e.path.length ? h('div', {},
      h('h3', {}, 'Why'),
      h('ul', { class: 'path' }, e.path.map(s => h('li', {}, h('code', {}, condText(s)),
        s.missing ? h('span', { class: 'small muted' }, ' (missing → followed the larger branch)')
          : h('span', { class: 'small muted' }, `  (was ${fmtVal(s.observed)})`)))),
      h('div', { class: 'small muted' }, `Leaf built from ${num(e.leaf_support)} examples · ${e.leaf_method === 'naive_bayes' ? 'naive Bayes leaf' : 'majority vote'}`,
        e.votes ? ` · votes: ${Object.entries(e.votes).map(([k, v]) => `${k} ${v}`).join(', ')}` : '')) : null,
    feedbackBox(d));
}

function feedbackBox(d) {
  const box = h('div', { class: 'feedback-box' });
  const classes = (state.model && state.model.classes) || [];
  const send = async (label, btn) => guard(btn, async () => {
    const r = await api('POST', `/api/models/${enc(d.model)}/feedback`, { decision_id: d.id, label });
    box.replaceChildren(h('div', { class: 'done' },
      r.correct ? '✓ Confirmed — the model reinforced this decision.' : `✓ Learned: the right answer was “${label}”.`),
      h('div', { class: 'small muted' }, `Rolling accuracy now ${pct(r.metrics.rolling_accuracy)}.`));
  });
  const other = h('input', { placeholder: 'another decision…', style: { width: '160px' } });
  box.append(h('strong', {}, 'Was this right?'),
    h('div', { class: 'row' },
      d.prediction != null ? h('button', { class: 'chip yes', onclick: e => send(d.prediction, e.currentTarget) }, `✓ Yes, ${d.prediction}`) : null,
      classes.filter(c => c !== d.prediction).map(c => h('button', { class: 'chip', onclick: e => send(c, e.currentTarget) }, `No → ${c}`)),
      other, h('button', { class: 'sm', onclick: e => other.value.trim() && send(other.value.trim(), e.currentTarget) }, 'Teach')));
  return box;
}

// ------------------------------------------------------------------ chart
function accuracyChart(history) {
  if (!history || history.length < 2) {
    return h('div', { class: 'empty' }, 'No feedback yet. Every label you give (or every training row) adds a point here.');
  }
  const W = Math.max(320, ($('#chart-slot') && $('#chart-slot').clientWidth) || 600), H = 220;
  const P = { l: 40, r: 64, t: 8, b: 24 };
  const n0 = history[0].n, n1 = history[history.length - 1].n;
  const x = n => P.l + ((n - n0) / Math.max(n1 - n0, 1)) * (W - P.l - P.r);
  const y = v => P.t + (1 - v) * (H - P.t - P.b);
  const series = [['rolling', 'Rolling', 'var(--series-1)'], ['overall', 'Overall', 'var(--series-2)']];
  const svg = sv('svg', { viewBox: `0 0 ${W} ${H}`, height: H, role: 'img', 'aria-label': 'Accuracy over time' });
  const grid = sv('g', { class: 'grid axis' });
  for (const t of [0, 0.25, 0.5, 0.75, 1]) {
    grid.append(sv('line', { x1: P.l, x2: W - P.r, y1: y(t), y2: y(t) }),
      sv('text', { x: P.l - 6, y: y(t) + 4, 'text-anchor': 'end' }, `${t * 100}%`));
  }
  grid.append(sv('text', { x: P.l, y: H - 6 }, num(n0)), sv('text', { x: W - P.r, y: H - 6, 'text-anchor': 'end' }, `${num(n1)} labels`));
  svg.append(grid);
  const last = history[history.length - 1];
  const labelYs = [];
  for (const [key, label, color] of series) {
    const d = history.map((p, i) => `${i ? 'L' : 'M'}${x(p.n).toFixed(1)},${y(p[key]).toFixed(1)}`).join('');
    svg.append(sv('path', { class: 'line', d, stroke: color }));
    let ly = y(last[key]) + 4;
    for (const o of labelYs) if (Math.abs(o - ly) < 13) ly = o + (ly >= o ? 13 : -13);
    labelYs.push(ly);
    svg.append(sv('text', { class: 'dlabel', x: W - P.r + 6, y: ly }, `${label} ${pct(last[key])}`));
  }
  const cross = sv('line', { class: 'crosshair', y1: P.t, y2: H - P.b, visibility: 'hidden' });
  const dots = series.map(([, , color]) => sv('circle', { r: 4, fill: color, stroke: 'var(--surface)', 'stroke-width': 2, visibility: 'hidden' }));
  svg.append(cross, ...dots);
  const tip = h('div', { class: 'tip', style: { display: 'none' } });
  const hit = sv('rect', { x: P.l, y: P.t, width: W - P.l - P.r, height: H - P.t - P.b, fill: 'transparent' });
  const hide = () => { tip.style.display = 'none'; cross.setAttribute('visibility', 'hidden'); dots.forEach(d => d.setAttribute('visibility', 'hidden')); };
  hit.addEventListener('pointermove', ev => {
    const rect = svg.getBoundingClientRect();
    const px = ((ev.clientX - rect.left) / rect.width) * W;
    let best = history[0];
    for (const p of history) if (Math.abs(x(p.n) - px) < Math.abs(x(best.n) - px)) best = p;
    const cx = x(best.n);
    cross.setAttribute('x1', cx); cross.setAttribute('x2', cx); cross.setAttribute('visibility', 'visible');
    series.forEach(([key], i) => { dots[i].setAttribute('cx', cx); dots[i].setAttribute('cy', y(best[key])); dots[i].setAttribute('visibility', 'visible'); });
    tip.replaceChildren(h('div', {}, h('strong', {}, `after ${num(best.n)} labels`)),
      h('div', {}, `Rolling: ${pct(best.rolling)}`), h('div', {}, `Overall: ${pct(best.overall)}`));
    tip.style.display = 'block';
    tip.style.left = `${(cx / W) * 100}%`;
    tip.style.top = `${svg.offsetTop + (Math.min(y(best.rolling), y(best.overall)) / H) * rect.height - 8}px`;
  });
  hit.addEventListener('pointerleave', hide);
  svg.append(hit);
  const legend = h('div', { class: 'legend' }, series.map(([, label, color]) => h('span', {}, h('i', { style: { background: color } }), label === 'Rolling' ? 'Rolling (recent window)' : 'Overall')));
  return h('div', { class: 'chart' }, legend, svg, tip);
}
window.addEventListener('resize', debounce(() => { if (state.model && $('#chart-slot')) $('#chart-slot').replaceChildren(accuracyChart(state.model.history)); }, 200));

// ------------------------------------------------------------------ tabs
function renderTab() {
  const body = $('#tab-body');
  if (!body || !state.model) return;
  const m = state.model;
  const render = { feed: tabFeed, review: tabReview, learned: tabLearned, rules: tabRules, tree: tabTree, perf: tabPerf, api: tabApi }[state.tab];
  body.replaceChildren(h('div', { class: 'muted' }, 'loading…'));
  Promise.resolve(render(m)).then(el => { if ($('#tab-body') === body) body.replaceChildren(el); })
    .catch(e => body.replaceChildren(h('div', { class: 'empty' }, e.message)));
}

function feedItem(d, fresh) {
  const feats = Object.entries(d.features).filter(([, v]) => v != null).map(([k, v]) => `${k}=${fmtVal(v)}`).join(' · ');
  const actions = h('div', { class: 'actions', 'data-actions': '' });
  if (d.label != null) {
    actions.append(h('span', { class: `badge ${d.correct ? 'good' : 'bad'}` }, d.correct ? '✓ correct' : `✗ was ${d.label}`));
  } else {
    const classes = (state.model && state.model.classes) || [];
    const send = (label, btn) => guard(btn, () => api('POST', `/api/models/${enc(d.model)}/feedback`, { decision_id: d.id, label }));
    if (d.prediction != null) actions.append(h('button', { class: 'chip yes', title: 'Confirm', onclick: e => send(d.prediction, e.currentTarget) }, '✓'));
    for (const c of classes.filter(c => c !== d.prediction)) actions.append(h('button', { class: 'chip', title: `Correct to ${c}`, onclick: e => send(c, e.currentTarget) }, `→ ${c}`));
  }
  return h('div', { class: `feed-item${fresh ? ' new' : ''}`, 'data-id': d.id },
    h('div', { style: { minWidth: 0 } },
      h('div', {}, h('span', { class: 'dec' }, d.prediction ?? 'no idea'), ' ',
        h('span', { class: 'small muted' }, `${pct(d.confidence)} · ${d.source} · ${ago(d.created_at)}`)),
      h('div', { class: 'feats', title: feats }, feats || '(no features)')),
    actions);
}
function feedPrepend(d) {
  const list = $('#feed-list');
  if (!list || state.tab !== 'feed') return;
  const empty = $('.empty', list);
  if (empty) empty.remove();
  list.prepend(feedItem(d, true));
  while (list.children.length > 100) list.lastElementChild.remove();
}
function markFeedItem(id, label, correct) {
  document.querySelectorAll(`[data-id="${CSS.escape(id)}"] [data-actions]`).forEach(a =>
    a.replaceChildren(h('span', { class: `badge ${correct ? 'good' : 'bad'}` }, correct ? '✓ correct' : `✗ was ${label}`)));
  if (state.tab === 'review') document.querySelectorAll(`#review-list [data-id="${CSS.escape(id)}"]`).forEach(el => el.remove());
}

async function tabFeed(m) {
  const items = await api('GET', `/api/models/${enc(m.name)}/decisions?limit=50`);
  return h('div', {},
    h('p', { class: 'small muted' }, 'Decisions appear here the moment they are made — from this dashboard or from the API. Click ✓ or the right answer to teach the model.'),
    h('div', { id: 'feed-list' }, items.length ? items.map(d => feedItem(d)) : h('div', { class: 'empty' }, 'No decisions yet.')));
}

async function tabReview(m) {
  const items = await api('GET', `/api/models/${enc(m.name)}/decisions?limit=50&pending=true&uncertain_first=true`);
  return h('div', {},
    h('p', { class: 'small muted' }, 'Active learning: unlabelled decisions the model was least sure about come first — labelling these teaches it the most.'),
    h('div', { id: 'review-list' }, items.length ? items.map(d => feedItem(d)) : h('div', { class: 'empty' }, 'Nothing waiting for review. 🎉')));
}

function ruleText(conds, decision, extra) {
  return h('div', { class: 'rule-text' },
    h('span', { class: 'kw' }, 'IF '),
    conds.length ? conds.map((c, i) => [i ? h('span', { class: 'kw' }, ' AND ') : null, condText(c)]) : 'always',
    h('span', { class: 'kw' }, ' THEN '), decision, extra ? h('span', { class: 'muted' }, `  ${extra}`) : null);
}

async function tabLearned(m) {
  const rules = await api('GET', `/api/models/${enc(m.name)}/learned-rules`);
  if (!rules.length) return h('div', { class: 'empty' }, 'Nothing learned yet.');
  const pin = (r, btn) => guard(btn, async () => {
    if (!r.conditions.length) throw new Error('This rule has no conditions — it would match everything.');
    const rules = [...state.model.rules, { name: `pinned: ${r.prediction}`, conditions: r.conditions, decision: r.prediction, priority: 0, enabled: true }];
    await api('PUT', `/api/models/${enc(m.name)}/rules`, { rules });
    await refreshModel();
    toast('Pinned as a hard rule — it now overrides the model.', 'good');
  });
  return h('div', {},
    h('p', { class: 'small muted' }, m.kind === 'forest' ? 'Rules extracted from the most accurate tree in the forest.' : 'Every path of the tree, as a human-readable rule. Pin one to freeze it as a hard rule.'),
    h('div', { class: 'table-wrap' }, h('table', {},
      h('thead', {}, h('tr', {}, h('th', {}, 'Rule'), h('th', { class: 'num' }, 'Confidence'), h('th', { class: 'num' }, 'Support'), h('th', {}))),
      h('tbody', {}, rules.map(r => h('tr', {},
        h('td', {}, ruleText(r.conditions, r.prediction)),
        h('td', { class: 'num' }, pct(r.confidence)),
        h('td', { class: 'num' }, num(Math.round(r.support))),
        h('td', {}, h('button', { class: 'sm', onclick: e => pin(r, e.currentTarget) }, 'Pin'))))))));
}

function tabRules(m) {
  const OPS = ['==', '!=', '>', '>=', '<', '<=', 'in', 'not_in', 'contains', 'is_missing'];
  const list = h('div');
  const featNames = m.schema.features.map(f => f.name);
  const condRow = c => {
    const row = h('div', { class: 'cond-row' },
      h('select', { 'data-k': 'feature' }, featNames.map(n => h('option', { value: n, selected: n === c.feature }, n))),
      h('select', { 'data-k': 'op' }, OPS.map(o => h('option', { value: o, selected: o === (c.op || '==') }, o))),
      h('input', { 'data-k': 'value', value: Array.isArray(c.value) ? c.value.join(', ') : (c.value ?? ''), placeholder: 'value (comma list for in)' }),
      h('button', { class: 'ghost sm', onclick: () => row.remove() }, '✕'));
    return row;
  };
  const ruleCard = r => {
    const conds = h('div', {}, (r.conditions || []).map(condRow));
    const card = h('div', { class: 'rule-card', 'data-id': r.id || '' },
      h('div', { class: 'form-grid' },
        h('label', { class: 'field' }, h('span', {}, 'Name'), h('input', { 'data-k': 'name', value: r.name || '' })),
        h('label', { class: 'field' }, h('span', {}, 'Then decide'), h('input', { 'data-k': 'decision', value: r.decision || '', list: 'dl-classes' })),
        h('label', { class: 'field' }, h('span', {}, 'Priority'), h('input', { 'data-k': 'priority', type: 'number', value: r.priority ?? 0 })),
        h('label', { class: 'field check', style: { marginTop: '22px' } }, h('input', { type: 'checkbox', 'data-k': 'enabled', checked: r.enabled !== false }), 'enabled')),
      h('div', { class: 'small muted', style: { marginBottom: '6px' } }, 'All conditions must match:'),
      conds,
      h('div', { class: 'row', style: { justifyContent: 'space-between' } },
        h('button', { class: 'sm', onclick: () => conds.append(condRow({ feature: featNames[0] })) }, '+ condition'),
        h('span', { class: 'small muted' }, r.id ? `used ${num(r.hits || 0)}× · confirmed ${num(r.confirmed || 0)} · overridden ${num(r.overridden || 0)}` : 'new'),
        h('button', { class: 'sm danger', onclick: () => card.remove() }, 'Remove rule')));
    return card;
  };
  list.append(...m.rules.map(ruleCard));
  const read = () => [...list.children].map(card => ({
    id: card.dataset.id || undefined,
    name: $('[data-k=name]', card).value,
    decision: $('[data-k=decision]', card).value.trim(),
    priority: Number($('[data-k=priority]', card).value || 0),
    enabled: $('[data-k=enabled]', card).checked,
    conditions: [...card.querySelectorAll('.cond-row')].map(r => {
      const op = $('[data-k=op]', r).value, raw = $('[data-k=value]', r).value.trim();
      const value = (op === 'in' || op === 'not_in') ? raw.split(',').map(s => s.trim()).filter(Boolean)
        : (raw !== '' && !isNaN(Number(raw)) ? Number(raw) : raw);
      return { feature: $('[data-k=feature]', r).value, op, value };
    }),
  }));
  const save = h('button', { class: 'primary' }, 'Save rules');
  save.onclick = () => guard(save, async () => {
    await api('PUT', `/api/models/${enc(m.name)}/rules`, { rules: read() });
    await refreshModel(); renderTab(); toast('Rules saved', 'good');
  });
  return h('div', {},
    h('p', { class: 'small muted' }, 'Hard rules are checked first (highest priority wins). Use them for policy, compliance or known edge cases — the model still learns from feedback on rule-made decisions and the dashboard shows how often humans override each rule.'),
    h('datalist', { id: 'dl-classes' }, m.classes.map(c => h('option', { value: c }))),
    list.children.length ? null : h('div', { class: 'empty', style: { marginBottom: '12px' } }, 'No hard rules. Add one, or pin a learned rule.'),
    list,
    h('div', { class: 'row' }, h('button', { onclick: () => list.append(ruleCard({ conditions: [{ feature: featNames[0] }] })) }, '+ Add rule'), save));
}

async function tabTree(m) {
  const members = m.structure.members;
  const q = state.memberSel != null ? `?member=${state.memberSel}` : '';
  const tree = await api('GET', `/api/models/${enc(m.name)}/tree${q}`);
  const walk = (node, depth, branch) => {
    const tag = branch == null ? null : h('span', { class: 'branch' }, branch);
    if (node.type === 'leaf') {
      return h('div', { class: 'leaf' }, tag, '→ ', h('strong', {}, node.prediction ?? '(empty)'),
        h('span', { class: 'muted small' }, `  ${pct(node.confidence)} · n=${num(Math.round(node.support))}${node.truncated ? ' · (deeper levels hidden)' : ''}`));
    }
    const cond = node.kind === 'numeric' ? `${node.feature} ≤ ${fmtVal(node.value)}` : `${node.feature} = ${fmtVal(node.value)}`;
    return h('details', { open: depth < 3 },
      h('summary', {}, tag, h('span', { class: 'cond' }, cond), h('span', { class: 'muted small' }, `  n=${num(Math.round(node.support))}`)),
      walk(node.children[0], depth + 1, 'yes'), walk(node.children[1], depth + 1, 'no'));
  };
  const sel = members > 1 ? h('label', { class: 'row small' }, 'Show tree ',
    h('select', { style: { width: 'auto' }, onchange: e => { state.memberSel = e.target.value === '' ? null : Number(e.target.value); renderTab(); } },
      h('option', { value: '' }, 'most accurate'),
      Array.from({ length: members }, (_, i) => h('option', { value: i, selected: state.memberSel === i }, `#${i + 1}`)))) : null;
  return h('div', {}, h('div', { class: 'row', style: { justifyContent: 'space-between', marginBottom: '8px' } }, sel,
    h('button', { class: 'sm', onclick: () => renderTab() }, 'Refresh')), h('div', { class: 'tree' }, walk(tree, 0, null)));
}

function tabPerf(m) {
  const pc = Object.entries(m.metrics.per_class);
  const cm = m.metrics.confusion;
  const labels = [...new Set([...Object.keys(cm), ...Object.values(cm).flatMap(r => Object.keys(r))])].sort();
  const maxCell = Math.max(1, ...Object.values(cm).flatMap(r => Object.values(r)));
  const events = [...m.events].reverse();
  return h('div', { class: 'grid-2' },
    h('div', { class: 'panel' }, h('h2', {}, 'Per-class quality'),
      pc.length ? h('div', { class: 'table-wrap' }, h('table', {},
        h('thead', {}, h('tr', {}, h('th', {}, 'Class'), h('th', { class: 'num' }, 'Precision'), h('th', { class: 'num' }, 'Recall'), h('th', { class: 'num' }, 'F1'), h('th', { class: 'num' }, 'Support'))),
        h('tbody', {}, pc.map(([c, s]) => h('tr', {}, h('td', {}, c), h('td', { class: 'num' }, pct(s.precision)),
          h('td', { class: 'num' }, pct(s.recall)), h('td', { class: 'num' }, pct(s.f1)), h('td', { class: 'num' }, num(s.support)))))))
        : h('div', { class: 'empty' }, 'No labels yet.'),
      h('h3', {}, 'Adaptation events'),
      events.length ? h('ul', { class: 'path' }, events.map(e => h('li', {},
        h('span', { class: `badge ${e.type === 'drift' ? 'warn' : ''}` }, e.type), ' ',
        e.type === 'drift' ? 'accuracy dropped — replaced the tree with one trained on recent data' : 'accuracy slipping — started training a background tree',
        h('span', { class: 'small muted' }, `  after ${num(e.at)} labels${m.kind === 'forest' ? ` · tree #${e.member + 1}` : ''}`))))
        : h('div', { class: 'small muted' }, 'No concept drift detected so far.')),
    h('div', { class: 'panel' }, h('h2', {}, 'Confusion matrix'),
      labels.length ? h('div', { class: 'table-wrap' }, h('table', { class: 'cm' },
        h('thead', {}, h('tr', {}, h('th', {}, 'actual ↓ / predicted →'), labels.map(l => h('th', {}, l)))),
        h('tbody', {}, Object.keys(cm).sort().map(a => h('tr', {}, h('th', {}, a), labels.map(p => {
          const v = (cm[a] || {})[p] || 0;
          return h('td', { title: `actual ${a}, predicted ${p}: ${v}`, style: { background: v ? `color-mix(in srgb, var(--series-1) ${Math.round(8 + 60 * v / maxCell)}%, transparent)` : '' } }, v || '');
        }))))))
        : h('div', { class: 'empty' }, 'No labels yet.')));
}

function tabApi(m) {
  const base = location.origin;
  const example = {};
  for (const f of m.schema.features) example[f.name] = f.type === 'numeric' ? 0 : (f.values[0] || 'value');
  const body = JSON.stringify({ features: example });
  return h('div', {},
    h('p', {}, 'Call Desic from any service. Every decision shows up live in this dashboard; send the correct answer later and the model learns from it.'),
    h('h3', {}, '1 · Ask for a decision'),
    h('pre', {}, `curl -s -X POST ${base}/api/models/${enc(m.name)}/decide \\\n  -H 'Content-Type: application/json' \\\n  -d '${body}'`),
    h('h3', {}, '2 · Send feedback (the real outcome)'),
    h('pre', {}, `curl -s -X POST ${base}/api/models/${enc(m.name)}/feedback \\\n  -H 'Content-Type: application/json' \\\n  -d '{"decision_id": "<id from step 1>", "label": "${m.classes[0] || 'right_answer'}"}'`),
    h('h3', {}, 'Python'),
    h('pre', {}, `import httpx\n\nBASE = "${base}/api/models/${m.name}"\nd = httpx.post(f"{BASE}/decide", json=${body}).json()\nprint(d["prediction"], d["confidence"], d["explanation"]["path"])\n\n# later, when you know the real outcome:\nhttpx.post(f"{BASE}/feedback", json={"decision_id": d["id"], "label": "${m.classes[0] || 'right_answer'}"})`),
    h('h3', {}, 'Bulk learning from labelled rows'),
    h('pre', {}, `curl -s -X POST ${base}/api/models/${enc(m.name)}/learn \\\n  -H 'Content-Type: application/json' \\\n  -d '{"rows": [${JSON.stringify({ ...example, [m.target]: m.classes[0] || 'label' })}]}'`),
    h('p', { class: 'small muted' }, h('a', { href: '/docs', target: '_blank' }, 'Full interactive API reference →')));
}

// ================================================================== datasets
async function viewDatasets() {
  state.view = { kind: 'datasets' };
  const list = await api('GET', '/api/datasets').catch(e => { toast(e.message, 'error'); return []; });
  if (!state.view || state.view.kind !== 'datasets') return;
  const fileInput = h('input', { type: 'file', accept: '.csv,.tsv,.txt,.json,.jsonl,.ndjson', style: { display: 'none' } });
  const nameInput = h('input', { placeholder: 'dataset name (optional)', style: { maxWidth: '280px' } });
  const drop = h('div', { class: 'drop', tabindex: 0, onclick: () => fileInput.click(),
    onkeydown: e => { if (e.key === 'Enter' || e.key === ' ') fileInput.click(); } },
    h('strong', {}, 'Drop a CSV / JSON file here'), h('div', { class: 'small' }, 'or click to choose · first row = column names · up to 50 MB'));
  const upload = file => guard(null, async () => {
    const fd = new FormData();
    fd.append('file', file);
    fd.append('name', nameInput.value.trim() || file.name.replace(/\.[^.]+$/, ''));
    drop.classList.add('over');
    try {
      const ds = await api('POST', '/api/datasets', fd, true);
      toast(`Uploaded ${ds.n_rows} rows`, 'good');
      location.hash = `#/datasets/${ds.id}`;
    } finally { drop.classList.remove('over'); }
  });
  fileInput.onchange = () => fileInput.files[0] && upload(fileInput.files[0]);
  drop.addEventListener('dragover', e => { e.preventDefault(); drop.classList.add('over'); });
  drop.addEventListener('dragleave', () => drop.classList.remove('over'));
  drop.addEventListener('drop', e => { e.preventDefault(); drop.classList.remove('over'); if (e.dataTransfer.files[0]) upload(e.dataTransfer.files[0]); });

  mount(
    h('div', { class: 'page-head' }, h('div', {}, h('h1', {}, 'Datasets'),
      h('div', { class: 'sub' }, 'Bring your own data, or ', h('a', { href: '#/generate' }, 'generate it with your AI key'), '. Then train a model on it in one click.'))),
    h('div', { class: 'panel', style: { marginBottom: '16px' } }, h('div', { class: 'row', style: { marginBottom: '10px' } }, nameInput), drop, fileInput),
    list.length ? h('div', { class: 'panel' }, h('div', { class: 'table-wrap' }, h('table', {},
      h('thead', {}, h('tr', {}, h('th', {}, 'Name'), h('th', {}, 'Source'), h('th', { class: 'num' }, 'Rows'), h('th', { class: 'num' }, 'Columns'), h('th', {}, 'Created'), h('th', {}))),
      h('tbody', {}, list.map(d => h('tr', { class: 'clickable', onclick: () => { location.hash = `#/datasets/${d.id}`; } },
        h('td', {}, h('strong', {}, d.name)),
        h('td', {}, h('span', { class: `badge ${d.source === 'generated' ? 'accent' : ''}` }, d.source)),
        h('td', { class: 'num' }, num(d.n_rows)), h('td', { class: 'num' }, d.columns.length),
        h('td', { class: 'small muted' }, ago(d.created_at)),
        h('td', {}, h('button', { class: 'sm danger', onclick: e => { e.stopPropagation(); guard(e.currentTarget, async () => {
          if (!confirm(`Delete dataset ${d.name}?`)) return;
          await api('DELETE', `/api/datasets/${d.id}`); viewDatasets();
        }); } }, 'Delete'))))))))
      : h('div', { class: 'empty' }, 'No datasets yet.'));
}

async function viewDataset(id) {
  state.view = { kind: 'dataset', id };
  let ds, models;
  try { [ds, models] = await Promise.all([api('GET', `/api/datasets/${enc(id)}`), api('GET', '/api/models')]); }
  catch (e) { mount(h('div', { class: 'empty' }, e.message)); return; }
  const cols = ds.columns;
  const design = ds.meta && ds.meta.design;
  const guessTarget = (design && design.target) || cols.find(c => /^(label|target|class|decision|outcome|y)$/i.test(c))
    || [...cols].reverse().find(c => ds.profile[c].type === 'categorical' && ds.profile[c].distinct <= 20) || cols[cols.length - 1];

  const modelName = h('input', { value: (ds.name || 'model').toLowerCase().replace(/[^a-z0-9_]+/g, '_').replace(/^_|_$/g, '') || 'model', list: 'dl-models' });
  const target = h('select', {}, cols.map(c => h('option', { value: c, selected: c === guessTarget }, c)));
  const featBox = h('div', { class: 'row' });
  const renderFeats = () => featBox.replaceChildren(...cols.filter(c => c !== target.value).map(c =>
    h('label', { class: 'check badge' }, h('input', { type: 'checkbox', value: c, checked: true }), c)));
  target.onchange = renderFeats; renderFeats();
  const kind = h('select', {}, h('option', { value: 'tree' }, 'Adaptive tree'), h('option', { value: 'forest' }, 'Adaptive random forest'));
  const holdout = h('input', { type: 'number', min: 0, max: 0.5, step: 0.05, value: 0.2 });
  const passes = h('input', { type: 'number', min: 1, max: 10, value: 1 });
  const jobSlot = h('div');
  const start = h('button', { class: 'primary' }, 'Train');
  const existing = () => models.find(m => m.name === modelName.value.trim());
  const hint = h('div', { class: 'note' });
  const updHint = () => {
    const ex = existing();
    hint.textContent = ex ? `“${ex.name}” exists — it will continue learning from these rows (target “${ex.target}”).`
      : 'A new model will be created with a schema inferred from this dataset.';
  };
  modelName.oninput = updHint; updHint();
  start.onclick = () => guard(start, async () => {
    const features = [...featBox.querySelectorAll('input:checked')].map(i => i.value);
    const panel = jobPanel((j, out) => {
      const r = j.result;
      out.replaceChildren(h('div', { class: 'done', style: { marginTop: '8px' } },
        h('p', {}, h('strong', {}, `Trained on ${num(r.trained_rows)} rows. `),
          r.holdout ? `Hold-out accuracy: ${pct(r.holdout.accuracy)} on ${num(r.holdout.rows)} unseen rows. ` : '',
          `Prequential accuracy: ${pct(r.prequential.accuracy)}.`),
        h('a', { href: `#/models/${enc(r.model)}` }, `Open ${r.model} →`)));
    });
    jobSlot.replaceChildren(panel.el);
    const job = await api('POST', `/api/datasets/${enc(id)}/train`, {
      model: modelName.value.trim(), target: target.value, features: existing() ? null : features,
      kind: kind.value, holdout: Number(holdout.value), passes: Number(passes.value),
    });
    panel.watch(job);
  });

  const preview = ds.preview;
  mount(
    h('div', { class: 'page-head' }, h('div', {},
      h('div', { class: 'row' }, h('h1', {}, ds.name), h('span', { class: `badge ${ds.source === 'generated' ? 'accent' : ''}` }, ds.source)),
      h('div', { class: 'sub' }, `${num(ds.n_rows)} rows · ${cols.length} columns`)),
      h('a', { href: '#/datasets' }, '← all datasets')),
    ds.meta && ds.meta.description ? h('div', { class: 'note' }, h('strong', {}, 'Generated for: '), ds.meta.description) : null,
    h('div', { class: 'grid-2' },
      h('div', { class: 'panel' }, h('h2', {}, 'Train a model'),
        h('datalist', { id: 'dl-models' }, models.map(m => h('option', { value: m.name }))),
        h('div', { class: 'form-grid' },
          h('label', { class: 'field' }, h('span', {}, 'Model name (new or existing)'), modelName),
          h('label', { class: 'field' }, h('span', {}, 'Decision column (target)'), target),
          h('label', { class: 'field' }, h('span', {}, 'Model type'), kind),
          h('label', { class: 'field' }, h('span', {}, 'Hold-out fraction'), holdout),
          h('label', { class: 'field' }, h('span', {}, 'Passes over data'), passes)),
        h('label', { class: 'field' }, h('span', {}, 'Features'), featBox),
        hint, start, jobSlot),
      h('div', { class: 'panel' }, h('h2', {}, 'Columns'),
        h('div', { class: 'table-wrap' }, h('table', {},
          h('thead', {}, h('tr', {}, h('th', {}, 'Column'), h('th', {}, 'Type'), h('th', { class: 'num' }, 'Distinct'), h('th', {}, 'Examples'))),
          h('tbody', {}, cols.map(c => h('tr', {}, h('td', {}, h('strong', {}, c)), h('td', {}, h('span', { class: 'badge' }, ds.profile[c].type)),
            h('td', { class: 'num' }, num(ds.profile[c].distinct)),
            h('td', { class: 'small muted' }, ds.profile[c].values.slice(0, 6).join(', '))))))))),
    h('div', { class: 'panel', style: { marginTop: '16px' } }, h('h2', {}, `Preview (first ${preview.length} rows)`),
      h('div', { class: 'table-wrap' }, h('table', {},
        h('thead', {}, h('tr', {}, cols.map(c => h('th', {}, c)))),
        h('tbody', {}, preview.map(r => h('tr', {}, cols.map(c => h('td', {}, fmtVal(r[c]))))))))));
}

// ================================================================== generate
const PRESETS = {
  anthropic: { label: 'Anthropic (Claude)', provider: 'anthropic', base_url: '', model: 'claude-opus-5-5' },
  openai: { label: 'OpenAI', provider: 'openai', base_url: 'https://api.openai.com/v1', model: '' },
  gemini: { label: 'Google Gemini (OpenAI-compatible)', provider: 'openai', base_url: 'https://generativelanguage.googleapis.com/v1beta/openai', model: '' },
  groq: { label: 'Groq', provider: 'openai', base_url: 'https://api.groq.com/openai/v1', model: '' },
  openrouter: { label: 'OpenRouter', provider: 'openai', base_url: 'https://openrouter.ai/api/v1', model: '' },
  ollama: { label: 'Ollama (local, no key)', provider: 'openai', base_url: 'http://localhost:11434/v1', model: '' },
  custom: { label: 'Other OpenAI-compatible', provider: 'openai', base_url: '', model: '' },
};
function storedProvider() {
  try { return JSON.parse(localStorage.getItem('desic.provider') || 'null'); } catch { return null; }
}

function viewGenerate() {
  state.view = { kind: 'generate' };
  const saved = storedProvider() || {};
  const preset = h('select', {}, Object.entries(PRESETS).map(([k, p]) => h('option', { value: k, selected: k === (saved.preset || 'anthropic') }, p.label)));
  const key = h('input', { type: 'password', autocomplete: 'off', placeholder: 'sk-…', value: saved.api_key || '' });
  const model = h('input', { value: saved.model || PRESETS.anthropic.model });
  const baseUrl = h('input', { value: saved.base_url || '', placeholder: 'https://…/v1' });
  const remember = h('input', { type: 'checkbox', checked: !!saved.api_key });
  const baseField = h('label', { class: 'field' }, h('span', {}, 'Base URL'), baseUrl);
  const syncPreset = (initial) => {
    const p = PRESETS[preset.value];
    baseField.style.display = p.provider === 'anthropic' ? 'none' : '';
    if (!initial) { baseUrl.value = p.base_url; model.value = p.model; }
    model.placeholder = p.provider === 'anthropic' ? 'claude-opus-5-5' : 'model name, e.g. from your provider docs';
  };
  preset.onchange = () => syncPreset(false); syncPreset(true);
  const providerBody = () => {
    const p = PRESETS[preset.value];
    const cfg = { provider: p.provider, api_key: key.value.trim(), model: model.value.trim(), base_url: baseUrl.value.trim() };
    try {
      if (remember.checked) localStorage.setItem('desic.provider', JSON.stringify({ ...cfg, preset: preset.value }));
      else localStorage.removeItem('desic.provider');
    } catch { /* storage unavailable */ }
    return cfg;
  };

  const desc = h('textarea', { placeholder: 'e.g. Decide whether to approve a small-business loan application at a Turkish bank. Decisions: approve, review, reject.' });
  const nFeat = h('input', { type: 'number', min: 2, max: 20, value: 6 });
  const designBtn = h('button', { class: 'primary' }, 'Design dataset');
  const designSlot = h('div');
  designBtn.onclick = () => guard(designBtn, async () => {
    if (desc.value.trim().length < 5) throw new Error('Describe the decision first.');
    designBtn.textContent = 'Designing…';
    try {
      const design = await api('POST', '/api/generate/design', { ...providerBody(), description: desc.value.trim(), n_features: Number(nFeat.value) });
      designSlot.replaceChildren(designEditor(design, desc.value.trim(), providerBody));
    } finally { designBtn.textContent = 'Design dataset'; }
  });

  mount(
    h('div', { class: 'page-head' }, h('div', {}, h('h1', {}, 'Generate data with AI'),
      h('div', { class: 'sub' }, 'Describe a decision in plain language. Your own LLM key designs the features and writes labelled examples; Desic learns from them.'))),
    h('div', { class: 'steps stack' },
      h('div', { class: 'panel' }, h('h2', { class: 'step-title' }, 'Your AI provider'),
        h('div', { class: 'form-grid' },
          h('label', { class: 'field' }, h('span', {}, 'Provider'), preset),
          h('label', { class: 'field' }, h('span', {}, 'API key'), key),
          h('label', { class: 'field' }, h('span', {}, 'Model'), model),
          baseField),
        h('label', { class: 'check small' }, remember, 'Remember provider and key in this browser'),
        h('div', { class: 'note' }, '🔒 The key is sent to your Desic server only for this request and forwarded to the provider. Desic never stores or logs it. Leave it empty to use the server’s own ANTHROPIC_API_KEY.')),
      h('div', { class: 'panel' }, h('h2', { class: 'step-title' }, 'Describe the decision'),
        h('label', { class: 'field' }, h('span', {}, 'What should the model decide?'), desc),
        h('div', { class: 'row' }, h('label', { class: 'row small' }, 'Features ', h('span', { style: { width: '80px' } }, nFeat)), designBtn)),
      designSlot));
}

function designEditor(design, description, providerBody) {
  const nameI = h('input', { value: design.name });
  const targetI = h('input', { value: design.target });
  const classesI = h('input', { value: design.classes.join(', ') });
  const logicI = h('textarea', { value: design.decision_logic, rows: 4 });
  const feats = h('tbody');
  const featRow = f => {
    const tr = h('tr', {},
      h('td', {}, h('input', { 'data-k': 'name', value: f.name })),
      h('td', {}, h('select', { 'data-k': 'type' }, ['numeric', 'categorical'].map(t => h('option', { value: t, selected: f.type === t }, t)))),
      h('td', {}, h('input', { 'data-k': 'values', value: (f.values || []).join(', '), placeholder: 'categories' })),
      h('td', {}, h('input', { 'data-k': 'min', type: 'number', step: 'any', value: f.min ?? '' })),
      h('td', {}, h('input', { 'data-k': 'max', type: 'number', step: 'any', value: f.max ?? '' })),
      h('td', {}, h('input', { 'data-k': 'description', value: f.description || '' })),
      h('td', {}, h('button', { class: 'ghost sm', onclick: () => tr.remove() }, '✕')));
    return tr;
  };
  feats.append(...design.features.map(featRow));
  const rows = h('input', { type: 'number', min: 10, max: 5000, step: 10, value: 300 });
  const dsName = h('input', { value: design.name });
  const trainChk = h('input', { type: 'checkbox', checked: true });
  const trainName = h('input', { value: design.name });
  const kind = h('select', {}, h('option', { value: 'tree' }, 'Adaptive tree'), h('option', { value: 'forest' }, 'Adaptive random forest'));
  const read = () => ({
    name: nameI.value.trim(), target: targetI.value.trim(), decision_logic: logicI.value,
    classes: classesI.value.split(',').map(s => s.trim()).filter(Boolean),
    features: [...feats.children].map(tr => {
      const g = k => $(`[data-k=${k}]`, tr).value;
      return { name: g('name').trim(), type: g('type'), description: g('description'),
        values: g('values').split(',').map(s => s.trim()).filter(Boolean), min: Number(g('min') || 0), max: Number(g('max') || 0) };
    }).filter(f => f.name),
  });
  const jobSlot = h('div');
  const go = h('button', { class: 'primary' }, 'Generate dataset');
  go.onclick = () => guard(go, async () => {
    const panel = jobPanel((j, out) => {
      const r = j.result;
      const links = [h('a', { href: `#/datasets/${r.dataset}` }, `Open dataset (${num(r.rows)} rows) →`)];
      out.replaceChildren(h('div', { class: 'row', style: { marginTop: '8px' } }, links));
      if (r.train_job) {
        const tp = jobPanel((tj, tout) => tout.replaceChildren(h('p', {},
          `Model trained. Hold-out accuracy ${pct(tj.result.holdout && tj.result.holdout.accuracy)}. `,
          h('a', { href: `#/models/${enc(tj.result.model)}` }, 'Open model →'))));
        out.append(h('h3', {}, 'Training'), tp.el);
        tp.watch({ id: r.train_job });
      }
    });
    jobSlot.replaceChildren(panel.el);
    const job = await api('POST', '/api/generate/dataset', {
      ...providerBody(), description, design: read(), n_rows: Number(rows.value), name: dsName.value.trim(),
      train_model: trainChk.checked ? trainName.value.trim() : '', kind: kind.value,
    });
    panel.watch(job);
  });
  return h('div', { class: 'stack' },
    h('div', { class: 'panel' }, h('h2', { class: 'step-title' }, 'Review the design'),
      h('p', { class: 'small muted' }, 'Everything here is editable — fix names, add categories, or sharpen the decision logic before generating.'),
      h('div', { class: 'form-grid' },
        h('label', { class: 'field' }, h('span', {}, 'Name'), nameI),
        h('label', { class: 'field' }, h('span', {}, 'Decision column'), targetI),
        h('label', { class: 'field' }, h('span', {}, 'Decisions (classes)'), classesI)),
      h('label', { class: 'field' }, h('span', {}, 'Decision logic the data should follow'), logicI),
      h('div', { class: 'table-wrap' }, h('table', {},
        h('thead', {}, h('tr', {}, ['Feature', 'Type', 'Categories', 'Min', 'Max', 'Description', ''].map(t => h('th', {}, t)))), feats)),
      h('button', { class: 'sm', style: { marginTop: '8px' }, onclick: () => feats.append(featRow({ name: '', type: 'numeric' })) }, '+ feature')),
    h('div', { class: 'panel' }, h('h2', { class: 'step-title' }, 'Generate & train'),
      h('div', { class: 'form-grid' },
        h('label', { class: 'field' }, h('span', {}, 'Rows'), rows),
        h('label', { class: 'field' }, h('span', {}, 'Dataset name'), dsName),
        h('label', { class: 'field' }, h('span', {}, 'Model name'), trainName),
        h('label', { class: 'field' }, h('span', {}, 'Model type'), kind)),
      h('label', { class: 'check small', style: { marginBottom: '10px' } }, trainChk, 'Train a model on the data as soon as it is ready'),
      h('div', { class: 'note' }, 'Rows are requested in batches of 40, three at a time. Cost depends on your provider and model.'),
      go, jobSlot));
}

// ------------------------------------------------------------------ boot
window.addEventListener('hashchange', route);
connect();
route();
