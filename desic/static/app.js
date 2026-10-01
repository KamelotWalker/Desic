'use strict';
/* Desic dashboard — dependency-free. All user/LLM-provided strings are
   inserted as text nodes (never innerHTML) so data cannot inject markup. */

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
const dec = (v, d = 3) => (v == null ? '—' : Number(v).toFixed(d));
const enc = encodeURIComponent;
function fmtVal(v) {
  if (v == null) return '∅';
  if (typeof v === 'number') return Number.isInteger(v) ? String(v) : String(+v.toFixed(3));
  if (Array.isArray(v)) return v.join(', ');
  if (typeof v === 'object') return JSON.stringify(v);
  return String(v);
}
function stateText(s) { return typeof s === 'string' ? s : JSON.stringify(s); }
function clip(s, n = 160) { s = stateText(s); return s.length > n ? s.slice(0, n - 1) + '…' : s; }
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
function featName(f) {
  const [kind, ...rest] = f.split(':');
  const r = rest.join(':');
  if (kind === 'w') return `“${r}”`;
  if (kind === 'p5') return `“${r}…”`;
  if (kind === 'b') return `“${r.replace('_', ' ')}”`;
  if (kind === 'k') return r.replace('=', ' = ');
  if (kind === 'n') return `${r} (value)`;
  return f;
}
function barRows(entries, fmt = pct, max) {
  const m = max ?? Math.max(...entries.map(e => e[1]), 1e-9);
  return entries.map(([name, v, note]) => h('div', { class: 'bar-row', title: `${name}: ${fmt(v)}` },
    h('span', { class: 'name' }, name),
    h('div', { class: 'track' }, h('div', { class: 'fill', style: { width: `${Math.max(0, v / m) * 100}%` } })),
    h('span', { class: 'val' }, note ?? fmt(v))));
}
function answerLabel(a) {
  if (a.type === 'choice') return a.choice;
  if (a.type === 'score') return a.level;
  return a.answer ? 'true' : 'false';
}
function parseOptions(text) {
  const out = {};
  for (const line of text.split('\n')) {
    const t = line.trim();
    if (!t) continue;
    const i = t.indexOf(':');
    const k = (i === -1 ? t : t.slice(0, i)).trim();
    if (k) out[k] = i === -1 ? '' : t.slice(i + 1).trim();
  }
  return out;
}
function optionsText(opts) { return Object.entries(opts || {}).map(([k, v]) => (v ? `${k}: ${v}` : k)).join('\n'); }
function parseState(text) {
  const t = text.trim();
  if (t.startsWith('{') || t.startsWith('[')) { try { return JSON.parse(t); } catch { /* plain text */ } }
  return text;
}

// ------------------------------------------------------------------ state
const state = { view: null, q: null, tab: 'feed', jobWatchers: new Map(), questions: [], lastResult: null };

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
const isView = kind => state.view && state.view.kind === kind;
const isQuestion = name => isView('question') && state.view.name === name;
const refreshQuestionSoon = debounce(() => refreshQuestion(), 500);
const refreshListSoon = debounce(() => { if (isView('questions')) viewQuestions(); }, 800);

function onEvent(ev) {
  switch (ev.type) {
    case 'job': onJob(ev.job); break;
    case 'decision':
      for (const [name, a] of Object.entries(ev.answers)) {
        if (isQuestion(name)) { feedPrepend({ decision_id: ev.id, task: name, created_at: ev.time, state: ev.state, result: a,
          answer: answerLabel(a), confidence: a.confidence, abstain: a.abstain, source: a.source, label: null }); refreshQuestionSoon(); }
      }
      refreshListSoon();
      break;
    case 'feedback':
      if (isQuestion(ev.task)) { markFeedItem(ev.decision_id, ev.label, ev.correct); refreshQuestionSoon(); }
      refreshListSoon();
      break;
    case 'learned': case 'task_updated': case 'task_reset':
      if (isQuestion(ev.task)) refreshQuestionSoon();
      refreshListSoon();
      break;
    case 'drift':
      if (isQuestion(ev.task) && ev.event.type === 'drift') toast(`Concept drift detected in ${ev.task} — the student is re-weighting its experts.`, 'warn');
      break;
    case 'task_created': case 'task_deleted':
      refreshListSoon();
      if (ev.type === 'task_deleted' && isQuestion(ev.task)) location.hash = '#/questions';
      break;
    case 'dataset_created':
      if (isView('data')) viewData();
      break;
    case 'teacher_updated':
      if (isView('teacher')) viewTeacher();
      break;
    case 'neural_updated':
      if (isView('neural')) refreshNeuralSoon();
      if (ev.outcome) toast(`Neural checkpoint ${ev.checkpoint} was ${ev.outcome} after its shadow test.`, ev.outcome === 'promoted' ? 'good' : 'warn', 7000);
      if (isView('question')) refreshQuestionSoon();
      break;
  }
}

function onJob(job) {
  const w = state.jobWatchers.get(job.id);
  if (w) w(job);
  const label = { train: 'Training', distill: 'Distillation', generate: 'Generation', rebuild: 'Rebuild', neural: 'Neural training' }[job.kind] || job.kind;
  if (job.status === 'done' && !w) toast(`${label} finished: ${job.message}`, 'good');
  if (job.status === 'error' && !w) toast(`${label} failed: ${job.error}`, 'error', 8000);
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
      api('GET', `/v1/jobs/${job.id}`).then(update).catch(() => {});
    },
  };
}

// ------------------------------------------------------------------ router
function route() {
  const parts = location.hash.replace(/^#\/?/, '').split('/').filter(Boolean).map(decodeURIComponent);
  const section = parts[0] || 'playground';
  document.querySelectorAll('[data-nav]').forEach(a => a.classList.toggle('active', a.dataset.nav === section));
  if (section === 'questions' && parts[1]) return viewQuestion(parts[1]);
  if (section === 'questions') return viewQuestions();
  if (section === 'data' && parts[1]) return viewDataset(parts[1]);
  if (section === 'data') return viewData();
  if (section === 'teacher') return viewTeacher();
  if (section === 'neural') return viewNeural();
  return viewPlayground();
}
function mount(...kids) {
  const main = $('#main');
  main.replaceChildren(...kids.flat().filter(k => k != null && k !== false));
  return main;
}
function pageHead(title, sub, ...actions) {
  return h('div', { class: 'page-head' }, h('div', {}, h('h1', {}, title), sub ? h('div', { class: 'sub' }, sub) : null),
    actions.length ? h('div', { class: 'row' }, actions) : null);
}
async function loadQuestions() {
  try { state.questions = await api('GET', '/v1/questions'); } catch { /* keep cache */ }
  return state.questions;
}

// ================================================================== playground
const EXAMPLES = {
  ticket: {
    state: 'Merhaba, kartımdan aynı sipariş için iki kez ödeme çekildi. Acil iade istiyorum, yoksa aboneliği iptal edeceğim.',
    questions: [
      { name: 'department', type: 'choice', instructions: 'Which team should handle this ticket?', options: 'billing: payments, invoices, refunds\ntechnical: bugs, errors, outages\nsales: pricing, plans, purchasing' },
      { name: 'urgent', type: 'noul', instructions: 'The customer needs a response today.', options: '' },
    ],
  },
  loan: {
    state: JSON.stringify({ applicant: { monthly_income: 52000, credit_score: 1640, employment: 'salaried', debt_ratio: 0.22, city: 'izmir' }, loan: { amount: 180000, purpose: 'car' } }, null, 2),
    questions: [{ name: 'loan_decision', type: 'choice', instructions: 'What should we do with this loan application?', options: 'approve\nreview: a credit officer should look at it\nreject' }],
  },
};

function questionRow(q, list) {
  const known = () => state.questions.find(x => x.name === nameI.value.trim());
  const nameI = h('input', { value: q.name || '', placeholder: 'question name', list: 'dl-questions' });
  const typeS = h('select', {}, ['choice', 'score', 'noul'].map(t => h('option', { value: t, selected: t === (q.type || 'choice') }, t)));
  const instr = h('input', { value: q.instructions || '', placeholder: 'the question, e.g. Which team should handle this?' });
  const opts = h('textarea', { rows: 3, value: q.options || '', placeholder: 'one answer per line — name: description' });
  const badge = h('span', { class: 'small muted' });
  const optsField = h('label', { class: 'field' }, h('span', {}, 'Answers'), opts);
  const sync = () => {
    optsField.style.display = typeS.value === 'noul' ? 'none' : '';
    const k = known();
    badge.textContent = k ? `known question · ${num(k.labels)} labels` : (nameI.value.trim() ? 'new question — it will be registered' : '');
  };
  nameI.addEventListener('change', async () => {
    const k = known();
    if (k) {
      const d = await api('GET', `/v1/questions/${enc(k.name)}`).catch(() => null);
      if (d) { typeS.value = d.type; instr.value = d.instructions; opts.value = d.type === 'noul' ? '' : optionsText(d.options); }
    }
    sync();
  });
  typeS.onchange = sync;
  const row = h('div', { class: 'rule-card' },
    h('div', { class: 'form-grid' },
      h('label', { class: 'field' }, h('span', {}, 'Name'), nameI),
      h('label', { class: 'field' }, h('span', {}, 'Type'), typeS)),
    h('label', { class: 'field' }, h('span', {}, typeS.value === 'noul' ? 'Proposition' : 'Instructions'), instr),
    optsField,
    h('div', { class: 'row', style: { justifyContent: 'space-between' } }, badge,
      h('button', { class: 'ghost sm', onclick: () => row.remove() }, 'Remove')));
  row.read = () => {
    const name = nameI.value.trim();
    if (!name) return null;
    const spec = { type: typeS.value, instructions: instr.value.trim() };
    if (typeS.value !== 'noul') spec.criteria = parseOptions(opts.value);
    return [name, spec];
  };
  sync();
  list.append(row);
  return row;
}

async function viewPlayground() {
  state.view = { kind: 'playground' };
  await loadQuestions();
  if (!isView('playground')) return;
  const stateI = h('textarea', { rows: 9, class: 'mono', placeholder: 'Paste a message, a ticket, or a JSON object…' });
  const qList = h('div');
  const explain = h('input', { type: 'checkbox', checked: true });
  const escalate = h('select', { style: { width: 'auto' } }, ['auto', 'never', 'always'].map(v => h('option', { value: v }, v)));
  const results = h('div', { id: 'pg-results' });
  const load = key => {
    const ex = EXAMPLES[key];
    stateI.value = ex.state;
    qList.replaceChildren();
    ex.questions.forEach(q => questionRow(q, qList));
    qList.querySelectorAll('input[list]').forEach(i => i.dispatchEvent(new Event('change')));
  };
  const go = h('button', { class: 'primary' }, 'Decide');
  const requestBody = () => {
    const questions = {};
    for (const row of qList.children) { const r = row.read(); if (r) questions[r[0]] = r[1]; }
    return { state: parseState(stateI.value), questions, explain: explain.checked, escalate: escalate.value };
  };
  go.onclick = () => guard(go, async () => {
    const body = requestBody();
    if (!Object.keys(body.questions).length) throw new Error('Add at least one question.');
    const r = await api('POST', '/v1/decide', body);
    state.lastResult = { body, r };
    renderResults(results, body, r);
    loadQuestions();
  });
  const curl = h('button', { onclick: () => {
    const body = requestBody();
    const txt = `curl -s -X POST ${location.origin}/v1/decide -H 'Content-Type: application/json' -d '${JSON.stringify(body).replace(/'/g, "'\\''")}'`;
    navigator.clipboard.writeText(txt).then(() => toast('curl command copied', 'good'), () => toast(txt));
  } }, 'Copy as curl');

  mount(
    pageHead('Playground', 'Ask typed questions about any state. Answers are calibrated probabilities over the answers you allow — nothing else can come out.'),
    h('datalist', { id: 'dl-questions' }, state.questions.map(q => h('option', { value: q.name }))),
    h('div', { class: 'grid-2' },
      h('div', { class: 'stack' },
        h('div', { class: 'panel' },
          h('div', { class: 'panel-head' }, h('h2', {}, 'State'),
            h('div', { class: 'row' }, h('span', { class: 'small muted' }, 'examples:'),
              h('button', { class: 'sm', onclick: () => load('ticket') }, 'Support ticket'),
              h('button', { class: 'sm', onclick: () => load('loan') }, 'Loan (JSON)'))),
          stateI, h('div', { class: 'small muted', style: { marginTop: '6px' } }, 'Text or JSON — JSON is detected automatically.')),
        h('div', { class: 'panel' },
          h('div', { class: 'panel-head' }, h('h2', {}, 'Questions'),
            h('button', { class: 'sm', onclick: () => questionRow({}, qList) }, '+ question')),
          qList,
          h('div', { class: 'row', style: { marginTop: '8px' } }, go, curl,
            h('label', { class: 'check small' }, explain, 'explain'),
            h('label', { class: 'row small' }, 'teacher:', escalate)))),
      results));
  load('ticket');
  if (state.lastResult) renderResults(results, state.lastResult.body, state.lastResult.r);
  else results.append(h('div', { class: 'empty' }, 'Answers appear here. Try the example and press Decide.'));
}

function renderResults(el, body, r) {
  el.replaceChildren(h('div', { class: 'stack' },
    Object.entries(r.answers).map(([name, a]) => answerCard(name, a, r.id, body.state)),
    h('div', { class: 'small muted' }, `decision ${r.id}`)));
}

function answerCard(name, a, decisionId, st) {
  const head = h('div', { class: 'row', style: { justifyContent: 'space-between' } },
    h('div', { class: 'row' }, h('h2', { style: { margin: 0 } }, name), h('span', { class: 'badge' }, a.type)),
    h('div', { class: 'row' },
      h('span', { class: `badge ${a.source === 'teacher' ? 'warn' : a.source === 'rule' ? 'bad' : 'accent'}` },
        a.source === 'student' ? 'student (System 1)' : a.source === 'teacher' ? 'teacher (System 2)' : `rule: ${(a.rule && a.rule.name) || ''}`),
      a.abstain ? h('span', { class: 'badge warn', title: 'confidence below the abstain threshold' }, 'abstains') : null));
  let main;
  const probs = Object.entries(a.probabilities || {});
  if (a.type === 'noul') {
    main = h('div', {}, h('div', { class: 'verdict' }, h('span', { class: 'label' }, a.answer ? 'true' : 'false'),
      h('span', { class: 'badge' }, `P(true) = ${dec(a.probability)}`)));
  } else if (a.type === 'score') {
    main = h('div', {}, h('div', { class: 'verdict' }, h('span', { class: 'label' }, a.level),
      h('span', { class: 'badge' }, `score ${dec(a.score, 2)} of ${probs.length - 1}`)), h('div', { class: 'probs' }, barRows(probs, pct, 1)));
  } else {
    main = h('div', {}, h('div', { class: 'verdict' }, h('span', { class: 'label' }, a.choice)), h('div', { class: 'probs' }, barRows(probs, pct, 1)));
  }
  return h('div', { class: 'panel' }, head, h('div', { class: 'result', style: { borderTop: 'none', paddingTop: '8px', marginTop: 0 } },
    main,
    h('div', { class: 'small muted' }, `confidence ${pct(a.confidence)}`,
      a.student ? ` · the student alone was ${pct(a.student.confidence)} sure` : '',
      a.teacher_error ? ` · teacher failed: ${a.teacher_error}` : ''),
    a.rationale ? h('p', { class: 'note' }, h('strong', {}, 'Teacher: '), a.rationale) : null,
    a.explanation ? explanationView(a.explanation) : null,
    decisionId ? feedbackBox(name, a, decisionId) : null));
}

function explanationView(e) {
  const experts = Object.entries(e.experts || {});
  const awake = experts.filter(([, x]) => x.awake);
  const parts = [h('h3', {}, 'Why')];
  if (!awake.length) parts.push(h('p', { class: 'small muted' }, 'The student has no experience with this question yet.'));
  else parts.push(h('div', {}, barRows(awake.map(([n, x]) => [`${n} → ${x.answer}`, x.weight, `${pct(x.weight)} weight`]), pct, 1)));
  if (e.linear && e.linear.for && e.linear.for.length) {
    parts.push(h('div', { class: 'small' }, h('span', { class: 'muted' }, 'evidence for: '),
      e.linear.for.map((f, i) => [i ? ', ' : '', h('code', {}, featName(f.feature))]),
      e.linear.against && e.linear.against.length ? [h('span', { class: 'muted' }, ' · against: '),
        e.linear.against.map((f, i) => [i ? ', ' : '', h('code', {}, featName(f.feature))])] : null));
  }
  if (e.tree && e.tree.path && e.tree.path.length) {
    parts.push(h('ul', { class: 'path' }, e.tree.path.map(s => h('li', {}, h('code', {}, condText(s)),
      h('span', { class: 'small muted' }, `  (was ${fmtVal(s.observed)})`)))));
  }
  if (e.memory) {
    parts.push(h('div', { class: 'small muted' }, e.memory.exact_match ? 'memory: this exact state was labelled before'
      : `memory: ${(e.memory.neighbours || []).map(n => `${n.answer} (sim ${dec(n.similarity, 2)})`).join(', ')}`));
  }
  if (e.patches && e.patches.length) {
    parts.push(h('div', { class: 'small muted' }, 'patches: ', e.patches.map((p, i) => [i ? ', ' : '',
      `${p.answer} (sim ${dec(p.similarity, 2)}, trust ${pct(p.trust)}${p.on_probation ? ', on probation' : ''})`])));
  }
  if (e.neural && e.neural.act != null) {
    parts.push(h('div', { class: 'small muted' }, `neural act/escalate head: P(its top answer is right) = ${pct(e.neural.act)}`));
  }
  parts.push(h('div', { class: 'small muted' }, `familiarity ${pct(e.familiarity)} of this state's evidence was seen in training`,
    e.familiarity < 0.5 ? ' — unfamiliar, so the answer is pulled toward “don’t know”' : '', ` · calibration temperature ${dec(e.temperature, 2)}`));
  return h('div', {}, parts);
}

function feedbackBox(name, a, decisionId) {
  const box = h('div', { class: 'feedback-box' });
  const q = state.questions.find(x => x.name === name);
  const options = a.type === 'noul' ? ['true', 'false'] : Object.keys(a.probabilities || {});
  const current = answerLabel(a);
  const send = (label, btn) => guard(btn, async () => {
    const r = await api('POST', '/v1/feedback', { decision_id: decisionId, answers: { [name]: label } });
    const res = r.answers[name];
    box.replaceChildren(h('div', { class: 'done' }, res.correct ? '✓ Confirmed — the student reinforced this answer.' : `✓ Learned: the right answer is “${res.label}”.`),
      h('div', { class: 'small muted' }, `accuracy (recent) ${pct(res.metrics.accuracy)} · ECE ${dec(res.metrics.ece)}`));
  });
  const other = a.type === 'choice' ? h('input', { placeholder: 'new answer…', style: { width: '140px' } }) : null;
  box.append(h('strong', {}, 'Correct answer?'),
    h('div', { class: 'row' },
      options.map(o => h('button', { class: `chip${o === current ? ' yes' : ''}`, onclick: e => send(o, e.currentTarget) }, o === current ? `✓ ${o}` : o)),
      other, other ? h('button', { class: 'sm', onclick: e => other.value.trim() && send(other.value.trim(), e.currentTarget) }, 'Teach') : null));
  if (!q) box.append(h('div', { class: 'small muted' }, ''));
  return box;
}

// ================================================================== questions
async function viewQuestions() {
  const first = !isView('questions');
  state.view = { kind: 'questions' };
  const qs = await loadQuestions();
  if (!isView('questions')) return;
  const cards = qs.length ? h('div', { class: 'grid-cards' }, qs.map(questionCard)) : h('div', { class: 'empty' },
    h('p', {}, 'No questions yet.'),
    h('p', {}, 'Ask one in the ', h('a', { href: '#/playground' }, 'Playground'), ' (it is registered automatically), ',
      h('a', { href: '#/data' }, 'train from a dataset'), ', or run ', h('code', {}, 'desic demo'), '.'));
  if (first || !$('#create-slot')) {
    mount(pageHead('Questions', 'Every question has its own self-learning student. Click one to see how well it is calibrated.',
      h('button', { class: 'primary', onclick: showCreateForm }, '+ New question')), h('div', { id: 'create-slot' }), cards);
  } else {
    const main = $('#main');
    main.replaceChild(cards, main.lastElementChild);
  }
}

function questionCard(q) {
  return h('div', { class: 'panel model-card', onclick: () => { location.hash = `#/questions/${enc(q.name)}`; } },
    h('div', { class: 'row', style: { justifyContent: 'space-between' } }, h('div', { class: 'title' }, q.name), h('span', { class: 'badge accent' }, q.type)),
    h('div', { class: 'muted small' }, q.instructions || '—'),
    h('div', { class: 'small', style: { marginTop: '6px' } }, q.options.join(' · ')),
    h('div', { class: 'meta' },
      h('div', {}, h('b', {}, pct(q.accuracy)), h('span', { class: 'small muted' }, 'accuracy')),
      h('div', {}, h('b', {}, dec(q.ece)), h('span', { class: 'small muted' }, 'ECE')),
      h('div', {}, h('b', {}, num(q.labels)), h('span', { class: 'small muted' }, 'labels')),
      h('div', {}, h('b', {}, pct(q.teacher_rate)), h('span', { class: 'small muted' }, 'teacher'))));
}

function showCreateForm() {
  const slot = $('#create-slot');
  if (!slot || slot.firstChild) return;
  const list = h('div');
  const row = questionRow({ type: 'choice' }, list);
  const create = h('button', { class: 'primary' }, 'Create');
  create.onclick = () => guard(create, async () => {
    const r = row.read();
    if (!r) throw new Error('Give the question a name.');
    await api('POST', '/v1/questions', { name: r[0], ...r[1] });
    location.hash = `#/questions/${enc(r[0])}`;
  });
  slot.append(h('div', { class: 'panel', style: { marginBottom: '20px' } },
    h('div', { class: 'panel-head' }, h('h2', {}, 'New question'), h('button', { class: 'ghost sm', onclick: () => slot.replaceChildren() }, 'Cancel')),
    list, create));
}

async function viewQuestion(name) {
  const changed = !isQuestion(name);
  state.view = { kind: 'question', name };
  if (changed) state.tab = 'feed';
  let q;
  try { q = await api('GET', `/v1/questions/${enc(name)}`); }
  catch (e) { mount(h('div', { class: 'empty' }, e.message, ' — ', h('a', { href: '#/questions' }, 'all questions'))); return; }
  state.q = q;
  loadQuestions();
  const tabs = h('div', { class: 'tabs', role: 'tablist' });
  const TABS = [['feed', 'Live feed'], ['review', 'Review queue'], ['settings', 'Settings'], ['rules', 'Hard rules'],
    ['log', 'Feedback log'], ['versions', 'Versions'], ['api', 'Integrate']];
  for (const [id, label] of TABS) {
    tabs.append(h('button', { role: 'tab', class: state.tab === id ? 'active' : '', 'data-tab': id, onclick: () => {
      state.tab = id;
      tabs.querySelectorAll('button').forEach(b => b.classList.toggle('active', b.dataset.tab === id));
      renderTab();
    } }, label));
  }
  const panel = (title, sub, id) => h('div', { class: 'panel' }, h('div', { class: 'panel-head' }, h('h2', {}, title),
    h('span', { class: 'small muted' }, sub)), h('div', { id }));
  mount(
    h('div', { class: 'page-head' },
      h('div', {},
        h('div', { class: 'row' }, h('h1', {}, q.name), h('span', { class: 'badge accent' }, q.type)),
        h('div', { class: 'sub' }, q.instructions || 'No instructions yet.'),
        h('div', { class: 'small', style: { marginTop: '4px' } }, Object.keys(q.options).join(' · '))),
      h('div', { class: 'row' },
        h('button', { onclick: e => guard(e.currentTarget, async () => {
          const s = await api('POST', `/v1/questions/${enc(q.name)}/snapshots`, { note: 'manual' });
          toast(`Snapshot v${s.version} saved`, 'good');
          if (state.tab === 'versions') renderTab();
        }) }, 'Snapshot'),
        h('button', { onclick: e => guard(e.currentTarget, async () => {
          if (!confirm(`Forget everything ${q.name} has learned? The feedback log is kept, so you can rebuild later.`)) return;
          await api('POST', `/v1/questions/${enc(q.name)}/reset`); refreshQuestion();
        }) }, 'Reset'),
        h('button', { class: 'danger', onclick: e => guard(e.currentTarget, async () => {
          if (!confirm(`Delete ${q.name}, its student, snapshots and feedback log?`)) return;
          await api('DELETE', `/v1/questions/${enc(q.name)}`); location.hash = '#/questions';
        }) }, 'Delete'))),
    h('div', { class: 'kpis', id: 'kpis' }),
    h('div', { class: 'grid-2 even' },
      panel('Learning curve', 'recent accuracy and teacher calls, per label', 'chart-curve'),
      panel('Reliability', 'does 80% confidence mean 80% correct?', 'chart-rel')),
    h('div', { class: 'grid-2 even', style: { marginTop: '16px' } },
      panel('Risk – coverage', 'accuracy if you only answer the most confident', 'chart-rc'),
      panel('Student experts', 'mixture weights (log-loss Hedge)', 'chart-experts')),
    tabs, h('div', { id: 'tab-body' }));
  renderLive(q);
  renderTab();
}

async function refreshQuestion() {
  if (!isView('question')) return;
  try { state.q = await api('GET', `/v1/questions/${enc(state.view.name)}`); } catch { return; }
  renderLive(state.q);
  if (state.tab === 'log' || state.tab === 'versions') renderTab();
}

function renderLive(q) {
  const kpis = $('#kpis');
  if (!kpis) return;
  const m = q.metrics;
  const kpi = (k, v, d, title) => h('div', { class: 'kpi', title: title || '' }, h('div', { class: 'k' }, k), h('div', { class: 'v' }, v), h('div', { class: 'd' }, d));
  const src = Object.entries(q.labels_by_source || {}).map(([k, v]) => `${k} ${num(v)}`).join(' · ') || 'none yet';
  kpis.replaceChildren(
    kpi('Accuracy', pct(m.accuracy), `last ${num(m.window)} human/dataset labels`),
    kpi('ECE', dec(m.ece), 'expected calibration error — lower is better', 'Average gap between confidence and accuracy'),
    kpi('Log loss', dec(m.nll, 2), `Brier ${dec(m.brier, 3)} · proper scoring rules`),
    kpi('Answered accuracy', pct(m.answered_accuracy), `abstains on ${pct(m.abstain_rate)} of decisions`),
    kpi('Teacher calls', pct(m.teacher_rate), `${num(m.teacher_calls)} of ${num(m.decisions)} decisions`),
    kpi('Labels', num(q.labels), src),
    kpi('Awaiting feedback', num(q.pending), `temperature ${dec(q.temperature, 2)}`));
  $('#chart-curve').replaceChildren(lineChart({
    points: q.history, x: 'n', xLabel: 'labels', empty: 'No labels yet — give feedback or train on a dataset.',
    series: [{ key: 'accuracy', label: 'Accuracy', color: 'var(--series-1)' }, { key: 'teacher_rate', label: 'Teacher calls', color: 'var(--series-2)' }],
  }));
  $('#chart-rel').replaceChildren(reliabilityChart(m.reliability, m.ece));
  const thr = q.settings.abstain_threshold;
  $('#chart-rc').replaceChildren(lineChart({
    points: (m.risk_coverage || []).map(p => ({ ...p, coverage: p.coverage })), x: 'coverage', xFmt: pct, xDomain: [0, 1],
    empty: 'Needs labelled decisions.', series: [{ key: 'accuracy', label: 'Accuracy', color: 'var(--series-1)' }],
    marker: (() => { const pts = (m.risk_coverage || []).filter(p => p.threshold >= thr); return pts.length ? { x: pts[pts.length - 1].coverage, label: `abstain < ${pct(thr)}` } : null; })(),
  }));
  const usage = q.expert_usage || {};
  const used = Object.entries(q.expert_weights || {}).filter(([n]) => usage[n] > 0);
  const usedTotal = used.reduce((a, [, v]) => a + v, 0) || 1;
  const idle = Object.keys(q.expert_weights || {}).filter(n => !(usage[n] > 0));
  $('#chart-experts').replaceChildren(used.length ? h('div', {},
    barRows(used.map(([n, v]) => [n, v / usedTotal]), pct, 1),
    idle.length ? h('div', { class: 'small muted' }, `not used: ${idle.join(', ')}${idle.includes('tree') ? ' (the states have no structured fields)' : ''}`) : null,
    h('p', { class: 'small muted', style: { marginTop: '8px' } },
      `${q.neural_attached ? 'neural = Laya-style encoder (see the Neural page) · ' : ''}prior = base rates · linear = text/JSON evidence (${num(q.vocabulary)} features) · tree = thresholds on fields (${num(q.tree.nodes)} nodes) · memory = ${num(q.memory_size)} remembered examples`),
    patchesNote(q.patches))
    : h('div', { class: 'empty' }, 'No experts yet.'));
}

function patchesNote(p) {
  if (!p) return null;
  const gate = Object.values(p.gate || {});
  const avg = gate.length ? gate.reduce((a, b) => a + b, 0) / gate.length : null;
  return h('p', { class: 'small', style: { marginTop: '6px' } }, h('strong', {}, 'Patch layer: '),
    `${num(p.entries)} patches — ${num(p.on_probation)} on probation (reversible at once), ${num(p.consolidated)} consolidated into the experts above. `,
    'New labels act at once, but only on similar inputs; the experts learn them after ', num(p.probation), ' more labels.',
    avg != null ? ` Learned trust in patches: ${pct(avg)} on average across ${num(gate.length)} contexts.` : '');
}

// ------------------------------------------------------------------ charts
function lineChart({ points, x, series, xLabel = '', xFmt = num, xDomain, empty, marker }) {
  if (!points || points.length < 2) return h('div', { class: 'empty' }, empty || 'Not enough data yet.');
  const slot = $('#main');
  const W = Math.max(300, Math.min(700, (slot && slot.clientWidth / 2 - 60) || 520)), H = 200;
  const P = { l: 40, r: 14, t: 8, b: 24 };
  const [x0, x1] = xDomain || [points[0][x], points[points.length - 1][x]];
  const X = v => P.l + ((v - x0) / Math.max(x1 - x0, 1e-9)) * (W - P.l - P.r);
  const Y = v => P.t + (1 - v) * (H - P.t - P.b);
  const svg = sv('svg', { viewBox: `0 0 ${W} ${H}`, height: H, role: 'img' });
  const grid = sv('g', { class: 'grid axis' });
  for (const t of [0, 0.25, 0.5, 0.75, 1]) {
    grid.append(sv('line', { x1: P.l, x2: W - P.r, y1: Y(t), y2: Y(t) }), sv('text', { x: P.l - 6, y: Y(t) + 4, 'text-anchor': 'end' }, `${t * 100}%`));
  }
  grid.append(sv('text', { x: P.l, y: H - 6 }, xFmt(x0)), sv('text', { x: W - P.r, y: H - 6, 'text-anchor': 'end' }, `${xFmt(x1)} ${xLabel}`));
  svg.append(grid);
  if (marker) {
    svg.append(sv('line', { class: 'crosshair', x1: X(marker.x), x2: X(marker.x), y1: P.t, y2: H - P.b }),
      sv('text', { class: 'dlabel', x: X(marker.x) + (X(marker.x) > W * 0.7 ? -4 : 4), y: P.t + 12,
        'text-anchor': X(marker.x) > W * 0.7 ? 'end' : 'start' }, marker.label));
  }
  for (const s of series) {
    const d = points.filter(p => p[s.key] != null).map((p, i) => `${i ? 'L' : 'M'}${X(p[x]).toFixed(1)},${Y(p[s.key]).toFixed(1)}`).join('');
    svg.append(sv('path', { class: 'line', d, stroke: s.color }));
  }
  const cross = sv('line', { class: 'crosshair', y1: P.t, y2: H - P.b, visibility: 'hidden' });
  const dots = series.map(s => sv('circle', { r: 4, fill: s.color, stroke: 'var(--surface)', 'stroke-width': 2, visibility: 'hidden' }));
  svg.append(cross, ...dots);
  const tip = h('div', { class: 'tip', style: { display: 'none' } });
  const hit = sv('rect', { x: P.l, y: P.t, width: W - P.l - P.r, height: H - P.t - P.b, fill: 'transparent' });
  hit.addEventListener('pointermove', ev => {
    const rect = svg.getBoundingClientRect();
    const px = ((ev.clientX - rect.left) / rect.width) * W;
    let best = points[0];
    for (const p of points) if (Math.abs(X(p[x]) - px) < Math.abs(X(best[x]) - px)) best = p;
    const cx = X(best[x]);
    cross.setAttribute('x1', cx); cross.setAttribute('x2', cx); cross.setAttribute('visibility', 'visible');
    series.forEach((s, i) => { if (best[s.key] == null) return; dots[i].setAttribute('cx', cx); dots[i].setAttribute('cy', Y(best[s.key])); dots[i].setAttribute('visibility', 'visible'); });
    tip.replaceChildren(h('div', {}, h('strong', {}, `${xFmt(best[x])} ${xLabel}`)), series.map(s => h('div', {}, `${s.label}: ${pct(best[s.key])}`)));
    tip.style.display = 'block';
    tip.style.left = `${(cx / W) * 100}%`;
    tip.style.top = `${svg.offsetTop + (Math.min(...series.map(s => Y(best[s.key] ?? 0))) / H) * rect.height - 8}px`;
  });
  hit.addEventListener('pointerleave', () => { tip.style.display = 'none'; cross.setAttribute('visibility', 'hidden'); dots.forEach(d => d.setAttribute('visibility', 'hidden')); });
  svg.append(hit);
  const legend = series.length > 1 ? h('div', { class: 'legend' }, series.map(s => h('span', {}, h('i', { style: { background: s.color } }), s.label))) : null;
  return h('div', { class: 'chart' }, legend, svg, tip);
}

function reliabilityChart(bins, ece) {
  if (!bins || !bins.length) return h('div', { class: 'empty' }, 'Needs labelled decisions.');
  const W = 420, H = 200, P = { l: 40, r: 10, t: 8, b: 28 };
  const X = v => P.l + v * (W - P.l - P.r);
  const Y = v => P.t + (1 - v) * (H - P.t - P.b);
  const svg = sv('svg', { viewBox: `0 0 ${W} ${H}`, height: H, role: 'img', 'aria-label': 'Reliability diagram' });
  const grid = sv('g', { class: 'grid axis' });
  for (const t of [0, 0.5, 1]) {
    grid.append(sv('line', { x1: P.l, x2: W - P.r, y1: Y(t), y2: Y(t) }), sv('text', { x: P.l - 6, y: Y(t) + 4, 'text-anchor': 'end' }, `${t * 100}%`));
    grid.append(sv('text', { x: X(t), y: H - 10, 'text-anchor': t === 0 ? 'start' : t === 1 ? 'end' : 'middle' }, `${t * 100}%`));
  }
  svg.append(grid, sv('line', { class: 'crosshair', x1: X(0), y1: Y(0), x2: X(1), y2: Y(1) }));
  const tip = h('div', { class: 'tip', style: { display: 'none' } });
  for (const b of bins) {
    const x = X(b.lo) + 1, w = X(b.hi) - X(b.lo) - 2;
    const bar = sv('rect', { x, y: Y(b.accuracy), width: Math.max(w, 1), height: Math.max(Y(0) - Y(b.accuracy), 0.5), rx: 3, fill: 'var(--series-1)', opacity: 0.85 });
    const hit = sv('rect', { x: X(b.lo), y: P.t, width: X(b.hi) - X(b.lo), height: H - P.t - P.b, fill: 'transparent' });
    hit.addEventListener('pointerenter', () => {
      tip.replaceChildren(h('div', {}, h('strong', {}, `confidence ${pct(b.lo)}–${pct(b.hi)}`)),
        h('div', {}, `avg confidence ${pct(b.confidence)}`), h('div', {}, `accuracy ${pct(b.accuracy)}`), h('div', {}, `${num(b.count)} answers`));
      tip.style.display = 'block';
      tip.style.left = `${((X(b.lo) + X(b.hi)) / 2 / W) * 100}%`;
      tip.style.top = `${svg.offsetTop + (Y(b.accuracy) / H) * svg.getBoundingClientRect().height - 8}px`;
    });
    hit.addEventListener('pointerleave', () => { tip.style.display = 'none'; });
    svg.append(bar, hit);
  }
  return h('div', { class: 'chart' },
    h('div', { class: 'legend' }, h('span', {}, h('i', { style: { background: 'var(--series-1)' } }), 'accuracy per confidence bin'),
      h('span', {}, h('i', { style: { background: 'var(--text-2)' } }), 'perfect calibration'), h('span', {}, `ECE ${dec(ece)}`)),
    svg, tip);
}
window.addEventListener('resize', debounce(() => { if (isView('question') && state.q) renderLive(state.q); }, 250));

// ------------------------------------------------------------------ tabs
function renderTab() {
  const body = $('#tab-body');
  if (!body || !state.q) return;
  const render = { feed: tabFeed, review: tabReview, settings: tabSettings, rules: tabRules, log: tabLog, versions: tabVersions, api: tabApi }[state.tab];
  body.replaceChildren(h('div', { class: 'muted' }, 'loading…'));
  Promise.resolve(render(state.q)).then(el => { if ($('#tab-body') === body) body.replaceChildren(el); })
    .catch(e => body.replaceChildren(h('div', { class: 'empty' }, e.message)));
}

function feedItem(d, fresh) {
  const q = state.q;
  const actions = h('div', { class: 'actions', 'data-actions': '' });
  if (d.label != null) {
    actions.append(h('span', { class: `badge ${d.label === d.answer ? 'good' : 'bad'}` }, d.label === d.answer ? '✓ correct' : `✗ was ${d.label}`));
  } else {
    const opts = q.type === 'noul' ? ['true', 'false'] : Object.keys(q.options);
    const send = (label, btn) => guard(btn, () => api('POST', '/v1/feedback', { decision_id: d.decision_id, answers: { [q.name]: label } }));
    for (const o of opts) {
      actions.append(h('button', { class: `chip${o === d.answer ? ' yes' : ''}`, title: o === d.answer ? 'Confirm' : `Correct to ${o}`,
        onclick: e => send(o, e.currentTarget) }, o === d.answer ? `✓ ${o}` : o));
    }
  }
  return h('div', { class: `feed-item${fresh ? ' new' : ''}`, 'data-id': d.decision_id },
    h('div', { style: { minWidth: 0 } },
      h('div', {}, h('span', { class: 'dec' }, d.answer ?? '—'), ' ',
        h('span', { class: 'small muted' }, `${pct(d.confidence)} · ${d.source}${d.abstain ? ' · student abstained' : ''} · ${ago(d.created_at)}`)),
      h('div', { class: 'feats', title: stateText(d.state) }, clip(d.state, 220))),
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
function markFeedItem(id, label) {
  document.querySelectorAll(`[data-id="${CSS.escape(id)}"] [data-actions]`).forEach(a => {
    const answer = $('.dec', a.parentElement).textContent;
    a.replaceChildren(h('span', { class: `badge ${label === answer ? 'good' : 'bad'}` }, label === answer ? '✓ correct' : `✗ was ${label}`));
  });
  if (state.tab === 'review') document.querySelectorAll(`#review-list [data-id="${CSS.escape(id)}"]`).forEach(el => el.remove());
}

async function tabFeed(q) {
  const items = await api('GET', `/v1/questions/${enc(q.name)}/decisions?limit=50`);
  return h('div', {},
    h('p', { class: 'small muted' }, 'Decisions appear here the moment they are made — from the Playground or the API. Click the right answer to teach the student.'),
    h('div', { id: 'feed-list' }, items.length ? items.map(d => feedItem(d)) : h('div', { class: 'empty' }, 'No decisions yet.')));
}

async function tabReview(q) {
  const items = await api('GET', `/v1/questions/${enc(q.name)}/decisions?limit=50&pending=true&uncertain_first=true`);
  return h('div', {},
    h('p', { class: 'small muted' }, 'Active learning: unlabelled decisions the student was least sure about come first — labelling these teaches it the most.'),
    h('div', { id: 'review-list' }, items.length ? items.map(d => feedItem(d)) : h('div', { class: 'empty' }, 'Nothing waiting for review. 🎉')));
}

function tabSettings(q) {
  const instr = h('input', { value: q.instructions });
  const descs = Object.entries(q.options).map(([k, v]) => [k, h('input', { value: v, placeholder: 'description (helps the teacher)' })]);
  const thr = h('input', { type: 'range', min: 0, max: 1, step: 0.01, value: q.settings.abstain_threshold });
  const thrOut = h('strong', {}, pct(q.settings.abstain_threshold));
  thr.oninput = () => { thrOut.textContent = pct(+thr.value); };
  const mode = h('select', {}, [['on_abstain', 'when the student abstains'], ['always', 'on every decision (costly)'], ['off', 'never']]
    .map(([v, l]) => h('option', { value: v, selected: v === q.settings.teacher_mode }, l)));
  const tw = h('input', { type: 'number', min: 0, max: 1, step: 0.05, value: q.settings.teacher_weight });
  const add = q.type === 'choice' ? h('input', { placeholder: 'new answer name' }) : null;
  const save = h('button', { class: 'primary' }, 'Save settings');
  save.onclick = () => guard(save, async () => {
    await api('PATCH', `/v1/questions/${enc(q.name)}`, {
      instructions: instr.value, descriptions: Object.fromEntries(descs.map(([k, i]) => [k, i.value])),
      add_options: add && add.value.trim() ? [add.value.trim()] : undefined,
      settings: { abstain_threshold: +thr.value, teacher_mode: mode.value, teacher_weight: +tw.value },
    });
    toast('Saved', 'good'); await refreshQuestion(); renderTab();
  });
  return h('div', { class: 'grid-2' },
    h('div', { class: 'panel' }, h('h2', {}, 'Question'),
      h('label', { class: 'field' }, h('span', {}, q.type === 'noul' ? 'Proposition' : 'Instructions'), instr),
      q.type !== 'noul' ? descs.map(([k, i]) => h('label', { class: 'field' }, h('span', {}, k), i)) : null,
      add ? h('label', { class: 'field' }, h('span', {}, 'Add an answer'), add) : null),
    h('div', { class: 'panel' }, h('h2', {}, 'Behaviour'),
      h('label', { class: 'field' }, h('span', {}, 'Abstain below confidence '), h('div', { class: 'row' }, thr, thrOut)),
      h('p', { class: 'small muted' }, 'Use the risk–coverage chart above to pick this: a higher threshold means fewer but more accurate automatic answers.'),
      h('label', { class: 'field' }, h('span', {}, 'Ask the teacher'), mode),
      h('label', { class: 'field' }, h('span', {}, 'Teacher label weight (human = 1)'), tw),
      save));
}

function tabRules(q) {
  const OPS = ['==', '!=', '>', '>=', '<', '<=', 'in', 'not_in', 'contains', 'is_missing'];
  const opts = Object.keys(q.options);
  const list = h('div');
  const condRow = c => {
    const row = h('div', { class: 'cond-row' },
      h('input', { 'data-k': 'feature', value: c.feature || '', placeholder: 'field path, e.g. applicant.credit_score or $text' }),
      h('select', { 'data-k': 'op' }, OPS.map(o => h('option', { value: o, selected: o === (c.op || '==') }, o))),
      h('input', { 'data-k': 'value', value: Array.isArray(c.value) ? c.value.join(', ') : (c.value ?? ''), placeholder: 'value' }),
      h('button', { class: 'ghost sm', onclick: () => row.remove() }, '✕'));
    return row;
  };
  const ruleCard = r => {
    const conds = h('div', {}, (r.conditions || []).map(condRow));
    const card = h('div', { class: 'rule-card', 'data-id': r.id || '' },
      h('div', { class: 'form-grid' },
        h('label', { class: 'field' }, h('span', {}, 'Name'), h('input', { 'data-k': 'name', value: r.name || '' })),
        h('label', { class: 'field' }, h('span', {}, 'Then answer'), h('select', { 'data-k': 'decision' }, opts.map(o => h('option', { value: o, selected: o === r.decision }, o)))),
        h('label', { class: 'field' }, h('span', {}, 'Priority'), h('input', { 'data-k': 'priority', type: 'number', value: r.priority ?? 0 })),
        h('label', { class: 'field check', style: { marginTop: '22px' } }, h('input', { type: 'checkbox', 'data-k': 'enabled', checked: r.enabled !== false }), 'enabled')),
      h('div', { class: 'small muted', style: { marginBottom: '6px' } }, 'All conditions must match:'), conds,
      h('div', { class: 'row', style: { justifyContent: 'space-between' } },
        h('button', { class: 'sm', onclick: () => conds.append(condRow({})) }, '+ condition'),
        h('span', { class: 'small muted' }, r.id ? `used ${num(r.hits || 0)}×` : 'new'),
        h('button', { class: 'sm danger', onclick: () => card.remove() }, 'Remove rule')));
    return card;
  };
  list.append(...q.rules.map(ruleCard));
  const read = () => [...list.children].map(card => ({
    id: card.dataset.id || undefined, name: $('[data-k=name]', card).value, decision: $('[data-k=decision]', card).value,
    priority: Number($('[data-k=priority]', card).value || 0), enabled: $('[data-k=enabled]', card).checked,
    conditions: [...card.querySelectorAll('.cond-row')].map(r => {
      const op = $('[data-k=op]', r).value, raw = $('[data-k=value]', r).value.trim();
      const value = (op === 'in' || op === 'not_in') ? raw.split(',').map(s => s.trim()).filter(Boolean)
        : (raw !== '' && !isNaN(Number(raw)) ? Number(raw) : raw);
      return { feature: $('[data-k=feature]', r).value.trim(), op, value };
    }),
  }));
  const save = h('button', { class: 'primary' }, 'Save rules');
  save.onclick = () => guard(save, async () => {
    await api('PUT', `/v1/questions/${enc(q.name)}/rules`, { rules: read() });
    await refreshQuestion(); renderTab(); toast('Rules saved', 'good');
  });
  return h('div', {},
    h('p', { class: 'small muted' }, 'Hard rules run before the student (highest priority wins) — for policy, compliance and known edge cases. Fields use dotted JSON paths; ', h('code', {}, '$text'), ' is the whole state as text.'),
    list.children.length ? null : h('div', { class: 'empty', style: { marginBottom: '12px' } }, 'No hard rules.'),
    list,
    h('div', { class: 'row' }, h('button', { onclick: () => list.append(ruleCard({ conditions: [{}] })) }, '+ Add rule'), save));
}

async function tabLog(q) {
  const events = await api('GET', `/v1/questions/${enc(q.name)}/feedback?limit=100`);
  const excludeTeacher = h('input', { type: 'checkbox' });
  const jobSlot = h('div');
  const rebuild = h('button', {}, 'Rebuild student from log');
  rebuild.onclick = () => guard(rebuild, async () => {
    if (!confirm('Replay the whole feedback log into a fresh student? A snapshot of the current one is taken first.')) return;
    const panel = jobPanel((j, out) => out.replaceChildren(h('p', { class: 'done' }, `Rebuilt from ${num(j.result.replayed)} events.`)));
    jobSlot.replaceChildren(panel.el);
    panel.watch(await api('POST', `/v1/questions/${enc(q.name)}/rebuild`, { exclude_sources: excludeTeacher.checked ? ['teacher'] : [] }));
  });
  const topLabel = l => (typeof l === 'string' ? l : Object.entries(l).sort((a, b) => b[1] - a[1]).map(([k, v]) => `${k} ${pct(v)}`).slice(0, 2).join(', '));
  const counts = Object.entries(q.log || {}).map(([k, v]) => `${k}: ${num(v)}`).join(' · ');
  const undoText = u => !u ? '' : u.restored ? 'learned again' :
    `undone in ${u.seconds < 1 ? `${Math.max(1, Math.round(u.seconds * 1000))} ms` : `${dec(u.seconds, 1)} s`}` +
    (u.replayed ? ` (${num(u.replayed)} later labels replayed from a checkpoint)` : ' (still on probation: dropped exactly)');
  const lastN = h('input', { type: 'number', min: 1, max: 1000, value: 10, style: { width: '72px' } });
  const lastSource = h('select', {}, h('option', { value: '' }, 'any source'), ['human', 'teacher', 'dataset', 'system'].map(v => h('option', { value: v }, v)));
  const undoLast = h('button', {}, 'Undo last');
  undoLast.onclick = () => guard(undoLast, async () => {
    const n = Math.max(1, Math.min(1000, parseInt(lastN.value, 10) || 1));
    if (!confirm(`Retract the ${n} most recent ${lastSource.value || ''} labels of ${q.name}?`)) return;
    const r = await api('POST', `/v1/questions/${enc(q.name)}/retract-recent`, { n, sources: lastSource.value ? [lastSource.value] : null });
    toast(`${num(r.events.length)} labels retracted — ${undoText(r.undo)}${r.hint ? '. ' + r.hint : ''}`, r.hint ? 'warn' : 'good');
    renderTab();
  });
  const p = q.patches || {};
  return h('div', {},
    h('p', { class: 'small muted' }, 'Every label the student learned from, append-only. Retracting a label undoes it at once: ',
      `the last ${num(p.probation || 0)} labels are still on probation and are simply dropped; older ones are replayed out from the nearest checkpoint. `,
      'Rebuild replays the whole log into a fresh student. ', counts),
    h('div', { class: 'row', style: { marginBottom: '10px' } }, undoLast, lastN, 'labels from', lastSource),
    h('div', { class: 'row', style: { marginBottom: '10px' } }, rebuild, h('label', { class: 'check small' }, excludeTeacher, 'skip teacher labels'), jobSlot),
    events.length ? h('div', { class: 'table-wrap' }, h('table', {},
      h('thead', {}, h('tr', {}, h('th', {}, '#'), h('th', {}, 'When'), h('th', {}, 'Source'), h('th', {}, 'Label'), h('th', {}, 'State'), h('th', {}))),
      h('tbody', {}, events.map(e => h('tr', { style: e.retracted ? { opacity: 0.5, textDecoration: 'line-through' } : null },
        h('td', { class: 'num' }, e.id), h('td', { class: 'small muted' }, ago(e.created_at)),
        h('td', {}, h('span', { class: `badge ${e.source === 'teacher' ? 'warn' : e.source === 'human' ? 'accent' : ''}` }, e.source)),
        h('td', {}, topLabel(e.label)), h('td', { class: 'small', title: stateText(e.state) }, clip(e.state, 90)),
        h('td', {}, h('button', { class: 'sm', onclick: ev => guard(ev.currentTarget, async () => {
          const r = await api('POST', `/v1/feedback/${e.id}/${e.retracted ? 'restore' : 'retract'}`);
          toast(r.applied ? `#${e.id} ${undoText(r.undo)}` : (r.hint || 'nothing to do'), r.applied ? 'good' : 'warn');
          renderTab();
        }) }, e.retracted ? 'Restore' : 'Retract')))))))
      : h('div', { class: 'empty' }, 'The log is empty.'));
}

async function tabVersions(q) {
  const snaps = await api('GET', `/v1/questions/${enc(q.name)}/snapshots`);
  return h('div', {},
    h('p', { class: 'small muted' }, 'Snapshots are taken automatically every 250 labels, before every rebuild, and whenever you click Snapshot. Roll back if a batch of feedback made things worse.'),
    snaps.length ? h('div', { class: 'table-wrap' }, h('table', {},
      h('thead', {}, h('tr', {}, h('th', {}, 'Version'), h('th', {}, 'When'), h('th', {}, 'Note'), h('th', { class: 'num' }, 'Labels'),
        h('th', { class: 'num' }, 'Accuracy'), h('th', { class: 'num' }, 'ECE'), h('th', {}))),
      h('tbody', {}, snaps.map(s => h('tr', {},
        h('td', {}, `v${s.version}`), h('td', { class: 'small muted' }, ago(s.created_at)), h('td', {}, s.note),
        h('td', { class: 'num' }, num(s.labels)), h('td', { class: 'num' }, pct(s.metrics.accuracy)), h('td', { class: 'num' }, dec(s.metrics.ece)),
        h('td', {}, h('button', { class: 'sm', onclick: e => guard(e.currentTarget, async () => {
          if (!confirm(`Roll ${q.name} back to v${s.version}?`)) return;
          await api('POST', `/v1/questions/${enc(q.name)}/rollback`, { version: s.version });
          toast(`Rolled back to v${s.version}`, 'good'); refreshQuestion(); renderTab();
        }) }, 'Roll back')))))))
      : h('div', { class: 'empty' }, 'No snapshots yet.'));
}

function tabApi(q) {
  const base = location.origin;
  const spec = { type: q.type, instructions: q.instructions };
  if (q.type !== 'noul') spec.criteria = q.options;
  const body = JSON.stringify({ state: 'I was charged twice for the same order', questions: { [q.name]: {} } });
  const example = q.type === 'noul' ? 'true' : `"${Object.keys(q.options)[0]}"`;
  return h('div', {},
    h('p', {}, 'The request shape follows Jev: a state plus typed questions. Send ', h('code', {}, '{}'), ' for a question Desic already knows, or the full spec to register / extend it.'),
    h('h3', {}, '1 · Decide'),
    h('pre', {}, `curl -s -X POST ${base}/v1/decide -H 'Content-Type: application/json' \\\n  -d '${body}'`),
    h('h3', {}, 'Full question spec'),
    h('pre', {}, JSON.stringify({ [q.name]: spec }, null, 2)),
    h('h3', {}, '2 · Feedback when you know the real outcome'),
    h('pre', {}, `curl -s -X POST ${base}/v1/feedback -H 'Content-Type: application/json' \\\n  -d '{"decision_id": "<id>", "answers": {"${q.name}": ${example}}}'`),
    h('h3', {}, 'Python'),
    h('pre', {}, `import httpx\n\nr = httpx.post("${base}/v1/decide", json=${body}).json()\na = r["answers"]["${q.name}"]\nif a["abstain"]:\n    ...  # route to a human or a bigger model\nhttpx.post("${base}/v1/feedback", json={"decision_id": r["id"], "answers": {"${q.name}": ${example === 'true' ? 'True' : example}}})`),
    h('p', { class: 'small muted' }, h('a', { href: '/docs', target: '_blank' }, 'Full interactive API reference →')));
}

// ================================================================== data
async function viewData() {
  state.view = { kind: 'data' };
  const [list, teacher] = await Promise.all([api('GET', '/v1/datasets').catch(() => []), api('GET', '/v1/teacher').catch(() => ({}))]);
  if (!isView('data')) return;
  const fileInput = h('input', { type: 'file', accept: '.csv,.tsv,.txt,.json,.jsonl,.ndjson', style: { display: 'none' } });
  const nameInput = h('input', { placeholder: 'dataset name (optional)', style: { maxWidth: '280px' } });
  const drop = h('div', { class: 'drop', tabindex: 0, onclick: () => fileInput.click(), onkeydown: e => { if (e.key === 'Enter' || e.key === ' ') fileInput.click(); } },
    h('strong', {}, 'Drop a CSV / JSON file here'), h('div', { class: 'small' }, 'labelled rows to train on, or unlabelled rows for the teacher to label · up to 50 MB'));
  const upload = file => guard(null, async () => {
    const fd = new FormData();
    fd.append('file', file);
    fd.append('name', nameInput.value.trim() || file.name.replace(/\.[^.]+$/, ''));
    drop.classList.add('over');
    try {
      const ds = await api('POST', '/v1/datasets', fd, true);
      toast(`Uploaded ${ds.n_rows} rows`, 'good');
      location.hash = `#/data/${ds.id}`;
    } finally { drop.classList.remove('over'); }
  });
  fileInput.onchange = () => fileInput.files[0] && upload(fileInput.files[0]);
  drop.addEventListener('dragover', e => { e.preventDefault(); drop.classList.add('over'); });
  drop.addEventListener('dragleave', () => drop.classList.remove('over'));
  drop.addEventListener('drop', e => { e.preventDefault(); drop.classList.remove('over'); if (e.dataTransfer.files[0]) upload(e.dataTransfer.files[0]); });

  mount(
    pageHead('Data', 'Bring your own data, or let your AI provider write it. Then train a question on it in one click.'),
    h('div', { class: 'grid-2' },
      h('div', { class: 'panel' }, h('h2', {}, 'Upload'), h('div', { class: 'row', style: { marginBottom: '10px' } }, nameInput), drop, fileInput),
      generatePanel(teacher)),
    h('div', { class: 'panel', style: { marginTop: '16px' } }, h('h2', {}, 'Datasets'),
      list.length ? h('div', { class: 'table-wrap' }, h('table', {},
        h('thead', {}, h('tr', {}, h('th', {}, 'Name'), h('th', {}, 'Source'), h('th', { class: 'num' }, 'Rows'), h('th', {}, 'Columns'), h('th', {}, 'Created'), h('th', {}))),
        h('tbody', {}, list.map(d => h('tr', { class: 'clickable', onclick: () => { location.hash = `#/data/${d.id}`; } },
          h('td', {}, h('strong', {}, d.name)),
          h('td', {}, h('span', { class: `badge ${d.source === 'generated' ? 'accent' : ''}` }, d.source)),
          h('td', { class: 'num' }, num(d.n_rows)), h('td', { class: 'small' }, d.columns.join(', ')),
          h('td', { class: 'small muted' }, ago(d.created_at)),
          h('td', {}, h('button', { class: 'sm danger', onclick: e => { e.stopPropagation(); guard(e.currentTarget, async () => {
            if (!confirm(`Delete dataset ${d.name}?`)) return;
            await api('DELETE', `/v1/datasets/${d.id}`); viewData();
          }); } }, 'Delete')))))))
        : h('div', { class: 'empty' }, 'No datasets yet.')));
}

function generatePanel(teacher) {
  if (!teacher.configured || teacher.provider === 'jev_compatible') {
    return h('div', { class: 'panel' }, h('h2', {}, 'Generate with AI'),
      h('div', { class: 'empty' }, 'Connect a generative provider (Anthropic or OpenAI-compatible) on the ', h('a', { href: '#/teacher' }, 'Teacher'), ' page first.'));
  }
  const desc = h('textarea', { rows: 3, placeholder: 'e.g. Route incoming support tickets of a Turkish e-commerce site to billing, logistics, technical or sales.' });
  const designBtn = h('button', { class: 'primary' }, 'Design question');
  const slot = h('div');
  designBtn.onclick = () => guard(designBtn, async () => {
    if (desc.value.trim().length < 5) throw new Error('Describe the decision first.');
    designBtn.textContent = 'Designing…';
    try { slot.replaceChildren(designEditor(await api('POST', '/v1/generate/design', { description: desc.value.trim() }), desc.value.trim())); }
    finally { designBtn.textContent = 'Design question'; }
  });
  return h('div', { class: 'panel' }, h('h2', {}, 'Generate with AI'),
    h('p', { class: 'small muted' }, `Uses your ${teacher.provider} key (${teacher.model || 'default model'}). You review the design before any examples are written.`),
    h('label', { class: 'field' }, h('span', {}, 'What should be decided?'), desc), designBtn, slot);
}

function designEditor(d, description) {
  const name = h('input', { value: d.name });
  const type = h('select', {}, ['choice', 'score', 'noul'].map(t => h('option', { value: t, selected: t === d.type }, t)));
  const instr = h('input', { value: d.instructions });
  const opts = h('textarea', { rows: 5, value: optionsText(d.options) });
  const fmt = h('select', {}, ['text', 'json'].map(t => h('option', { value: t, selected: t === d.state_format }, t)));
  const sdesc = h('input', { value: d.state_description });
  const guide = h('textarea', { rows: 3, value: d.decision_guidelines });
  const n = h('input', { type: 'number', min: 5, max: 5000, step: 5, value: 200 });
  const train = h('input', { type: 'checkbox', checked: true });
  const go = h('button', { class: 'primary' }, 'Generate examples');
  const jobSlot = h('div');
  go.onclick = () => guard(go, async () => {
    const design = { name: name.value.trim(), type: type.value, instructions: instr.value, options: type.value === 'noul' ? { true: '', false: '' } : parseOptions(opts.value),
      state_format: fmt.value, state_description: sdesc.value, decision_guidelines: guide.value };
    const panel = jobPanel((j, out) => {
      out.replaceChildren(h('p', {}, h('a', { href: `#/data/${j.result.dataset}` }, `Open dataset (${num(j.result.examples)} examples) →`)));
      if (j.result.train_job) {
        const tp = jobPanel((tj, tout) => tout.replaceChildren(h('p', {}, `Trained. Hold-out accuracy ${pct(tj.result.holdout && tj.result.holdout.accuracy)}. `,
          h('a', { href: `#/questions/${enc(tj.result.task)}` }, 'Open question →'))));
        out.append(h('h3', {}, 'Training'), tp.el);
        tp.watch({ id: j.result.train_job });
      }
    });
    jobSlot.replaceChildren(panel.el);
    panel.watch(await api('POST', '/v1/generate/examples', { description, design, n: +n.value, train: train.checked }));
  });
  return h('div', { style: { marginTop: '12px' } },
    h('div', { class: 'form-grid' }, h('label', { class: 'field' }, h('span', {}, 'Question name'), name), h('label', { class: 'field' }, h('span', {}, 'Type'), type),
      h('label', { class: 'field' }, h('span', {}, 'State format'), fmt), h('label', { class: 'field' }, h('span', {}, 'Examples'), n)),
    h('label', { class: 'field' }, h('span', {}, 'Instructions'), instr),
    h('label', { class: 'field' }, h('span', {}, 'Answers (name: description)'), opts),
    h('label', { class: 'field' }, h('span', {}, 'What a state looks like'), sdesc),
    h('label', { class: 'field' }, h('span', {}, 'How an expert decides'), guide),
    h('label', { class: 'check small', style: { marginBottom: '10px' } }, train, 'train the question as soon as the data is ready'),
    go, jobSlot);
}

async function viewDataset(id) {
  state.view = { kind: 'dataset', id };
  let ds;
  try { [ds] = await Promise.all([api('GET', `/v1/datasets/${enc(id)}`), loadQuestions()]); }
  catch (e) { mount(h('div', { class: 'empty' }, e.message)); return; }
  const cols = ds.columns;
  const design = ds.meta && ds.meta.design;
  const guessAnswer = (design && 'answer') || cols.find(c => /^(label|target|class|decision|answer|outcome|team|y)$/i.test(c))
    || [...cols].reverse().find(c => ds.profile[c].type === 'category') || cols[cols.length - 1];
  const qName = h('input', { value: (design && design.name) || (ds.name || 'question').toLowerCase().replace(/[^a-z0-9_]+/g, '_').replace(/^_|_$/g, '') || 'question', list: 'dl-q' });
  const answerCol = h('select', {}, cols.map(c => h('option', { value: c, selected: c === guessAnswer }, c)));
  const stateBox = h('div', { class: 'row' });
  const renderCols = (box, exclude) => box.replaceChildren(...cols.filter(c => c !== exclude).map(c =>
    h('label', { class: 'check badge' }, h('input', { type: 'checkbox', value: c, checked: true }), c)));
  answerCol.onchange = () => renderCols(stateBox, answerCol.value);
  renderCols(stateBox, answerCol.value);
  const mode = h('select', {}, [['auto', 'auto'], ['text', 'text (join columns)'], ['json', 'JSON object']].map(([v, l]) => h('option', { value: v }, l)));
  const type = h('select', {}, ['choice', 'noul', 'score'].map(t => h('option', { value: t, selected: design && t === design.type }, t)));
  const levels = h('input', { placeholder: 'score levels, lowest first: low, medium, high' });
  const instr = h('input', { value: (design && design.instructions) || '', placeholder: 'the question' });
  const holdout = h('input', { type: 'number', min: 0, max: 0.5, step: 0.05, value: 0.2 });
  const hint = h('div', { class: 'note' });
  const upd = () => {
    const ex = state.questions.find(q => q.name === qName.value.trim());
    hint.textContent = ex ? `“${ex.name}” exists (${ex.type}: ${ex.options.join(', ')}) — it will keep learning from these rows.`
      : 'A new question will be created; its answers are the distinct values of the answer column.';
  };
  qName.oninput = upd; upd();
  const trainSlot = h('div');
  const trainBtn = h('button', { class: 'primary' }, 'Train');
  trainBtn.onclick = () => guard(trainBtn, async () => {
    const panel = jobPanel((j, out) => {
      const r = j.result;
      out.replaceChildren(h('p', { class: 'done' }, `Trained on ${num(r.trained)} examples. `,
        r.holdout ? `Hold-out: accuracy ${pct(r.holdout.accuracy)}, ECE ${dec(r.holdout.ece)}, coverage ${pct(r.holdout.coverage)}.` : ''),
        h('a', { href: `#/questions/${enc(r.task)}` }, `Open ${r.task} →`));
    });
    trainSlot.replaceChildren(panel.el);
    panel.watch(await api('POST', `/v1/datasets/${enc(id)}/train`, {
      task: qName.value.trim(), answer_column: answerCol.value, state_columns: [...stateBox.querySelectorAll('input:checked')].map(i => i.value),
      state_mode: mode.value, type: type.value, levels: levels.value.trim() ? levels.value.split(',').map(s => s.trim()).filter(Boolean) : null,
      instructions: instr.value, holdout: +holdout.value,
    }));
  });

  const dQ = h('select', {}, state.questions.map(q => h('option', { value: q.name }, q.name)));
  const dBox = h('div', { class: 'row' });
  renderCols(dBox, null);
  const dLimit = h('input', { type: 'number', min: 1, max: 5000, value: Math.min(200, ds.n_rows) });
  const dSlot = h('div');
  const dBtn = h('button', {}, 'Let the teacher label');
  dBtn.onclick = () => guard(dBtn, async () => {
    if (!dQ.value) throw new Error('Create the question first.');
    const panel = jobPanel((j, out) => out.replaceChildren(h('p', { class: 'done' }, `Teacher labelled ${num(j.result.labelled)} rows${j.result.failed ? ` (${j.result.failed} failed)` : ''}. `,
      h('a', { href: `#/questions/${enc(j.result.task)}` }, 'Open question →'))));
    dSlot.replaceChildren(panel.el);
    panel.watch(await api('POST', `/v1/datasets/${enc(id)}/distill`, {
      task: dQ.value, state_columns: [...dBox.querySelectorAll('input:checked')].map(i => i.value), state_mode: mode.value, limit: +dLimit.value,
    }));
  });

  mount(
    h('div', { class: 'page-head' }, h('div', {},
      h('div', { class: 'row' }, h('h1', {}, ds.name), h('span', { class: `badge ${ds.source === 'generated' ? 'accent' : ''}` }, ds.source)),
      h('div', { class: 'sub' }, `${num(ds.n_rows)} rows · ${cols.length} columns`)), h('a', { href: '#/data' }, '← all data')),
    ds.meta && ds.meta.description ? h('div', { class: 'note' }, h('strong', {}, 'Generated for: '), ds.meta.description) : null,
    h('datalist', { id: 'dl-q' }, state.questions.map(q => h('option', { value: q.name }))),
    h('div', { class: 'grid-2' },
      h('div', { class: 'panel' }, h('h2', {}, 'Train on labelled rows'),
        h('div', { class: 'form-grid' },
          h('label', { class: 'field' }, h('span', {}, 'Question (new or existing)'), qName),
          h('label', { class: 'field' }, h('span', {}, 'Answer column'), answerCol),
          h('label', { class: 'field' }, h('span', {}, 'Question type'), type),
          h('label', { class: 'field' }, h('span', {}, 'State from columns as'), mode),
          h('label', { class: 'field' }, h('span', {}, 'Hold-out fraction'), holdout)),
        h('label', { class: 'field' }, h('span', {}, 'Instructions'), instr),
        type.value === 'score' ? levels : null,
        h('label', { class: 'field' }, h('span', {}, 'State columns'), stateBox),
        hint, trainBtn, trainSlot),
      h('div', { class: 'panel' }, h('h2', {}, 'Distill: teacher labels unlabelled rows'),
        h('p', { class: 'small muted' }, 'The teacher (System 2) answers each row; the student (System 1) learns from its probabilities. Costs one teacher call per row.'),
        state.questions.length ? [
          h('label', { class: 'field' }, h('span', {}, 'Question'), dQ),
          h('label', { class: 'field' }, h('span', {}, 'State columns'), dBox),
          h('label', { class: 'field' }, h('span', {}, 'Rows to label'), dLimit), dBtn, dSlot]
          : h('div', { class: 'empty' }, 'Create a question first (Questions → New question).'))),
    h('div', { class: 'panel', style: { marginTop: '16px' } }, h('h2', {}, 'Columns'),
      h('div', { class: 'table-wrap' }, h('table', {},
        h('thead', {}, h('tr', {}, h('th', {}, 'Column'), h('th', {}, 'Type'), h('th', { class: 'num' }, 'Distinct'), h('th', {}, 'Examples'))),
        h('tbody', {}, cols.map(c => h('tr', {}, h('td', {}, h('strong', {}, c)), h('td', {}, h('span', { class: 'badge' }, ds.profile[c].type)),
          h('td', { class: 'num' }, num(ds.profile[c].distinct)), h('td', { class: 'small muted' }, ds.profile[c].values.slice(0, 6).join(', '))))))),
      h('h3', {}, `Preview (first ${ds.preview.length} rows)`),
      h('div', { class: 'table-wrap' }, h('table', {},
        h('thead', {}, h('tr', {}, cols.map(c => h('th', {}, c)))),
        h('tbody', {}, ds.preview.map(r => h('tr', {}, cols.map(c => h('td', { class: 'small' }, clip(fmtVal(r[c]), 140))))))))));
  type.onchange = () => { levels.style.display = type.value === 'score' ? '' : 'none'; if (!levels.isConnected) instr.parentElement.after(levels); };
}

// ================================================================== neural
const refreshNeuralSoon = debounce(() => { if (isView('neural')) viewNeural(); }, 400);
const STATUS_BADGE = { active: 'good', shadow: 'warn', candidate: 'accent', rejected: 'bad', failed: 'bad' };

async function viewNeural() {
  state.view = { kind: 'neural' };
  let st;
  try { st = await api('GET', '/v1/neural'); } catch (e) { mount(h('div', { class: 'empty' }, e.message)); return; }
  if (!isView('neural')) return;
  const head = pageHead('Neural student', 'A Laya-style encoder with a decision head, trained from the feedback log. A candidate must pass an offline test and a live shadow test before it joins the students as the “neural” expert.');
  if (!st.available) {
    mount(head, h('div', { class: 'panel' }, h('h2', {}, 'PyTorch is not installed'), h('p', {}, st.reason),
      h('pre', {}, "pip install -e '.[neural]'"), h('p', { class: 'small muted' }, 'Everything else in Desic keeps working without it.')));
    return;
  }
  const cks = st.checkpoints.filter(c => c.status !== 'deleted');
  const active = cks.find(c => c.status === 'active');
  const shadow = cks.find(c => c.status === 'shadow');
  const kpi = (k, v, d) => h('div', { class: 'kpi' }, h('div', { class: 'k' }, k), h('div', { class: 'v' }, v), h('div', { class: 'd' }, d));
  const test = c => (c && c.report.test && c.report.test.overall) || {};
  const sh = c => (c && c.shadow && c.shadow.n) ? c.shadow : null;
  const kpis = h('div', { class: 'kpis' },
    kpi('Active checkpoint', active ? active.id.slice(-11) : 'none', active ? `test accuracy ${pct(test(active).accuracy)} · log loss ${dec(test(active).nll, 3)}` : 'students run on the online experts only'),
    kpi('In shadow', shadow ? shadow.id.slice(-11) : 'none', shadow ? (sh(shadow) ? `${sh(shadow).n}/${st.config.shadow_min} labels · log loss ${dec(sh(shadow).nll / sh(shadow).n, 3)} vs live ${dec(sh(shadow).live_nll / sh(shadow).n, 3)}` : `waiting for ${st.config.shadow_min} labelled decisions`) : '—'),
    kpi('Training', st.training ? 'running' : 'idle', st.config.auto_train_every ? `automatic every ${num(st.config.auto_train_every)} new labels` : 'manual'),
    kpi('Backbone', st.config.backbone === 'scratch' ? 'built-in' : st.config.backbone.split('/').pop(), `objective ${st.config.objective.toUpperCase()} · ${st.config.epochs} epochs`));

  const jobSlot = h('div');
  const trainBtn = h('button', { class: 'primary', disabled: !!st.training }, 'Train a candidate now');
  trainBtn.onclick = () => guard(trainBtn, async () => {
    const panel = jobPanel((j, out) => out.replaceChildren(h('p', { class: 'done' }, `Checkpoint ${j.result.checkpoint}: ${j.result.status}. ${j.result.note || ''}`)));
    jobSlot.replaceChildren(panel.el);
    panel.watch(await api('POST', '/v1/neural/train'));
  });
  if (st.training) {
    const panel = jobPanel(() => {});
    jobSlot.replaceChildren(panel.el);
    panel.watch({ id: st.training.job });
  }
  const cancel = st.training ? h('button', { onclick: e => guard(e.currentTarget, () => api('POST', '/v1/neural/cancel')) }, 'Cancel') : null;

  mount(head, kpis,
    h('div', { class: 'grid-2' },
      neuralSettings(st),
      h('div', { class: 'panel' }, h('h2', {}, 'Pipeline'),
        h('ol', { class: 'small', style: { paddingLeft: '18px' } },
          h('li', {}, h('strong', {}, 'Dataset'), ' — every non-retracted label in the feedback log, one per (question, state); human > dataset > teacher. Teacher probabilities are soft targets (distillation).'),
          h('li', {}, h('strong', {}, 'Train'), ' — encoder + 2-layer decision transformer; each answer is scored at its own [MASK] marker, so questions can bring new answers. Loss: log score (+ RPS for score questions) + act/escalate head; optional RLCD stage.'),
          h('li', {}, h('strong', {}, 'Calibrate'), ' — one temperature per question on a validation slice.'),
          h('li', {}, h('strong', {}, 'Offline gate'), ' — on a hash-stable test slice no checkpoint ever trains on: must beat the base rates and not lose to the active checkpoint.'),
          h('li', {}, h('strong', {}, 'Shadow'), ` — predicts on the next ${st.config.shadow_min} labelled decisions without being served; promoted if its log loss is within ${st.config.promote_margin} of what users are served.`),
          h('li', {}, h('strong', {}, 'Active'), ' — joins every question’s mixture as the “neural” expert; Hedge decides how much to trust it per question, and it can be retired at any time.')),
        h('div', { class: 'row' }, trainBtn, cancel), jobSlot)),
    h('div', { class: 'panel', style: { marginTop: '16px' } }, h('h2', {}, 'Checkpoints'),
      cks.length ? checkpointTable(cks) : h('div', { class: 'empty' }, 'No checkpoints yet. Train one once your questions have some labels.')));
}

function neuralSettings(st) {
  const c = st.config;
  const known = Object.keys(st.backbones);
  const bb = h('select', {}, [...known.map(k => h('option', { value: k, selected: k === c.backbone }, st.backbones[k])),
    h('option', { value: '__custom', selected: !known.includes(c.backbone) }, 'Other Hugging Face encoder / local path…')]);
  const custom = h('input', { value: known.includes(c.backbone) ? '' : c.backbone, placeholder: 'e.g. dbmdz/bert-base-turkish-cased or /models/my-encoder' });
  const customField = h('label', { class: 'field' }, h('span', {}, 'Encoder id or path'), custom);
  const sync = () => { customField.style.display = bb.value === '__custom' ? '' : 'none'; };
  bb.onchange = sync; sync();
  const f = (label, key, attrs = {}) => {
    const el = h('input', { value: c[key], ...attrs });
    el.dataset.key = key;
    return h('label', { class: 'field' }, h('span', {}, label), el);
  };
  const objective = h('select', {}, [['ce', 'Cross-entropy (log score) — stable'], ['rlcd', 'RLCD-style: noisy logits + REINFORCE']].map(([v, l]) => h('option', { value: v, selected: v === c.objective }, l)));
  const autoPromote = h('input', { type: 'checkbox', checked: c.auto_promote });
  const initActive = h('input', { type: 'checkbox', checked: c.init_from_active });
  const fields = h('div', { class: 'form-grid' },
    f('Max tokens', 'max_len', { type: 'number', min: 32, max: 8192 }), f('Epochs', 'epochs', { type: 'number', min: 1, max: 100 }),
    f('Batch size', 'batch_size', { type: 'number', min: 1, max: 512 }), f('Encoder learning rate', 'lr_backbone', { type: 'number', step: 'any' }),
    f('Head learning rate', 'lr_head', { type: 'number', step: 'any' }), f('Teacher label weight', 'teacher_weight', { type: 'number', step: 0.05, min: 0, max: 1 }),
    f('Auto-train every N labels (0 = off)', 'auto_train_every', { type: 'number', min: 0 }), f('Shadow labels before promotion', 'shadow_min', { type: 'number', min: 0 }),
    f('Promotion margin (log loss)', 'promote_margin', { type: 'number', step: 0.01, min: 0 }), f('Device', 'device', { placeholder: 'auto, cpu, cuda, mps' }));
  const save = h('button', {}, 'Save settings');
  save.onclick = () => guard(save, async () => {
    const patch = { backbone: bb.value === '__custom' ? custom.value.trim() : bb.value, objective: objective.value,
      auto_promote: autoPromote.checked, init_from_active: initActive.checked };
    fields.querySelectorAll('input[data-key]').forEach(i => { patch[i.dataset.key] = i.type === 'number' ? Number(i.value) : i.value; });
    await api('PATCH', '/v1/neural/config', patch);
    toast('Neural settings saved', 'good'); viewNeural();
  });
  return h('div', { class: 'panel' }, h('h2', {}, 'Training settings'),
    h('label', { class: 'field' }, h('span', {}, 'Backbone'), bb), customField,
    h('p', { class: 'small muted' }, 'Pretrained encoders (ModernBERT, mmBERT) are downloaded from Hugging Face on first use and want a GPU; the built-in encoder trains from zero on CPU in seconds but only knows what your labels teach it. For Turkish, mmBERT is the Laya-multilingual choice.'),
    h('label', { class: 'field' }, h('span', {}, 'Objective'), objective), fields,
    h('label', { class: 'check small' }, autoPromote, 'promote automatically after a successful shadow test'),
    h('label', { class: 'check small', style: { marginLeft: '12px' } }, initActive, 'continue from the active checkpoint when the backbone matches'),
    h('div', { style: { marginTop: '10px' } }, save));
}

function checkpointTable(cks) {
  const t = c => c.report.test && c.report.test.overall || {};
  const body = h('tbody');
  for (const c of cks) {
    const r = c.report, sh = c.shadow && c.shadow.n ? c.shadow : null;
    const act = (label, action, cls = 'sm') => h('button', { class: cls, onclick: e => { e.stopPropagation(); guard(e.currentTarget, async () => {
      if (action === 'delete' && !confirm(`Delete checkpoint ${c.id} from disk?`)) return;
      await api('POST', `/v1/neural/checkpoints/${enc(c.id)}/${action}`); viewNeural();
    }); } }, label);
    const actions = [];
    if (c.status === 'shadow' || c.status === 'rejected' || c.status === 'retired' || c.status === 'candidate') actions.push(act('Promote', 'promote', 'sm'));
    if (c.status === 'shadow') actions.push(act('Reject', 'reject', 'sm danger'));
    if (c.status === 'active') actions.push(act('Retire', 'retire', 'sm danger'));
    if (c.status === 'rejected' || c.status === 'retired') actions.push(act('Delete', 'delete', 'sm ghost'));
    const detail = h('tr', { style: { display: 'none' } }, h('td', { colspan: 9 }, checkpointDetail(c)));
    const row = h('tr', { class: 'clickable', onclick: () => { detail.style.display = detail.style.display === 'none' ? '' : 'none'; } },
      h('td', {}, h('code', {}, c.id), h('div', { class: 'small muted' }, ago(c.created_at))),
      h('td', {}, h('span', { class: `badge ${STATUS_BADGE[c.status] || ''}` }, c.status)),
      h('td', { class: 'small' }, c.backbone === 'scratch' ? 'built-in' : c.backbone.split('/').pop(), h('div', { class: 'muted' }, r.init || '')),
      h('td', { class: 'num' }, num(r.train && r.train.examples)),
      h('td', { class: 'num' }, pct(t(c).accuracy)),
      h('td', { class: 'num', title: 'test log loss · base rates · previous active' }, dec(t(c).nll, 3),
        h('div', { class: 'small muted' }, `prior ${dec(r.prior && r.prior.nll, 3)}${r.baseline ? ` · prev ${dec(r.baseline.overall && r.baseline.overall.nll, 3)}` : ''}`)),
      h('td', { class: 'num' }, dec(t(c).ece, 3)),
      h('td', { class: 'small' }, sh ? `${sh.n} labels: ${dec(sh.nll / sh.n, 3)} vs live ${dec(sh.live_nll / sh.n, 3)}` : '—'),
      h('td', {}, h('div', { class: 'row' }, actions)));
    body.append(row, detail);
  }
  return h('div', { class: 'table-wrap' }, h('table', {},
    h('thead', {}, h('tr', {}, ['Checkpoint', 'Status', 'Backbone', 'Examples', 'Test acc.', 'Test log loss', 'ECE', 'Shadow', ''].map((x, i) =>
      h('th', { class: [3, 4, 5, 6].includes(i) ? 'num' : '' }, x)))), body),
    h('p', { class: 'small muted' }, 'Click a row for per-question results, the training curve and temperatures.'));
}

function checkpointDetail(c) {
  const r = c.report;
  const tasks = Object.entries((r.test && r.test.tasks) || {});
  const hist = (r.train && r.train.history) || [];
  return h('div', { class: 'grid-2', style: { padding: '8px 0' } },
    h('div', {},
      c.note ? h('p', { class: 'note' }, c.note) : null,
      tasks.length ? h('table', {}, h('thead', {}, h('tr', {}, h('th', {}, 'Question'), h('th', { class: 'num' }, 'n'), h('th', { class: 'num' }, 'Accuracy'),
        h('th', { class: 'num' }, 'Log loss'), h('th', { class: 'num' }, 'ECE'), h('th', { class: 'num' }, 'T'))),
        h('tbody', {}, tasks.map(([name, m]) => h('tr', {}, h('td', {}, h('a', { href: `#/questions/${enc(name)}` }, name)),
          h('td', { class: 'num' }, num(m.n)), h('td', { class: 'num' }, pct(m.accuracy)), h('td', { class: 'num' }, dec(m.nll, 3)),
          h('td', { class: 'num' }, dec(m.ece, 3)), h('td', { class: 'num' }, dec((r.temperatures || {})[name], 2))))))
        : h('div', { class: 'small muted' }, 'No labelled test examples.')),
    h('div', {},
      h('h3', {}, 'Training'),
      h('div', { class: 'small muted' }, hist.map(x => `epoch ${x.epoch}: train ${dec(x.train_loss, 3)} · val ${dec(x.val_nll, 3)}`).join('  ·  ')),
      h('div', { class: 'small muted' }, r.train ? `${num(r.train.examples)} train / ${num(r.train.val)} val / ${num(r.train.test)} test · ${r.train.seconds}s on ${r.train.device} · objective ${r.train.objective}` : '')));
}

// ================================================================== teacher
const PRESETS = {
  anthropic: { label: 'Anthropic (Claude)', provider: 'anthropic', base_url: '', model: 'claude-opus-5-5' },
  openai: { label: 'OpenAI', provider: 'openai', base_url: 'https://api.openai.com/v1', model: '' },
  gemini: { label: 'Google Gemini (OpenAI-compatible)', provider: 'openai', base_url: 'https://generativelanguage.googleapis.com/v1beta/openai', model: '' },
  groq: { label: 'Groq', provider: 'openai', base_url: 'https://api.groq.com/openai/v1', model: '' },
  openrouter: { label: 'OpenRouter', provider: 'openai', base_url: 'https://openrouter.ai/api/v1', model: '' },
  ollama: { label: 'Ollama (local)', provider: 'openai', base_url: 'http://localhost:11434/v1', model: '' },
  jev: { label: 'Jev-compatible decision API (e.g. self-hosted Laya)', provider: 'jev_compatible', base_url: '', model: '' },
  custom: { label: 'Other OpenAI-compatible', provider: 'openai', base_url: '', model: '' },
};
function remembered() { try { return JSON.parse(localStorage.getItem('desic.teacher') || 'null'); } catch { return null; } }

async function viewTeacher() {
  state.view = { kind: 'teacher' };
  const t = await api('GET', '/v1/teacher').catch(() => ({ configured: false, stats: {} }));
  if (!isView('teacher')) return;
  const saved = remembered() || {};
  const presetFor = cfg => (cfg.provider === 'jev_compatible' ? 'jev' : cfg.provider === 'anthropic' ? 'anthropic'
    : Object.keys(PRESETS).find(k => PRESETS[k].provider === 'openai' && PRESETS[k].base_url && PRESETS[k].base_url === cfg.base_url) || 'custom');
  const presetKey = t.configured ? presetFor(t) : (saved.preset || 'anthropic');
  const preset = h('select', {}, Object.entries(PRESETS).map(([k, p]) => h('option', { value: k, selected: k === presetKey }, p.label)));
  const key = h('input', { type: 'password', autocomplete: 'off', placeholder: t.api_key_set ? `saved on the server (${t.api_key_hint}) — leave empty to keep` : 'sk-…', value: saved.api_key || '' });
  const model = h('input', { value: t.model || saved.model || PRESETS[presetKey].model });
  const baseUrl = h('input', { value: t.base_url || saved.base_url || PRESETS[presetKey].base_url, placeholder: 'https://…' });
  const remember = h('input', { type: 'checkbox', checked: !!saved.api_key });
  const baseField = h('label', { class: 'field' }, h('span', {}, 'Endpoint / base URL'), baseUrl);
  const sync = initial => {
    const p = PRESETS[preset.value];
    baseField.style.display = p.provider === 'anthropic' ? 'none' : '';
    if (!initial) { baseUrl.value = p.base_url; model.value = p.model; }
    model.placeholder = p.provider === 'anthropic' ? 'claude-opus-5-5' : p.provider === 'jev_compatible' ? 'optional model id' : 'model name from your provider';
  };
  preset.onchange = () => sync(false); sync(true);
  const save = h('button', { class: 'primary' }, 'Save');
  save.onclick = () => guard(save, async () => {
    const p = PRESETS[preset.value];
    const body = { provider: p.provider, api_key: key.value.trim(), model: model.value.trim(), base_url: baseUrl.value.trim() };
    await api('PUT', '/v1/teacher', body);
    try {
      if (remember.checked) localStorage.setItem('desic.teacher', JSON.stringify({ ...body, preset: preset.value }));
      else localStorage.removeItem('desic.teacher');
    } catch { /* storage unavailable */ }
    toast('Teacher saved', 'good'); viewTeacher();
  });
  const test = h('button', {}, 'Test');
  const testOut = h('div');
  test.onclick = () => guard(test, async () => {
    const r = await api('POST', '/v1/teacher/test');
    testOut.replaceChildren(h('p', { class: 'done' }, `✓ Works — ${r.latency_ms} ms. P("hello there" is a greeting) = ${r.p_true}.`));
  });
  const off = h('button', { class: 'danger' }, 'Disconnect');
  off.onclick = () => guard(off, async () => {
    await api('DELETE', '/v1/teacher');
    try { localStorage.removeItem('desic.teacher'); } catch { /* ignore */ }
    viewTeacher();
  });
  const s = t.stats || {};
  mount(
    pageHead('Teacher', 'System 2 for your System 1: a large model (or a Jev-compatible decision API) that answers when the student is unsure — and teaches it.'),
    h('div', { class: 'grid-2' },
      h('div', { class: 'panel' }, h('h2', {}, 'Provider'),
        h('div', { class: 'form-grid' },
          h('label', { class: 'field' }, h('span', {}, 'Provider'), preset),
          h('label', { class: 'field' }, h('span', {}, 'API key'), key),
          h('label', { class: 'field' }, h('span', {}, 'Model'), model), baseField),
        h('label', { class: 'check small' }, remember, 'Remember in this browser (re-sent to the server after a restart)'),
        h('div', { class: 'note' }, '🔒 The key lives only in the Desic server’s memory and, if you tick the box, in this browser. It is never written to the database or logs. Alternatively start the server with DESIC_TEACHER_PROVIDER / DESIC_TEACHER_API_KEY / DESIC_TEACHER_MODEL.'),
        h('div', { class: 'row' }, save, t.configured ? test : null, t.configured ? off : null), testOut),
      h('div', { class: 'panel' }, h('h2', {}, 'Status'),
        t.configured ? h('div', {},
          h('p', {}, h('span', { class: 'badge good' }, 'connected'), ' ', h('strong', {}, t.provider), ` · ${t.model || 'default model'}`, t.api_key_hint ? ` · key ${t.api_key_hint}` : ''),
          h('p', { class: 'small' }, `${num(s.calls)} calls · ${num(s.errors)} errors${s.last_call ? ` · last ${ago(s.last_call)}` : ''}`),
          s.last_error ? h('p', { class: 'note' }, h('strong', {}, 'Last error: '), s.last_error) : null)
          : h('p', {}, h('span', { class: 'badge' }, 'not connected'), ' The student works on its own; it just cannot escalate.'),
        h('h3', {}, 'How the teacher is used'),
        h('ul', { class: 'small' },
          h('li', {}, 'When a student abstains (confidence below its threshold), Desic asks the teacher the same typed question and serves its answer.'),
          h('li', {}, 'The student learns from the teacher’s probabilities (soft labels, weight 0.5 by default) — distillation. Teacher calls fall as the student improves.'),
          h('li', {}, 'Human feedback always outranks the teacher, and is the only thing the accuracy / calibration metrics are measured on.'),
          h('li', {}, 'Data → Distill labels unlabelled rows in bulk; Data → Generate writes brand-new examples.')))));
}

async function restoreTeacher() {
  const saved = remembered();
  if (!saved || !saved.api_key) return;
  const t = await api('GET', '/v1/teacher').catch(() => null);
  if (t && !t.configured) await api('PUT', '/v1/teacher', saved).catch(() => {});
}

// ------------------------------------------------------------------ boot
window.addEventListener('hashchange', route);
connect();
restoreTeacher().finally(route);
