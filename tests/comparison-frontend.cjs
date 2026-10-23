const { JSDOM } = require('jsdom');
const fs = require('node:fs');
const assert = require('node:assert/strict');
const root = require('node:path').resolve(__dirname, '..');
const id = 'a'.repeat(32);
const dom = new JSDOM(fs.readFileSync(root + '/templates/compare.html', 'utf8'), {
  url: `http://127.0.0.1:8765/compare?take=${id}`, runScripts: 'outside-only', pretendToBeVisual: true,
});
const w = dom.window, calls = [];
w.HTMLMediaElement.prototype.pause = () => {};
w.HTMLMediaElement.prototype.play = () => Promise.resolve();
const providers = [
  {id: 'openai_current', label: 'Current', ready: true, model: 'gpt-audio-1.5', description: 'Current task', destination: 'OpenAI'},
  {id: 'openai_detail', label: 'Detailed', ready: true, model: 'gpt-audio-1.5', description: 'Detailed task', destination: 'OpenAI'},
  {id: 'gemini', label: 'Gemini', ready: false, reason: 'Add a Google key.', description: 'Other listener'},
];
let report = {id, title: 'My take', duration: 8, runs: [{id: 'run1', revision: 'version1', created_at: 100,
  status: 'complete', review: {}, results: [
    {...providers[0], status: 'ok', elapsed_seconds: 1, phrases: [{start_time: 1, end_time: 2.5,
      target_syllables: 3, phonetic_hint: '<img src=x onerror=alert(1)>', delivery: 'Held run', timing_source: 'model_estimate'}]},
    {...providers[1], status: 'failed', error: 'Listening unavailable', phrases: []},
  ]}]};
w.fetch = async (url, opts) => {
  const body = opts?.body ? JSON.parse(opts.body) : null;
  calls.push({url, method: opts?.method || 'GET', body, headers: opts?.headers});
  let response;
  if (url.endsWith('/options')) response = {providers};
  else if (url.endsWith('/review')) {
    report.runs[0].review = {preferred: body.provider, ratings: {[body.provider]: {scores: body.scores, notes: body.notes}}};
    response = report;
  } else if (url.endsWith('/runs')) response = {status: 'running'};
  else if (url === '/api/comparisons') response = {comparisons: [report]};
  else response = report;
  return {ok: true, json: async () => JSON.parse(JSON.stringify(response))};
};
const tick = () => new Promise(r => setTimeout(r, 0));
w.eval(fs.readFileSync(root + '/static/compare.js', 'utf8'));
(async () => {
  await tick(); await tick();
  assert.equal(calls.filter(x => x.method === 'POST').length, 0, 'opening results must never buy a model call');
  assert.equal(w.document.querySelector('input[value=gemini]').disabled, true);
  assert.equal(w.document.querySelector('#comparisonResults img'), null, 'model text must not execute as HTML');
  assert.match(w.document.getElementById('comparisonResults').textContent, /Listening unavailable/);
  const loopButton = [...w.document.querySelectorAll('.listener-phrase button')][0];
  loopButton.click(); assert.equal(w.document.getElementById('sharedAudio').currentTime, 1);
  w.document.getElementById('sharedAudio').currentTime = 2.6;
  w.document.getElementById('sharedAudio').dispatchEvent(new w.Event('timeupdate'));
  assert.equal(w.document.getElementById('sharedAudio').currentTime, 1, 'loop uses the chosen phrase bounds');
  const form = w.document.querySelector('.listener-card form');
  for (const field of form.querySelectorAll('select')) field.value = '4';
  form.querySelector('textarea').value = 'Missed the second flow';
  form.querySelector('[name=preferred]').checked = true;
  form.dispatchEvent(new w.Event('submit', {bubbles: true, cancelable: true})); await tick();
  const review = calls.find(x => x.url.endsWith('/review'));
  assert.deepEqual(review.body.scores, {flow: 4, sounds: 4, syllables: 4});
  assert.ok(review.headers['X-Comparison-CSRF']);
  assert.match(w.document.getElementById('comparisonResults').textContent, /Your pick/);
  w.document.getElementById('runComparison').click(); await tick(); await tick();
  const run = calls.find(x => x.url.endsWith('/runs'));
  assert.deepEqual(run.body.providers, ['openai_current', 'openai_detail']);
  assert.equal(w.document.querySelectorAll('.compare-run').length, 1, 'previous results remain visible');
  console.log('Comparison UI checks passed: no auto-spend, unavailable providers, safe model text, phrase loops, saved ratings, selected listeners.');
  dom.window.close();
})().catch(e => { console.error(e); dom.window.close(); process.exitCode = 1; });
