(() => {
  const $ = id => document.getElementById(id);
  const csrf = document.querySelector('meta[name="comparison-csrf"]').content;
  const audio = $('sharedAudio');
  let active = null, providers = [], pollTimer = null, loop = null, lastRendered = '';

  function element(tag, text, className) {
    const node = document.createElement(tag);
    if (text !== undefined) node.textContent = text;
    if (className) node.className = className;
    return node;
  }
  function status(text, error = false) {
    $('compareStatus').textContent = text;
    $('compareStatus').classList.toggle('compare-error', error);
  }
  async function api(url, options = {}) {
    const res = await fetch(url, { ...options, headers: { ...options.headers, 'X-Comparison-CSRF': csrf } });
    let data;
    try { data = await res.json(); } catch (_) { throw Error('The app did not return a usable response. Refresh this page and try again.'); }
    if (!res.ok) throw Error(data.error || 'Could not complete that request.');
    return data;
  }
  async function history() {
    const data = await api('/api/comparisons');
    const select = $('savedTakes');
    select.replaceChildren(element('option', 'Choose a take…'));
    select.firstChild.value = '';
    for (const take of data.comparisons) {
      const option = element('option', `${take.title} · ${take.duration}s`);
      option.value = take.id; select.appendChild(option);
    }
    if (active) select.value = active.id;
  }
  function choices() {
    const container = $('providerChoices'); container.replaceChildren();
    for (const provider of providers) {
      const label = element('label', undefined, 'compare-provider');
      const check = document.createElement('input'); check.type = 'checkbox'; check.value = provider.id;
      check.name = 'provider'; check.disabled = !provider.ready;
      check.checked = provider.ready && ['openai_current', 'openai_detail', 'transcribe'].includes(provider.id);
      const body = element('span');
      body.append(element('strong', provider.label), element('small', provider.description),
        element('small', provider.ready ? `${provider.model} · ${provider.destination}` : provider.reason));
      label.append(check, body); container.appendChild(label);
    }
  }
  async function load(id) {
    clearTimeout(pollTimer); loop = null;
    const report = await api(`/api/comparisons/${id}`);
    active = report; lastRendered = '';
    audio.src = `/api/comparisons/${id}/audio`;
    $('compareControls').hidden = false;
    $('activeTitle').textContent = report.title;
    $('takeInfo').textContent = `${report.duration}s · Saved locally · The same saved audio is used by every listener and rerun.`;
    $('savedTakes').value = id;
    window.history.replaceState(null, '', `/compare?take=${id}`);
    render(report); schedule(report);
    status('Take ready. Select listeners, then run the comparison.');
  }
  function schedule(report) {
    clearTimeout(pollTimer);
    if (!report.runs.some(r => r.status === 'running')) return;
    const id = report.id;
    pollTimer = setTimeout(async () => {
      try {
        const fresh = await api(`/api/comparisons/${id}`);
        if (active?.id !== id) return;
        active = fresh; render(fresh); schedule(fresh);
        if (!fresh.runs.some(r => r.status === 'running')) status('Comparison finished. Replay the phrases and rate what matches your take.');
      } catch (e) { status(e.message, true); }
    }, 1500);
  }
  function render(report) {
    const encoded = JSON.stringify(report.runs);
    if (encoded === lastRendered) return;
    lastRendered = encoded;
    $('runComparison').disabled = report.runs.some(r => r.status === 'running');
    const container = $('comparisonResults'); container.replaceChildren();
    for (const [index, run] of [...report.runs.entries()].reverse()) {
      const section = element('section', undefined, 'compare-run');
      section.append(element('h2', `Run ${index + 1} · ${new Date(run.created_at * 1000).toLocaleString()}`),
        element('p', `Listening setup ${run.revision} · ${run.status === 'running' ? 'Listening…' : 'Saved results'}`, 'muted'));
      if (run.error) section.appendChild(element('p', run.error, 'compare-error'));
      const grid = element('div', undefined, 'compare-grid');
      for (const result of run.results) grid.appendChild(card(report, run, result));
      section.appendChild(grid); container.appendChild(section);
    }
  }
  function card(report, run, result) {
    const preferred = run.review.preferred === result.id;
    const node = element('article', undefined, `card listener-card${preferred ? ' compare-preferred' : ''}`);
    node.append(element('h3', result.label + (preferred ? ' · Your pick' : '')),
      element('p', result.model, 'muted'));
    if (result.status === 'pending') { node.appendChild(element('p', 'Listening…')); return node; }
    if (result.status !== 'ok') { node.appendChild(element('p', result.error || 'Unavailable', 'compare-error')); return node; }
    node.appendChild(element('p', `Completed in ${result.elapsed_seconds}s`, 'muted'));
    if (result.timing_note) node.appendChild(element('p', result.timing_note, 'muted'));
    if (result.summary) node.appendChild(element('p', result.summary));
    if (result.feel?.length) node.appendChild(element('p', result.feel.join(' · ')));
    if (result.transcript !== undefined) node.append(element('h3', 'Words it heard'), element('pre', result.transcript || '(No words returned)'));
    if (result.notation) {
      const details = element('details'); details.append(element('summary', 'Predicted singing notation'), element('pre', result.notation));
      node.appendChild(details);
    }
    const phrases = element('div', undefined, 'phrases');
    for (const [i, phrase] of (result.phrases || []).entries()) {
      const part = element('div', undefined, 'listener-phrase');
      const count = phrase.target_syllables ?? phrase.syllables ?? '?';
      part.append(element('strong', `Phrase ${i + 1} · ${phrase.start_time.toFixed(2)}–${phrase.end_time.toFixed(2)}s`),
        element('span', `${count} syllables${phrase.timing_source === 'model_estimate' ? ' · AI estimate' : ' · Acoustic timing / estimated count'}`, 'muted'),
        element('p', phrase.phonetic_hint || '(No sound description returned)'),
        element('p', phrase.delivery || ''));
      if (phrase.heard_words) part.appendChild(element('p', `Words: ${phrase.heard_words}`));
      const play = element('button', 'Loop this phrase', 'ws-btn'); play.type = 'button';
      play.addEventListener('click', () => {
        loop = { start: phrase.start_time, end: phrase.end_time };
        audio.currentTime = loop.start;
        audio.play().catch(() => status('Press Play on the recording to hear this phrase.'));
      });
      part.appendChild(play); phrases.appendChild(part);
    }
    node.appendChild(phrases);
    if (run.status === 'complete') node.appendChild(reviewForm(report, run, result));
    return node;
  }
  function reviewForm(report, run, result) {
    const form = document.createElement('form');
    const saved = run.review.ratings?.[result.id];
    form.appendChild(element('strong', 'Your ears decide · 1 = poor, 5 = close'));
    for (const [key, text] of [['flow', 'Flow and phrase breaks'], ['sounds', 'Mumble sounds'], ['syllables', 'Syllable count']]) {
      const label = element('label', text), select = document.createElement('select');
      select.name = key; select.required = true; select.setAttribute('aria-label', `${result.label}: ${text}`);
      const empty = element('option', 'Choose a rating'); empty.value = ''; select.appendChild(empty);
      const na = element('option', 'Not supplied / not applicable'); na.value = 'na'; select.appendChild(na);
      for (let i = 1; i <= 5; i++) { const option = element('option', String(i)); option.value = String(i); select.appendChild(option); }
      select.value = saved ? (saved.scores[key] === null ? 'na' : String(saved.scores[key])) : ''; label.appendChild(select); form.appendChild(label);
    }
    const notesLabel = element('label', 'What did it miss?'), notes = document.createElement('textarea');
    notes.name = 'notes'; notes.maxLength = 2000; notes.value = saved?.notes || '';
    notesLabel.appendChild(notes); form.appendChild(notesLabel);
    const preferredLabel = element('label', undefined, 'check-label'), check = document.createElement('input');
    check.type = 'checkbox'; check.name = 'preferred'; check.checked = run.review.preferred === result.id;
    preferredLabel.append(check, element('span', 'My preferred listener for this take'));
    const button = element('button', saved ? 'Update my rating' : 'Save my rating', 'ws-btn'); button.type = 'submit';
    form.append(preferredLabel, button);
    form.addEventListener('submit', async event => {
      event.preventDefault(); button.disabled = true;
      try {
        const fields = new FormData(form);
        const scores = Object.fromEntries(['flow', 'sounds', 'syllables'].map(k => [k, fields.get(k) === 'na' ? null : Number(fields.get(k))]));
        active = await api(`/api/comparisons/${report.id}/review`, { method: 'POST',
          headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ run_id: run.id,
            provider: result.id, scores, notes: fields.get('notes'), preferred: fields.has('preferred') }) });
        render(active); status('Your rating is saved locally. It is feedback, not automatic model training.');
      } catch (e) { status(e.message, true); button.disabled = false; }
    });
    return form;
  }
  audio.addEventListener('timeupdate', () => { if (loop && audio.currentTime >= loop.end) audio.currentTime = loop.start; });
  audio.addEventListener('ended', () => { if (loop) { audio.currentTime = loop.start; audio.play().catch(() => {}); } });
  $('stopLoop').addEventListener('click', () => { loop = null; audio.pause(); });
  $('takeForm').addEventListener('submit', async event => {
    event.preventDefault(); const button = $('saveTake'); button.disabled = true;
    status('Saving and analyzing this take locally. No paid listening calls yet.');
    try {
      const file = $('takeFile').files[0];
      if (!file) throw Error('Choose an audio recording first.');
      const form = new FormData(); form.append('audio', file); form.append('title', $('takeTitle').value || file.name);
      form.append('bpm', $('takeBpm').value || '0');
      const report = await api('/api/comparisons', { method: 'POST', body: form });
      await load(report.id); await history();
    } catch (e) { status(e.message, true); }
    finally { button.disabled = false; }
  });
  $('savedTakes').addEventListener('change', event => { if (event.target.value) load(event.target.value).catch(e => status(e.message, true)); });
  $('runComparison').addEventListener('click', async () => {
    const selected = [...document.querySelectorAll('input[name=provider]:checked')].map(x => x.value);
    if (!active || !selected.length) { status('Select at least one ready listener.', true); return; }
    $('runComparison').disabled = true; status('Listening to the same saved take with the selected options…');
    try {
      await api(`/api/comparisons/${active.id}/runs`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ providers: selected }) });
      active = await api(`/api/comparisons/${active.id}`); render(active); schedule(active);
    } catch (e) { status(e.message, true); $('runComparison').disabled = false; }
  });
  window.addEventListener('pagehide', () => clearTimeout(pollTimer));
  (async () => {
    try {
      const options = await api('/api/comparisons/options'); providers = options.providers; choices(); await history();
      const id = new URLSearchParams(window.location.search).get('take');
      if (id) await load(id);
    } catch (e) { status(e.message, true); }
  })();
})();
