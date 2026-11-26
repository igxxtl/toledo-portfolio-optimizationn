'use strict';
const $ = id => document.getElementById(id);
const labels = ['positivo', 'negativo', 'neutro'];
const columns = ['sample_id','ticker','titulo','data','rotulo_avaliador_1','rotulo_avaliador_2','rotulo_consenso','observacao'];
const key = 'tcc-news-review-' + DATASET.id;
let rows = DATASET.rows.map(r => ({...r})), index = 0, handle = null, writes = Promise.resolve();
const status = message => { $('status').textContent = message; };
function validate(candidate) {
  if (!Array.isArray(candidate) || candidate.length !== rows.length) throw Error('O arquivo precisa conter as mesmas 300 notícias.');
  const ids = new Set();
  return DATASET.rows.map(original => {
    const r = candidate.find(r => String(r.sample_id) === String(original.sample_id));
    if (!r || ids.has(String(r.sample_id)) || r.titulo !== original.titulo || r.ticker !== original.ticker || r.data !== original.data) throw Error('As notícias do arquivo não correspondem a esta amostra.');
    ids.add(String(r.sample_id));
    for (const field of ['rotulo_avaliador_1','rotulo_avaliador_2','rotulo_consenso']) {
      if (r[field] && !labels.includes(r[field])) throw Error('Rótulo inválido no arquivo.');
    }
    return Object.fromEntries(columns.map(c => [c, r[c] ?? '']));
  });
}
function persist() {
  try { localStorage.setItem(key, JSON.stringify({rows, index})); $('saveStatus').textContent = handle ? 'Salvo no navegador · gravando CSV…' : 'Progresso salvo neste navegador'; }
  catch { $('saveStatus').textContent = 'Não foi possível salvar no navegador. Baixe o CSV ou backup.'; }
  if (handle) {
    const csv = csvText(), target = handle;
    writes = writes.then(async () => { const stream = await target.createWritable(); await stream.write(csv); await stream.close(); $('saveStatus').textContent = 'CSV salvo no arquivo vinculado'; }).catch(e => { handle = null; $('saveStatus').textContent = 'Falha ao salvar CSV. Baixe uma cópia.'; status(e.message); });
  }
}
function render() {
  const r = rows[index], done = rows.filter(r => labels.includes(r.rotulo_consenso)).length;
  $('ticker').textContent = r.ticker;
  $('position').textContent = `Notícia ${index + 1} de ${rows.length}`;
  $('headline').textContent = r.titulo;
  const date = new Date(r.data);
  $('date').textContent = Number.isNaN(+date) ? r.data : 'Publicada em ' + date.toLocaleDateString('pt-BR', {timeZone:'UTC'});
  $('observation').value = r.observacao;
  $('progressText').textContent = `${done} de ${rows.length} avaliadas · ${rows.length - done} pendentes`;
  $('progress').max = rows.length; $('progress').value = done;
  $('complete').hidden = done !== rows.length;
  $('previous').disabled = index === 0; $('next').disabled = index === rows.length - 1;
  document.querySelectorAll('[data-label]').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.label === r.rotulo_consenso)));
}
function classify(label) {
  rows[index].rotulo_avaliador_1 = label; rows[index].rotulo_consenso = label;
  status(`Notícia ${index + 1}: ${label}.`);
  if (index < rows.length - 1) index++;
  else { const pending = rows.findIndex(r => !r.rotulo_consenso); if (pending >= 0) index = pending; }
  persist(); render();
}
function csvText() {
  const quote = v => '"' + String(v ?? '').replaceAll('"','""') + '"';
  return '\uFEFF' + [columns, ...rows.map(r => columns.map(c => r[c]))].map(r => r.map(quote).join(',')).join('\r\n');
}
function parseCSV(text) {
  const records = []; let record = [], value = '', quoted = false;
  text = text.replace(/^\uFEFF/, '');
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (c === '"') { if (quoted && text[i+1] === '"') { value += '"'; i++; } else quoted = !quoted; }
    else if (!quoted && (c === ',' || c === '\n' || c === '\r')) {
      record.push(value); value = '';
      if (c !== ',') { if (record.some(v => v !== '')) records.push(record); record = []; if (c === '\r' && text[i+1] === '\n') i++; }
    } else value += c;
  }
  if (quoted) throw Error('CSV com aspas incompletas.');
  if (value || record.length) { record.push(value); records.push(record); }
  const header = records.shift();
  if (!header || !columns.every(c => header.includes(c))) throw Error('Colunas esperadas não encontradas no CSV.');
  return records.map(r => Object.fromEntries(header.map((c,i) => [c,r[i] ?? ''])));
}
function download(content, type, name) {
  const url = URL.createObjectURL(new Blob([content], {type}));
  const a = document.createElement('a'); a.href = url; a.download = name; a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
}
document.querySelectorAll('[data-label]').forEach(b => b.onclick = () => classify(b.dataset.label));
$('previous').onclick = () => { index = Math.max(0,index-1); persist(); render(); };
$('next').onclick = () => { index = Math.min(rows.length-1,index+1); persist(); render(); };
$('pending').onclick = () => { const next = rows.findIndex((r,i) => i > index && !r.rotulo_consenso); const first = rows.findIndex(r => !r.rotulo_consenso); if (first < 0) return status('Todas as notícias foram avaliadas.'); index = next < 0 ? first : next; persist(); render(); };
$('observation').oninput = () => { rows[index].observacao = $('observation').value; persist(); };
$('export').onclick = () => { download(csvText(), 'text/csv;charset=utf-8', 'rotulacao_humana.csv'); status('CSV baixado. Guarde-o para incorporar à análise.'); };
$('backup').onclick = () => download(JSON.stringify({dataset:DATASET.id,rows,index},null,2), 'application/json', 'avaliacao-noticias-backup.json');
$('restore').onclick = () => $('file').click();
async function importText(text, json) {
  const obj = json ? JSON.parse(text) : null;
  if (json && obj.dataset !== DATASET.id) throw Error('Backup pertence a outra amostra.');
  const candidate = validate(json ? obj.rows : parseCSV(text));
  if (!confirm('Importar substituirá as respostas atuais pelas respostas desse arquivo. Continuar?')) return false;
  rows = candidate; index = Math.max(0, rows.findIndex(r => !r.rotulo_consenso)); persist(); render(); return true;
}
$('file').onchange = async () => { try { const f = $('file').files[0]; if (f && await importText(await f.text(), f.name.endsWith('.json'))) status('Respostas importadas.'); } catch(e) { status(e.message); } finally { $('file').value = ''; } };
$('connect').onclick = async () => {
  if (!window.showOpenFilePicker) return status('Este navegador não permite salvar diretamente. Use “Baixar CSV preenchido”.');
  try {
    const [target] = await window.showOpenFilePicker({multiple:false,types:[{description:'CSV da avaliação',accept:{'text/csv':['.csv']}}]});
    const existing = validate(parseCSV(await (await target.getFile()).text()));
    const existingAnswers = existing.some(r => r.rotulo_consenso || r.observacao);
    if (existingAnswers) {
      if (!confirm('O CSV já contém respostas. Carregar essas respostas e substituir o progresso atual do navegador?')) return;
    }
    if (!confirm('Salvar as respostas desta tela diretamente no CSV selecionado? As colunas de avaliação serão atualizadas.')) return;
    if (existingAnswers) { rows = existing; index = Math.max(0, rows.findIndex(r => !r.rotulo_consenso)); }
    handle = target; persist(); render(); status('CSV vinculado. As próximas respostas serão salvas nesse arquivo.');
  } catch(e) { if (e.name !== 'AbortError') status(e.message); }
};
document.addEventListener('keydown', e => {
  if (['INPUT','TEXTAREA','SELECT'].includes(e.target.tagName) || e.ctrlKey || e.altKey || e.metaKey || e.repeat) return;
  if (['1','2','3'].includes(e.key)) { e.preventDefault(); classify(labels[Number(e.key)-1]); }
  if (e.key === 'ArrowLeft') $('previous').click();
  if (e.key === 'ArrowRight') $('next').click();
});
try { const saved = JSON.parse(localStorage.getItem(key)); if (saved) { rows = validate(saved.rows); index = Math.max(0,Math.min(rows.length-1, Number(saved.index) || 0)); } $('saveStatus').textContent = saved ? 'Progresso recuperado deste navegador' : 'Pronto para começar · 300 notícias'; }
catch { $('saveStatus').textContent = 'Não foi possível recuperar o progresso. Importe um backup, se necessário.'; }
render();
