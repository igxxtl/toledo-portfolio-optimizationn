// Testes de logica em DOM simulado; nao substituem inspecao visual.
const fs = require('node:fs'), vm = require('node:vm'), assert = require('node:assert/strict');
const path = require('node:path');
const elements = new Map(), buttons = ['positivo','negativo','neutro'].map(label => ({dataset:{label},setAttribute(){}}));
function element(id) { if (!elements.has(id)) elements.set(id, {value:'',textContent:'',click(){this.onclick?.();}}); return elements.get(id); }
const storage = new Map(), events = {};
const context = vm.createContext({console,Date,Blob,URL,setTimeout,confirm:()=>true,
  localStorage:{getItem:k=>storage.get(k)||null,setItem:(k,v)=>storage.set(k,v)},window:{},
  document:{getElementById:element,querySelectorAll:()=>buttons,addEventListener:(k,v)=>events[k]=v,createElement:()=>({click(){}})}});
vm.runInContext(fs.readFileSync(path.join(__dirname,'noticias.js'),'utf8'),context);
vm.runInContext(fs.readFileSync(path.join(__dirname,'app.js'),'utf8'),context);
const run = s=>vm.runInContext(s,context);
assert.equal(run('rows.length'),300);
assert.equal(run('rows.filter(r=>r.rotulo_consenso).length'),0);
run("classify('positivo')");
assert.equal(run('index'),1);
assert.equal(run('rows[0].rotulo_consenso'),'positivo');
assert.equal(run('rows[0].rotulo_avaliador_1'),'positivo');
run("$('previous').onclick(); classify('negativo')");
assert.equal(run('rows[0].rotulo_consenso'),'negativo');
run('rows[0].observacao = \'Teste "aspas", vírgula\\nsegunda linha\'');
assert.equal(run('validate(parseCSV(csvText()))[0].observacao'), 'Teste "aspas", vírgula\nsegunda linha');
assert.throws(()=>run("validate([{sample_id:'999'}])"));
run('index=299; classify("neutro")');
assert.equal(run('index'),1); // volta para primeira pendente
run('rows.forEach(r=>{r.rotulo_consenso="neutro";r.rotulo_avaliador_1="neutro"}); render()');
assert.equal(element('complete').hidden,false);
assert.equal(element('progress').value,300);
assert.equal(run('parseCSV(csvText()).length'),300);
assert.equal(storage.size,1);
console.log('OK: 300 registros, classificacao, correcao, consenso, persistencia, CSV com aspas/quebras, validacao e conclusao.');
