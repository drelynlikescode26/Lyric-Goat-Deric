const {JSDOM}=require('jsdom'),fs=require('fs'),assert=require('node:assert/strict');
const dom=new JSDOM(fs.readFileSync('templates/lyrics.html','utf8'),{url:'http://localhost/lyrics',runScripts:'outside-only',pretendToBeVisual:true}),w=dom.window,$=id=>w.document.getElementById(id);
let requests=[],book=null,historyId=0;
w.fetch=async(url,opts)=>{const input=opts?.body&&JSON.parse(opts.body);requests.push({url,input});
 if(url.endsWith('/options'))return{ok:true,json:async()=>({csrf:'csrf',ready:true})};
 if(url.endsWith('/notebooks')&&!opts)return{ok:true,json:async()=>({notebooks:book?[book]:[]})};
 if(url.endsWith('/radar'))return{ok:true,json:async()=>({lines:[{line:0,text:'A light at night',syllables:[4],hits:[{start:2,end:7,text:'light',family:'R1',kind:'exact'}]}],families:[{id:'R1',label:'Internal rhyme',kind:'exact',examples:['light','night']}],unknown:[],repetitions:[]})};
 if(url.endsWith('/suggestions')){const hook=input.action==='hook';return{ok:true,json:async()=>({action:input.action,start_line:input.start_line,end_line:input.end_line,hook_target:input.hook_target,suggestions:hook?[{text:'My own light\nI come back home',syllables:[3,4],warnings:[]},{text:'My own light\nI leave it on',syllables:[3,4],warnings:[]}]:[{text:'🔥 We see the light',syllables:[4],warnings:[]}]})};}
 if(!opts)return{ok:true,json:async()=>({notebook:book})};
 const history=book?.history||[];if(!history.length||history.at(-1).document!==input.document||input.edit_kind==='checkpoint')history.push({id:'h'+(++historyId),at:Date.now()/1000,label:input.edit_label||'Writing edits',document:input.document,locks:JSON.parse(JSON.stringify(input.locks||[]))});
 book={...input,id:'a'.repeat(32),revision:(book?.revision||0)+1,history};return{ok:true,json:async()=>({notebook:book})};};
w.eval(fs.readFileSync('static/lyrics.js','utf8'));
const tick=()=>new Promise(r=>setImmediate(r)),input=el=>el.dispatchEvent(new w.Event('input'));
async function flush(){await tick();await tick();await tick();}
(async()=>{await flush();const e=$('lyricDocument');e.value='🔥 I need the light\nI stay all night';input(e);const start=e.value.indexOf('light');e.setSelectionRange(start,start+5);e.dispatchEvent(new w.Event('select'));$('lockSelection').click();
 assert.match($('deskLocks').textContent,/light/);$('saveDraft').click();await flush();
 assert.equal(book.locks[0].start,13);assert.equal(book.locks[0].end,18);assert.equal(book.locks[0].text,'light');
 // Insert above a lock: its line moves, but the protected phrase survives.
 e.value='Before\n'+e.value;input(e);$('saveDraft').click();await flush();assert.equal(book.locks[0].line,1);
 // A manual edit inside protected words removes only that lock.
 e.value=e.value.replace('light','spark');input(e);assert.equal($('deskLocks').children.length,0);assert.match($('lockStatus').textContent,/lock was removed/);
 // Hook options rebuild from the same baseline; they never stack duplicate hooks.
 $('hookWorkspace').click();$('hookLines').value='2';$('hookLines').dispatchEvent(new w.Event('change'));$('hookSeed').value='My own light';input($('hookSeed'));
 $('getSuggestions').click();await flush();assert.equal(requests.filter(r=>r.url.endsWith('/suggestions')).length,1);
 w.document.querySelectorAll('.use-option')[0].click();await flush();assert.match(e.value,/\[Hook\]\nMy own light\nI come back home$/);
 w.document.querySelectorAll('.use-option')[1].click();await flush();assert.equal(e.value.match(/\[Hook\]/g).length,1);assert.match(e.value,/I leave it on$/);
 assert.equal(requests.filter(r=>r.url.endsWith('/suggestions')).length,1);
 $('undoSuggestion').click();assert.match(e.value,/I come back home$/);$('undoSuggestion').click();assert.equal(e.value.includes('[Hook]'),false);
 // A saved comparison supports restoring a single deleted line, then Undo.
 $('saveDraft').click();await flush();const old=e.value;e.value=old.replace('Before\n','');input(e);$('saveDraft').click();await flush();$('openHistory').click();await flush();
 const picker=$('historyPicker'),entry=book.history.find(h=>h.document===old);picker.value=entry.id;picker.dispatchEvent(new w.Event('change'));
 const bring=[...w.document.querySelectorAll('.restore-line')].find(b=>b.textContent==='Bring back line');assert.ok(bring);bring.click();assert.equal(e.value,old);$('undoSuggestion').click();assert.equal(e.value,old.replace('Before\n',''));
 // The radar exposes pronunciation evidence without calling the writer.
 const spend=requests.filter(r=>r.url.endsWith('/suggestions')).length;$('openRadar').click();await flush();assert.equal(w.document.querySelector('mark').textContent,'light');assert.match($('radarStatus').textContent,/no AI credit/);assert.equal(requests.filter(r=>r.url.endsWith('/suggestions')).length,spend);
 console.log('Writing workshop passed: Unicode locks, inserted-line tracking, direct edits, hook audition/Undo, individual line recovery, local Radar.');w.close();
})().catch(error=>{console.error(error);w.close();process.exitCode=1;});
