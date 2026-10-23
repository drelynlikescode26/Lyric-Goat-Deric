(() => {
 const $=id=>document.getElementById(id),editor=$('lyricDocument'),key='lyric-goat-desk-draft-v1';
 let csrf='',ready=false,busy=false,id=null,revision=0,action='rewrite',timer=null,saving=null,conflict=false;
 let undo=[],batch=null,notebooks=[],locks=[],history=[],lastText='',pendingEdit=null,radar=null,radarFilter='all',historySource='';
 const node=(tag,text,cls)=>{const e=document.createElement(tag);if(text!==undefined)e.textContent=text;if(cls)e.className=cls;return e;};
 const clone=value=>JSON.parse(JSON.stringify(value));
 const points=text=>Array.from(text),cpLength=text=>points(text).length;
 const content=()=>({document:editor.value,title:$('deskTitle').value,client:$('deskClient').value,genre:$('deskGenre').value,
  language:$('deskLanguage').value,theme:$('deskTheme').value,instruction:$('editInstruction').value,locks,
  hook_seed:$('hookSeed').value,hook_lines:Number($('hookLines').value),hook_target:$('hookTarget').value});
 const snapshot=()=>JSON.stringify({...content(),action,keep_length:$('keepLength').checked,keep_rhyme:$('keepRhyme').checked,
  rhyme_word:$('rhymeWord').value,hook_seed:$('hookSeed').value,hook_lines:Number($('hookLines').value),hook_target:$('hookTarget').value});
 function backup(){try{localStorage.setItem(key,JSON.stringify({...content(),id,revision,conflict,hook_seed:$('hookSeed').value}));}catch(_){$('saveStatus').textContent='Browser backup unavailable. Save or export your draft.';}}
 function selection(){const text=editor.value,a=editor.selectionStart,b=editor.selectionEnd,start=text.slice(0,a).split('\n').length-1;
  let end=text.slice(0,b).split('\n').length-1;if(b>a&&text[b-1]==='\n')end--;return{start_line:start,end_line:Math.max(start,end)};}
 function update(){const s=selection(),lines=editor.value.split('\n'),chosen=lines.slice(s.start_line,s.end_line+1),valid=chosen.length<=8&&chosen.every(t=>t.trim()&&!/^\s*\[[^\]]+\]\s*$/.test(t));
  $('documentStats').textContent=`${lines.filter(t=>t.trim()).length} lines · ${editor.value.trim()?editor.value.trim().split(/\s+/).length:0} words`;
  $('selectionSummary').textContent=editor.value?(s.start_line===s.end_line?`Working on line ${s.start_line+1}`:`Working on lines ${s.start_line+1}–${s.end_line+1}`):'Click a line or select a few.';
  const hookAppend=action==='hook'&&$('hookTarget').value==='append';
  $('getSuggestions').disabled=!csrf||!ready||busy||!(hookAppend?(!!editor.value.trim()||!!$('hookSeed').value.trim()||!!$('deskTheme').value.trim()):valid);
  $('lockSelection').disabled=!valid; $('undoSuggestion').disabled=!undo.length||undo.at(-1).after!==editor.value;
  const stale=batch&&batch.snapshot!==snapshot();document.querySelectorAll('.use-option').forEach(button=>button.disabled=!!stale);
  if(stale)$('suggestStatus').textContent='Your draft or direction changed. Ask for fresh options before applying these.';
 }
 function changed(kind='manual',label=''){
  if(typeof kind!=='string')kind='manual';
  if(lastText!==editor.value){const beforeCount=locks.length;locks=shiftLocks(lastText,editor.value,locks);lastText=editor.value;renderLocks();if(locks.length<beforeCount)$('lockStatus').textContent='You edited protected text directly, so its lock was removed. Protect the new words again if you want.';}
  if(kind!=='manual'||!pendingEdit)pendingEdit={kind,label};
  backup();update();clearTimeout(timer);if(!conflict)timer=setTimeout(()=>save(),1200);
  $('saveStatus').textContent=conflict?'A newer saved draft exists. Save a copy to keep these edits.':'Edits backed up here. Saving draft…';
 }
 function shiftLocks(before,after,source,allowMoves=false){
  const a=points(before),b=points(after);let prefix=0,suffix=0;
  while(prefix<Math.min(a.length,b.length)&&a[prefix]===b[prefix])prefix++;
  while(suffix<Math.min(a.length,b.length)-prefix&&a[a.length-1-suffix]===b[b.length-1-suffix])suffix++;
  const end=a.length-suffix,delta=b.length-a.length,oldLines=before.split('\n'),newLines=after.split('\n'),result=[],cursors={};
  for(const lock of source){const absolute=oldLines.slice(0,lock.line).reduce((n,l)=>n+cpLength(l)+1,0)+lock.start,stop=absolute+cpLength(lock.text);let start;
   if(stop<=prefix)start=absolute;else if(absolute>=end)start=absolute+delta;
   else if(allowMoves){const line=newLines[lock.line]||'',from=cursors[lock.line]||0,found=line.indexOf(lock.text,from);if(found<0)continue;cursors[lock.line]=found+lock.text.length;start=newLines.slice(0,lock.line).reduce((n,l)=>n+cpLength(l)+1,0)+cpLength(line.slice(0,found));}
   else continue;
   const line=b.slice(0,start).filter(c=>c==='\n').length,lineStart=b.slice(0,start).lastIndexOf('\n')+1,column=start-lineStart;
   if(points(newLines[line]||'').slice(column,column+cpLength(lock.text)).join('')===lock.text)result.push({...lock,line,start:column,end:column+cpLength(lock.text)});
  }return result;
 }
 function renderLocks(){const list=$('deskLocks');list.replaceChildren();
  for(const lock of locks){const chip=node('button',`L${lock.line+1} · ${lock.text.length>45?lock.text.slice(0,42)+'…':lock.text} ×`,'lock-chip');chip.type='button';chip.title='Unlock '+lock.text;chip.setAttribute('aria-label',chip.title);chip.addEventListener('click',()=>{locks=locks.filter(l=>l.id!==lock.id);renderLocks();changed();$('lockStatus').textContent='Words unlocked.';});list.append(chip);}
 }
 function protect(){const lines=editor.value.split('\n'),selected=selection(),a=editor.selectionStart,b=editor.selectionEnd;let offset=0,added=0;
  for(let i=0;i<lines.length;i++){const line=lines[i],lineEnd=offset+line.length;
   if(i>=selected.start_line&&i<=selected.end_line&&line.trim()){
    const start=a===b?0:Math.max(0,a-offset),end=a===b?line.length:Math.min(line.length,b-offset);
    let cpStart=cpLength(line.slice(0,start)),cpEnd=cpLength(line.slice(0,end));
    const overlapping=locks.filter(l=>l.line===i&&cpStart<l.end&&cpEnd>l.start);if(overlapping.length){cpStart=Math.min(cpStart,...overlapping.map(l=>l.start));cpEnd=Math.max(cpEnd,...overlapping.map(l=>l.end));}
    const text=points(line).slice(cpStart,cpEnd).join('');
    if(text.trim()&&!locks.some(l=>l.line===i&&l.start===cpStart&&l.end===cpEnd)){
     locks=locks.filter(l=>!(l.line===i&&l.start>=cpStart&&l.end<=cpEnd));
     if(locks.length>=100){$('lockStatus').textContent='You can protect up to 100 phrases.';break;}
     locks.push({id:'l'+Date.now().toString(36)+Math.random().toString(36).slice(2,8),line:i,start:cpStart,end:cpEnd,text});added++;
    }
   }offset=lineEnd+1;
  }locks.sort((a,b)=>a.line-b.line||a.start-b.start);renderLocks();changed();$('lockStatus').textContent=added?'Protected. AI options must keep these exact words.':'These words are already protected.';
 }
 async function refreshLibrary(){const response=await fetch('/api/lyric-desk/notebooks');if(!response.ok)return;notebooks=(await response.json()).notebooks||[];
  const select=$('notebookPicker');select.replaceChildren();const empty=node('option','Saved lyric drafts');empty.value='';select.append(empty);
  for(const book of notebooks){const option=node('option',book.title);option.value=book.id;select.append(option);}select.value=id||'';
 }
 async function save(copy=false){clearTimeout(timer);if(!csrf){if(editor.value.trim()){$('saveStatus').textContent='Connection is still starting. Your browser draft is kept.';return false;}return true;}
  if(!editor.value.trim()&&!id)return true;if(saving){await saving;return save(copy);}if(conflict&&!copy)return false;
  const before=snapshot(),value=content(),oldId=id,oldRevision=revision,event=pendingEdit||{kind:'manual',label:''};
  saving=(async()=>{try{const response=await fetch('/api/lyric-desk/notebooks'+(oldId&&!copy?'/'+oldId:''),{method:oldId&&!copy?'PUT':'POST',headers:{'Content-Type':'application/json','X-Lyric-Desk-CSRF':csrf},body:JSON.stringify({...value,revision:oldRevision,edit_kind:event.kind,edit_label:event.label})});const result=await response.json();
   if(!response.ok){if(response.status===409)conflict=true;throw Error(result.error||'Could not save this draft.');}
   id=result.notebook.id;revision=result.notebook.revision;history=result.notebook.history||[];conflict=false;if(pendingEdit===event)pendingEdit=null;backup();await refreshLibrary();
   $('saveStatus').textContent=result.cloud_retry_needed?'Saved on this Mac. Cloud upload needs a retry in Cloud library.':'Saved on this Mac · private cloud sync in background.';
   if(before!==snapshot()){backup();timer=setTimeout(()=>save(),600);}return true;
  }catch(error){backup();$('saveStatus').textContent=error.message+' Your browser backup is kept; export a copy if needed.';return false;}})();
  const ok=await saving;saving=null;return ok;
 }
 function restore(value){for(const [field,element]of Object.entries({document:'lyricDocument',title:'deskTitle',client:'deskClient',genre:'deskGenre',language:'deskLanguage',theme:'deskTheme',instruction:'editInstruction'}))$(element).value=value[field]|| (field==='language'?'clean':'');
  id=value.id||null;revision=value.revision||0;conflict=!!value.conflict;locks=value.locks||[];history=value.history||[];lastText=editor.value;pendingEdit=null;batch=null;undo=[];$('hookSeed').value=value.hook_seed||'';$('hookLines').value=String(value.hook_lines||4);$('hookTarget').value=value.hook_target||'append';$('deskSuggestions').replaceChildren();renderLocks();update();
 }
 async function open(identity){if(identity===id)return;if(!(await save())){$('notebookPicker').value=id||'';return;}try{const response=await fetch('/api/lyric-desk/notebooks/'+identity);if(!response.ok)throw Error('Could not open this draft.');restore((await response.json()).notebook);backup();$('saveStatus').textContent='Saved draft opened. Your changes will autosave.';}catch(error){$('saveStatus').textContent=error.message;}}
 function renderOptions(result,source){batch={...result,snapshot:source,document:editor.value,locks:clone(locks)};$('deskSuggestions').replaceChildren();
  for(const [i,option]of result.suggestions.entries()){const card=node('article',undefined,'desk-option'),meta=node('div',undefined,'option-meta');meta.append(node('span',result.action==='hook'?(option.direction||['Close & personal','Refrain','Setup & payoff'][i]).toUpperCase():`OPTION ${String(i+1).padStart(2,'0')}`),node('span',option.syllables.join(' / ')+' syllables · est.'));card.append(meta,node('p',option.text));
   if(option.warnings.length)card.append(node('div','Check the syllable fit before using these words.','option-warning'));
   const button=node('button',result.action==='continue'?'Add this next line ↗':result.action==='hook'?'Try this hook ↗':'Use these words ↗','use-option');button.type='button';button.addEventListener('click',()=>apply(option,i));card.append(button);$('deskSuggestions').append(card);
  }update();
 }
 function apply(option,index){if(!batch||batch.snapshot!==snapshot())return;const before=editor.value,beforeLocks=clone(locks),beforeSelection=[editor.selectionStart,editor.selectionEnd],lines=batch.document.split('\n');let selectedStart=batch.start_line;
  if(batch.action==='hook'&&batch.hook_target==='append'){
   const base=batch.document.replace(/\n+$/,'');editor.value=(base?base+'\n\n':'')+'[Hook]\n'+option.text;selectedStart=(base?base.split('\n').length+2:1);
  }else{if(batch.action==='continue'){lines.splice(batch.end_line+1,0,...option.text.split('\n'));selectedStart=batch.end_line+1;}else lines.splice(batch.start_line,batch.end_line-batch.start_line+1,...option.text.split('\n'));editor.value=lines.join('\n');}
  locks=shiftLocks(batch.document,editor.value,batch.locks,true);lastText=editor.value;renderLocks();
  const offset=editor.value.split('\n').slice(0,selectedStart).join('\n').length+(selectedStart?1:0);editor.setSelectionRange(offset,offset+option.text.length);
  undo.push({before,after:editor.value,beforeLocks,beforeSelection,batch});undo=undo.slice(-30);batch.snapshot=snapshot();changed(batch.action==='hook'?'hook':'ai',(batch.action==='hook'?'Hook':'Rewrite')+' · option '+(index+1));
  save();$('suggestStatus').textContent='Words updated. Audition another option from this set, or Undo. Your other lines stay in place.';editor.focus({preventScroll:true});
 }
 async function suggest(){if($('getSuggestions').disabled)return;batch=null;busy=true;update();const source=snapshot();let selected=selection();if(action==='hook'&&$('hookTarget').value==='append')selected={start_line:0,end_line:0};
  const request={...content(),...selected,action,keep_length:$('keepLength').checked,keep_rhyme:$('keepRhyme').checked,rhyme_word:$('rhymeWord').value,hook_seed:$('hookSeed').value,hook_lines:Number($('hookLines').value),hook_target:$('hookTarget').value};
  $('deskSuggestions').replaceChildren();$('suggestStatus').textContent=action==='hook'?'Building three hook directions. Your song stays editable.':'Finding a few ways to say it. Your draft stays editable.';$('suggestButtonText').textContent='Working…';document.querySelector('.desk-assistant').classList.add('busy');
  try{if(!(await save()))throw Error('Save this draft or save a copy before asking for new words.');const response=await fetch('/api/lyric-desk/suggestions',{method:'POST',headers:{'Content-Type':'application/json','X-Lyric-Desk-CSRF':csrf},body:JSON.stringify(request)});const result=await response.json();if(!response.ok)throw Error(result.error||'Could not write suggestions.');
   if(source!==snapshot()){$('suggestStatus').textContent='Your draft changed while the writer was working. Ask again for options that fit the new draft.';batch=null;return;}
   renderOptions(result,source);$('suggestStatus').textContent='Sing each option in the pocket. Choose one to try; switching these options uses no extra AI credit.';
  }catch(error){$('suggestStatus').textContent=error.message+' Your draft was kept.';}
  finally{busy=false;$('suggestButtonText').textContent=action==='hook'?'Build three hook directions':'Find new ways to say it';document.querySelector('.desk-assistant').classList.remove('busy');update();}
 }
 let lineLengthPreference=true;
 function setWorkspace(hook){if(hook){lineLengthPreference=$('keepLength').checked;$('keepLength').checked=false;}else $('keepLength').checked=lineLengthPreference;document.querySelectorAll('[data-action]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.action==='rewrite')));action=hook?'hook':'rewrite';$('hookTools').hidden=!hook;$('lineTools').hidden=hook;$('hookWorkspace').setAttribute('aria-pressed',String(hook));$('lineWorkspace').setAttribute('aria-pressed',String(!hook));$('assistantTitle').textContent=hook?'Make it stick.':'Work the line.';$('suggestButtonText').textContent=hook?'Build three hook directions':'Find new ways to say it';$('rhymeField').hidden=true;batch=null;$('deskSuggestions').replaceChildren();update();}
 function showDialog(dialog){if(dialog.showModal)dialog.showModal();else dialog.setAttribute('open','');}
 function closeDialog(dialog){if(dialog.close)dialog.close();else dialog.removeAttribute('open');}
 async function showRadar(){showDialog($('radarDialog'));$('radarStatus').textContent='Finding the sounds…';const source=editor.value;
  try{const response=await fetch('/api/lyric-desk/radar',{method:'POST',headers:{'Content-Type':'application/json','X-Lyric-Desk-CSRF':csrf},body:JSON.stringify(content())});if(!response.ok)throw Error('Could not scan this draft.');const result=await response.json();if(source!==editor.value)throw Error('The draft changed. Close and reopen Radar for the latest words.');radar=result;renderRadar();}
  catch(error){$('radarStatus').textContent=error.message;}
 }
 function renderRadar(){if(!radar)return;const families=radar.families||[],visible=families.filter(f=>radarFilter==='all'||f.kind===radarFilter),ids=new Set(visible.map(f=>f.id));$('radarLines').replaceChildren();$('radarFamilies').replaceChildren();
  for(const row of radar.lines){const line=node('div',undefined,'radar-line'),number=node('span',String(row.line+1),'radar-line-number'),text=node('span',undefined,'radar-words'),chars=points(row.text);let cursor=0;
   const hits=row.hits.filter(h=>ids.has(h.family)).sort((a,b)=>a.start-b.start||b.end-a.end);
   for(const hit of hits){if(hit.start<cursor)continue;text.append(document.createTextNode(chars.slice(cursor,hit.start).join('')));const mark=node('mark',chars.slice(hit.start,hit.end).join(''),'rhyme-mark '+hit.kind);mark.style.setProperty('--rhyme-hue',String((Number(hit.family.slice(1))*67)%360));mark.title=hit.family+' · '+families.find(f=>f.id===hit.family)?.label;text.append(mark);cursor=hit.end;}
   text.append(document.createTextNode(chars.slice(cursor).join('')||' '));line.append(number,text,node('span',row.text.trim()&&!/^\s*\[/.test(row.text)?row.syllables.join('/'):'','radar-count'));$('radarLines').append(line);
  }
  for(const family of visible){const card=node('div',undefined,'radar-family');card.style.setProperty('--rhyme-hue',String((Number(family.id.slice(1))*67)%360));card.append(node('strong',family.id+' · '+family.label),node('p',family.examples.join(' / ')));$('radarFamilies').append(card);}
  $('radarStatus').textContent=visible.length?`${visible.length} sound families · syllable counts on the right · no AI credit used.`:'No clear matches in this view. That is not a verdict on your rhyme or delivery.';
  $('radarRepetitions').textContent=radar.repetitions?.length?'Repeated words (separate from rhymes): '+radar.repetitions.map(r=>`${r.word} · L${r.lines.join(', ')}`).join('; '):'';
  $('radarUnknown').textContent=radar.unknown?.length?'Unrecognized pronunciation: '+radar.unknown.join(', ')+'. Sing these to judge the sound.':'';
 }
 async function showHistory(){if(!(await save()))return;populateHistory();showDialog($('historyDialog'));}
 function populateHistory(){const picker=$('historyPicker'),selected=picker.value;picker.replaceChildren();for(const entry of [...history].reverse()){const option=node('option',new Date(entry.at*1000).toLocaleString()+' · '+entry.label);option.value=entry.id;picker.append(option);}if(history.some(h=>h.id===selected))picker.value=selected;else{const earlier=[...history].reverse().find(h=>h.document!==editor.value);if(earlier)picker.value=earlier.id;}renderHistory();}
 // Line alignment uses LCS so inserted/deleted lines can be restored without replacing their neighbors.
 function lineDiff(oldText,newText){const a=oldText.split('\n'),b=newText.split('\n'),dp=Array.from({length:a.length+1},()=>new Uint16Array(b.length+1));
  for(let i=a.length-1;i>=0;i--)for(let j=b.length-1;j>=0;j--)dp[i][j]=a[i]===b[j]?dp[i+1][j+1]+1:Math.max(dp[i+1][j],dp[i][j+1]);
  const rows=[];let i=0,j=0;while(i<a.length||j<b.length){if(i<a.length&&j<b.length&&a[i]===b[j]){rows.push({old:a[i++],current:b[j++],index:j-1,same:true});continue;}
   const removed=[],added=[];const position=j;while(i<a.length||j<b.length){if(i<a.length&&j<b.length&&a[i]===b[j])break;if(i<a.length&&(j===b.length||dp[i+1][j]>=dp[i][j+1]))removed.push(a[i++]);else added.push(b[j++]);}
   for(let k=0;k<Math.max(removed.length,added.length);k++)rows.push({old:removed[k],current:added[k],index:position+Math.min(k,added.length)});
  }return rows;
 }
 function renderHistory(){const entry=history.find(h=>h.id===$('historyPicker').value);$('historyDiff').replaceChildren();historySource=editor.value;$('restoreVersion').disabled=!entry||entry.document===editor.value;
  if(!entry){$('historyDiff').append(node('p','Your first saved draft will appear here.'));return;}
  for(const row of lineDiff(entry.document,editor.value)){const pair=node('div',undefined,'diff-row'+(row.same?'':' diff-changed')),old=node('div',row.old===undefined?'—':row.old,'diff-old'),current=node('div',row.current===undefined?'—':row.current,'diff-current');
   if(!row.same){const button=node('button',row.old===undefined?'Remove added line':row.current===undefined?'Bring back line':'Restore this line','restore-line');button.type='button';button.addEventListener('click',()=>{
     if(historySource!==editor.value){$('historyStatus').textContent='The draft changed. Select the version again to refresh this comparison.';return;}
     const lines=editor.value.split('\n');lines.splice(row.index,row.current===undefined?0:1,...(row.old===undefined?[]:[row.old]));replaceDocument(lines.join('\n'),null,'Restored one line');renderHistory();
    });old.append(button);
   }pair.append(old,current);$('historyDiff').append(pair);
  }
 }
 function replaceDocument(text,nextLocks,label){const before=editor.value,beforeLocks=clone(locks);editor.value=text;locks=nextLocks?clone(nextLocks):shiftLocks(before,text,locks);lastText=text;renderLocks();batch=null;$('deskSuggestions').replaceChildren();undo.push({before,after:text,beforeLocks,beforeSelection:[0,0],batch:null});changed('restore',label);$('historyStatus').textContent='Restored. Your previous draft stays in the timeline, and Undo can bring it back.';}
 editor.addEventListener('input',changed);for(const event of ['select','click','keyup'])editor.addEventListener(event,update);
 editor.addEventListener('keydown',event=>{if((event.metaKey||event.ctrlKey)&&event.key==='Enter'){event.preventDefault();suggest();}});
 for(const name of ['deskTitle','deskClient','deskTheme','editInstruction'])$(name).addEventListener('input',changed);
 for(const name of ['deskGenre','deskLanguage'])$(name).addEventListener('change',changed);
 for(const name of ['keepLength','keepRhyme','rhymeWord','hookSeed','hookLines','hookTarget'])$(name).addEventListener(['rhymeWord','hookSeed'].includes(name)?'input':'change',()=>{batch=null;$('deskSuggestions').replaceChildren();update();backup();if(['hookSeed','hookLines','hookTarget'].includes(name))changed();});
 document.querySelectorAll('[data-action]').forEach(button=>button.addEventListener('click',()=>{action=button.dataset.action;document.querySelectorAll('[data-action]').forEach(b=>b.setAttribute('aria-pressed',String(b===button)));$('rhymeField').hidden=action!=='rhyme';batch=null;$('deskSuggestions').replaceChildren();update();}));
 $('lineWorkspace').addEventListener('click',()=>setWorkspace(false));$('hookWorkspace').addEventListener('click',()=>setWorkspace(true));
 $('lockSelection').addEventListener('click',protect);$('openRadar').addEventListener('click',showRadar);$('openHistory').addEventListener('click',showHistory);
 document.querySelectorAll('[data-close]').forEach(button=>button.addEventListener('click',()=>closeDialog($(button.dataset.close))));
 document.querySelectorAll('[data-radar-filter]').forEach(button=>button.addEventListener('click',()=>{radarFilter=button.dataset.radarFilter;document.querySelectorAll('[data-radar-filter]').forEach(b=>b.setAttribute('aria-pressed',String(b===button)));renderRadar();}));
 $('historyPicker').addEventListener('change',renderHistory);
 $('checkpointDraft').addEventListener('click',async()=>{pendingEdit={kind:'checkpoint',label:'Saved milestone'};if(await save())populateHistory();});
 $('restoreVersion').addEventListener('click',()=>{const entry=history.find(h=>h.id===$('historyPicker').value);if(!entry||historySource!==editor.value)return;replaceDocument(entry.document,entry.locks||[],'Restored earlier draft');renderHistory();});
 $('getSuggestions').addEventListener('click',suggest);$('saveDraft').addEventListener('click',()=>save());$('saveCopy').addEventListener('click',()=>save(true));
 $('notebookPicker').addEventListener('change',event=>{if(event.target.value)open(event.target.value);});
 $('newNotebook').addEventListener('click',async()=>{if(!(await save()))return;restore({});backup();$('notebookPicker').value='';$('saveStatus').textContent='New lyric draft. Paste a song or start writing.';editor.focus();});
 $('undoSuggestion').addEventListener('click',()=>{if(!undo.length||undo.at(-1).after!==editor.value)return;const last=undo.pop();editor.value=last.before;locks=last.beforeLocks||[];lastText=editor.value;renderLocks();editor.setSelectionRange(...last.beforeSelection);if(batch&&batch===last.batch)batch.snapshot=snapshot();else{batch=null;$('deskSuggestions').replaceChildren();}changed('restore','Undid last edit');$('suggestStatus').textContent='Edit undone. Your previous lyrics are back.';});
 $('copyDraft').addEventListener('click',async()=>{try{await navigator.clipboard.writeText(editor.value);$('saveStatus').textContent='Full lyrics copied.';}catch(_){$('saveStatus').textContent='Copy unavailable. Use Export .txt to keep the full lyrics.';}});
 $('downloadDraft').addEventListener('click',()=>{const url=URL.createObjectURL(new Blob([editor.value],{type:'text/plain;charset=utf-8'})),a=node('a');a.href=url;a.download=($('deskTitle').value.trim()||'Lyric-Goat-lyrics').replace(/[^a-z0-9 _-]/gi,'').slice(0,80)+'.txt';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);});
 window.addEventListener('beforeunload',()=>backup());
 (async()=>{try{let saved;try{saved=JSON.parse(localStorage.getItem(key)||'null');}catch(_){}if(saved)restore(saved);
  const response=await fetch('/api/lyric-desk/options');if(!response.ok)throw Error('Open Lyric Desk from your Mac studio.');const options=await response.json();csrf=options.csrf;ready=options.ready;await refreshLibrary();update();
  if(id){try{const response=await fetch('/api/lyric-desk/notebooks/'+id);if(response.ok)history=(await response.json()).notebook.history||[];}catch(_){}}
  let imported;try{imported=JSON.parse(localStorage.getItem('lyric-goat-desk-import-v1')||'null');}catch(_){}
  if(imported&&typeof imported.document==='string'&&await save()){restore(imported);backup();localStorage.removeItem('lyric-goat-desk-import-v1');changed('checkpoint','From Vocal Studio');}
  if(!ready)$('suggestStatus').textContent='Add your OpenAI key in Settings to enable suggestions. Writing and saving still work.';
  else if(saved)$('saveStatus').textContent='Your browser draft is back. Save a copy if another tab has newer edits.';
 }catch(error){$('suggestStatus').textContent=error.message;}})();
})();
