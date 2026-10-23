(() => {
 const badge=document.getElementById('storageBadge');
 async function refresh(){
  if(document.hidden)return;
  try{const response=await fetch('/api/storage/status',{cache:'no-store'});if(!response.ok)return;const s=await response.json();
   const title=s.error?'Cloud needs attention':s.enabled?(s.pending?'Saving to cloud…':'Cloud queue up to date'):'Stored on this Mac';
   if(badge){badge.textContent=s.enabled?(s.pending?`Cloud · ${s.pending} pending`:'Cloud saved'):'Mac storage';badge.dataset.state=s.error?'error':s.state;badge.title=s.error||title;}
   const fields={cloudState:title,cloudDetail:s.error||(s.enabled?'Pending files stay here until the cloud confirms them.':'Connect R2 below to start your private cloud library.'),cloudPending:String(s.pending),cloudCache:((s.cache_bytes||0)/1048576).toFixed(1)+' MB'};
   for(const [id,value]of Object.entries(fields)){const e=document.getElementById(id);if(e)e.textContent=value;}
  }catch(_){if(badge){badge.textContent='Storage status unavailable';badge.dataset.state='error';}}
 }
 refresh();setInterval(refresh,10000);document.addEventListener('visibilitychange',refresh);
})();
