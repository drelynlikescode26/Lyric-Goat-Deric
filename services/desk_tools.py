"""Hard word locks and bounded notebook history, shared by desk actions."""
import re
import time
import uuid


def validated_locks(document, value):
    if value is None: return []
    if not isinstance(value,list) or len(value)>100: raise ValueError('Use up to 100 word or line locks.')
    lines=document.split('\n');result=[];identities=set()
    for lock in value:
        if not isinstance(lock,dict): raise ValueError('Check your protected words.')
        line,start,end=lock.get('line'),lock.get('start'),lock.get('end')
        identity=lock.get('id')
        if (any(type(n) is not int for n in (line,start,end)) or not 0<=line<len(lines)
                or not 0<=start<end<=len(lines[line]) or not isinstance(identity,str)
                or not re.fullmatch(r'[a-zA-Z0-9_-]{1,60}',identity) or identity in identities
                or lines[line][start:end]!=lock.get('text')):
            raise ValueError('A protected phrase no longer matches this draft. Check the locks before continuing.')
        if any(l["line"]==line and start<l["end"] and end>l["start"] for l in result):
            raise ValueError("Protected phrases overlap. Protect the whole phrase once.")
        identities.add(identity)
        result.append({'id':identity,'line':line,'start':start,'end':end,'text':lock['text']})
    return result


def lock_issues(candidate, selected, start, locks):
    problems=[]
    if not isinstance(candidate,dict) or not isinstance(candidate.get('lines'),list): return problems
    for offset,line in enumerate(candidate['lines']):
        if not isinstance(line,str): continue
        source_index=start+offset
        protected=sorted([l for l in locks if l['line']==source_index],key=lambda l:l['start'])
        cursor=0
        for lock in protected:
            if lock['start']==0 and lock['end']==len(selected[offset]):
                if line!=selected[offset]: problems.append(f'Line {offset+1}: keep the entire locked line exactly.')
                break
            matches=[m for m in re.finditer(re.escape(lock['text']),line) if m.start()>=cursor
                     and not (lock['text'][0].isalnum() and m.start()>0 and line[m.start()-1].isalnum())
                     and not (lock['text'][-1].isalnum() and m.end()<len(line) and line[m.end()].isalnum())]
            found=matches[0].start() if matches else -1
            if found<0: problems.append(f'Line {offset+1}: keep locked phrase {lock["text"]!r} exactly and in order.')
            else: cursor=found+len(lock['text'])
    return problems


def history_for(previous, notebook, kind='manual', label=''):
    history=list(previous.get('history',[])) if previous else []
    now=time.time()
    def entry(book,description):
        return {'id':uuid.uuid4().hex,'at':now,'label':description,'document':book['document'],'locks':book.get('locks',[]),'kind':kind}
    if previous and not history: history.append(entry(previous,'Earlier draft'))
    changed=not history or any(history[-1].get(k)!=notebook.get(k,[]) for k in ('document','locks'))
    if changed or kind=='checkpoint':
        item=entry(notebook,label or {'manual':'Writing edits','ai':'Lyric rewrite','hook':'Hook audition','restore':'Restored earlier words','checkpoint':'Saved milestone'}[kind])
        if kind=='manual' and len(history)>1 and history[-1].get('kind')=='manual' and now-history[-1]['at']<60:
            history[-1]=item
        else: history.append(item)
    return history[:1]+history[-59:] if len(history)>60 else history
