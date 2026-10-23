"""Local pronunciation clues, never a judgement of a performed rhyme scheme."""
import re
from services.lyric_craft import pronunciations, line_syllable_options, VOWEL_FAMILIES

WORD = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)*")
STOP = {'a','an','the','i','you','he','she','it','we','they','my','your','our','to','of','and','or','but','in','on','at','is','was','be','am','are','for','with','that','this','so','as'}


def tails(variants, vowels=1):
    result=set()
    for phones in variants[:4]:
        positions=[i for i,p in enumerate(phones) if p[-1:].isdigit()]
        if len(positions)>=vowels:
            if vowels==1:
                stressed=[i for i in positions if phones[i].endswith(('1','2'))]
                start=stressed[-1] if stressed else positions[-1]
            else: start=positions[-vowels]
            result.add(tuple(re.sub(r'\d','',p) for p in phones[start:]))
    return result


def analyze(document):
    rows=[];words=[];phrases=[];unknown=[]
    for line_no,line in enumerate(document.split('\n')):
        row={'line':line_no,'text':line,'syllables':sorted(line_syllable_options(line)), 'hits':[]};rows.append(row)
        if re.fullmatch(r'\s*\[[^]]+\]\s*',line): continue
        found=list(WORD.finditer(line))
        for j,match in enumerate(found):
            text=match.group(); normalized=text.lower().replace('’',"'"); variants=pronunciations(normalized)
            if not variants:
                if normalized not in unknown: unknown.append(normalized)
                continue
            item={'line':line_no,'start':match.start(),'end':match.end(),'text':text,'normalized':normalized,
                  'ending':j==len(found)-1,'variants':variants,'keys':tails(variants)}
            # Function words create noise inside a line, but still count at its ending.
            if normalized not in STOP or item['ending']: words.append(item)
            multi=tails(variants,2)
            if multi and (normalized not in STOP or item['ending']): phrases.append({**item,'keys':multi})
        # End-of-line two-word chains catch across-word multisyllable matches.
        if len(found)>=2:
            a,b=found[-2:];av,bv=pronunciations(a.group()),pronunciations(b.group())
            if av and bv and all(sum(p[-1:].isdigit() for p in v)==1 for v in bv):
                variants=[x+y for x in av[:2] for y in bv[:2]]
                phrases.append({'line':line_no,'start':a.start(),'end':b.end(),'text':line[a.start():b.end()],
                    'normalized':line[a.start():b.end()].lower(),'ending':True,'variants':variants,'keys':tails(variants,2)})
    families=[];seen=set();seen_sounds=set()
    def groups(items,kind):
        index={}
        for item in items:
            for sound in item['keys']: index.setdefault(sound,[]).append(item)
        for sound,members in index.items():
            if len({m['normalized'] for m in members})<2: continue
            ids=tuple(sorted((m['line'],m['start'],m['end']) for m in members))
            if ids in seen or sound in seen_sounds: continue
            seen.add(ids)
            if kind=='chain' and len({(m['line'],m['end']) for m in members})<2: continue
            seen_sounds.add(sound)
            group='R'+str(len(families)+1)
            syllables=sum(p in VOWEL_FAMILIES for p in sound)
            label='Multisyllable' if syllables>=2 else ('End rhyme' if all(m['ending'] for m in members) else 'Internal / end rhyme')
            family={'id':group,'kind':'multi' if syllables>=2 else 'exact','label':label,'examples':list(dict.fromkeys(m['text'] for m in members))[:6]}
            families.append(family)
            for m in members: rows[m['line']]['hits'].append({k:m[k] for k in ('start','end','text')}|{'family':group,'kind':family['kind']})
            if len(families)>=24: return
    groups(phrases,'chain')
    groups(words,'word')
    # Vowel-only near rhymes are separate, deliberately tentative clues.
    near={}
    for word in words:
        if not word['ending']: continue
        for sound in word['keys']:
            vowels=tuple(p for p in sound if p in VOWEL_FAMILIES)
            if vowels: near.setdefault(vowels,[]).append(word)
    for members in near.values():
        unique={(m['line'],m['start']):m for m in members};members=list(unique.values())
        if len(members)<2 or len({m['normalized'] for m in members})<2: continue
        if all(a['keys']&b['keys'] for a in members for b in members): continue
        if len(families)>=30: break
        group='R'+str(len(families)+1);families.append({'id':group,'kind':'near','label':'Possible slant · shared vowels','examples':list(dict.fromkeys(m['text'] for m in members))[:6]})
        for m in members: rows[m['line']]['hits'].append({k:m[k] for k in ('start','end','text')}|{'family':group,'kind':'near'})
    repetition={}
    for word in words: repetition.setdefault(word['normalized'],[]).append(word['line']+1)
    return {'lines':rows,'families':families,'unknown':unknown[:30],
            'repetitions':[{'word':w,'lines':sorted(set(locations))} for w,locations in repetition.items() if len(locations)>1][:25],
            'note':'English pronunciation clues, not a score. Accents, slang and sung vowels can change the rhyme. Repeated words are listed separately.'}
