import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from app import app
from services import lyric_desk as module

class LyricDeskTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        env=patch.dict(os.environ,{'LYRIC_GOAT_LOCAL_SETUP':'1','LYRIC_GOAT_STORAGE':'local','LYRIC_GOAT_DATA_DIR':self.temp.name,'OPENAI_API_KEY':'sk-test-only'},clear=True)
        env.start();self.addCleanup(env.stop)
        config=patch.dict(app.config,{'TESTING':True,'TRUSTED_HOSTS':['localhost','127.0.0.1']});config.start();self.addCleanup(config.stop)
        self.client=app.test_client();self.csrf=self.client.get('/api/lyric-desk/options').json['csrf'];self.headers={'X-Lyric-Desk-CSRF':self.csrf}
        self.payload={'document':'You left last night\nI kept the light\nYou know my name','start_line':1,'end_line':1,'action':'rewrite','keep_length':True,'language':'clean'}
        self.options=[{'lines':['I stayed all night']},{'lines':['You lost the light']},{'lines':['I let you go']}]

    def test_opening_and_saving_make_no_ai_calls_and_stale_saves_are_rejected(self):
        with patch.object(module,'_client') as writer:
            self.assertEqual(self.client.get('/lyrics').status_code,200)
            created=self.client.post('/api/lyric-desk/notebooks',headers=self.headers,json={**self.payload,'title':'Client draft'}).json['notebook']
            reopened=self.client.get('/api/lyric-desk/notebooks/'+created['id']).json['notebook']
            self.assertEqual(reopened['document'],self.payload['document'])
            saved=self.client.put('/api/lyric-desk/notebooks/'+created['id'],headers=self.headers,json={**created,'document':'A new first line'}).json['notebook']
            self.assertEqual(saved['revision'],2)
            conflict=self.client.put('/api/lyric-desk/notebooks/'+created['id'],headers=self.headers,json=created)
            self.assertEqual(conflict.status_code,409);writer.assert_not_called()

    def test_suggestions_only_return_selected_lines_with_complete_context(self):
        with patch.object(module,'request_suggestions',return_value=self.options) as writer:
            response=self.client.post('/api/lyric-desk/suggestions',headers=self.headers,json=self.payload)
        self.assertEqual(response.status_code,200,response.json)
        self.assertEqual(response.json['start_line'],1);self.assertEqual(len(response.json['suggestions']),3)
        sent=writer.call_args.args[0]
        self.assertEqual(sent['selected_lines'],['I kept the light']);self.assertEqual(sent['document'],self.payload['document'])
        self.assertEqual(sent['existing_rhyme_clues'],[{'line':2,'rhymes_with_lines':[1]}])

    def test_next_line_keeps_source_separate_and_requests_one_line(self):
        with patch.object(module,'request_suggestions',return_value=self.options) as writer:
            response=self.client.post('/api/lyric-desk/suggestions',headers=self.headers,json={**self.payload,'action':'continue','start_line':0,'end_line':1})
        self.assertEqual(response.status_code,200);self.assertEqual(writer.call_args.args[0]['output_line_count'],1)
        self.assertEqual(response.json['action'],'continue')

    def test_invalid_selection_is_rejected_without_spend(self):
        with patch.object(module,'request_suggestions') as writer:
            for change in [{'start_line':True},{'start_line':-1},{'end_line':99},{'document':'[Verse]','start_line':0,'end_line':0},{'document':''},{'document':'x'*30001},{'action':'arbitrary'}]:
                response=self.client.post('/api/lyric-desk/suggestions',headers=self.headers,json={**self.payload,**change})
                self.assertEqual(response.status_code,400,response.json)
            writer.assert_not_called()

    def test_repair_is_bounded_and_clean_mode_never_returns_profanity(self):
        bad=[{'lines':['Fuck all that shit']}]*3
        with patch.object(module,'request_suggestions',side_effect=[bad,self.options]) as writer:
            response=self.client.post('/api/lyric-desk/suggestions',headers=self.headers,json=self.payload)
        self.assertEqual(writer.call_count,2);self.assertEqual(response.status_code,200)
        with patch.object(module,'request_suggestions',return_value=bad) as writer:
            response=self.client.post('/api/lyric-desk/suggestions',headers=self.headers,json=self.payload)
        self.assertEqual(writer.call_count,2);self.assertEqual(response.status_code,422)

    def test_remaining_count_misses_are_visible(self):
        with patch.object(module,'request_suggestions',return_value=[{'lines':['A very long replacement with too many syllables']}]*3) as writer:
            response=self.client.post('/api/lyric-desk/suggestions',headers=self.headers,json=self.payload)
        self.assertEqual(writer.call_count,2);self.assertTrue(response.json['suggestions'][0]['warnings'])

    def test_requests_are_private_and_csrf_protected(self):
        self.assertEqual(self.client.post('/api/lyric-desk/suggestions',json=self.payload).status_code,403)
        self.assertEqual(self.client.get('/lyrics',environ_base={'REMOTE_ADDR':'192.168.1.5'}).status_code,403)
        self.assertEqual(self.client.post('/api/lyric-desk/suggestions',headers={**self.headers,'Origin':'https://evil.example'},json=self.payload).status_code,403)

    def test_notebook_cloud_saves_have_their_own_namespace(self):
        with patch.object(module.cloud_storage,'queue_file') as cloud:
            response=self.client.post('/api/lyric-desk/notebooks',headers=self.headers,json=self.payload)
        self.assertEqual(response.status_code,201)
        book=response.json['notebook'];self.assertEqual(cloud.call_args.args[1],'notebooks/'+book['id']+'.json')
        self.assertTrue((Path(self.temp.name)/'notebooks'/(book['id']+'.json')).exists())

    def test_fixed_writer_model_private_structured_response(self):
        client=SimpleNamespace(responses=SimpleNamespace(create=lambda **kwargs:None))
        with patch.object(module,'_client') as factory:
            factory.return_value.responses.create.return_value=SimpleNamespace(status='completed',output_text=json.dumps({'alternatives':self.options}))
            module.request_suggestions({'document':'My draft'},'rnb','clean',1)
        args=factory.return_value.responses.create.call_args.kwargs
        self.assertEqual(args['model'],'gpt-6-luna');self.assertFalse(args['store']);self.assertEqual(args['reasoning'],{'effort':'low'})

    def test_word_lock_violations_are_repaired_and_never_returned(self):
        lock={'id':'keep-light','line':1,'start':11,'end':16,'text':'light'}
        good=[{'lines':['I see the light']},{'lines':['You kept the light']},{'lines':['I need the light']}]
        with patch.object(module,'request_suggestions',side_effect=[self.options,good]) as writer:
            response=self.client.post('/api/lyric-desk/suggestions',headers=self.headers,json={**self.payload,'locks':[lock]})
        self.assertEqual(response.status_code,200,response.json);self.assertEqual(writer.call_count,2)
        self.assertTrue(all('light' in c['text'] for c in response.json['suggestions']))
        with patch.object(module,'request_suggestions',return_value=[{'lines':['I see the lighthouse']}]*3) as writer:
            response=self.client.post('/api/lyric-desk/suggestions',headers=self.headers,json={**self.payload,'locks':[lock]})
        self.assertEqual(response.status_code,422);self.assertEqual(writer.call_count,2)

    def test_fully_locked_selection_is_rejected_without_spend(self):
        lock={'id':'full','line':1,'start':0,'end':16,'text':'I kept the light'}
        # The exact endpoint is validated, rather than trusting stale lock text.
        lock['end']=len(lock['text'])
        with patch.object(module,'request_suggestions') as writer:
            response=self.client.post('/api/lyric-desk/suggestions',headers=self.headers,json={**self.payload,'locks':[lock]})
        self.assertEqual(response.status_code,400);writer.assert_not_called()

    def test_multi_line_selection_can_rewrite_around_a_locked_line(self):
        text='You left last night';lock={'id':'first','line':0,'start':0,'end':len(text),'text':text}
        candidates=[{'lines':[text,'I see the light']}]*3
        with patch.object(module,'request_suggestions',return_value=candidates):
            response=self.client.post('/api/lyric-desk/suggestions',headers=self.headers,json={**self.payload,'start_line':0,'locks':[lock]})
        self.assertEqual(response.status_code,200,response.json);self.assertTrue(response.json['suggestions'][0]['text'].startswith(text+'\n'))

    def test_unicode_lock_offsets_and_stale_locks(self):
        payload={**self.payload,'document':'🔥 I need the light','start_line':0,'end_line':0,'locks':[{'id':'unicode','line':0,'start':13,'end':18,'text':'light'}]}
        with patch.object(module,'request_suggestions',return_value=[{'lines':['🔥 I see the light']}]*3):
            response=self.client.post('/api/lyric-desk/suggestions',headers=self.headers,json=payload)
        self.assertEqual(response.status_code,200,response.json)
        with patch.object(module,'request_suggestions') as writer:
            response=self.client.post('/api/lyric-desk/suggestions',headers=self.headers,json={**payload,'document':'Different words'})
        self.assertEqual(response.status_code,400);writer.assert_not_called()

    def test_hook_append_uses_whole_song_context_and_exact_seed(self):
        hooks=[{'lines':['I let you go','I take it slow','I need to know','I let you go']}]*3
        payload={**self.payload,'document':'[Verse]\nYou left last night','start_line':0,'end_line':0,'action':'hook','hook_target':'append','hook_lines':4,'hook_seed':'I let you go','keep_length':False}
        with patch.object(module,'request_suggestions',return_value=hooks) as writer:
            response=self.client.post('/api/lyric-desk/suggestions',headers=self.headers,json=payload)
        self.assertEqual(response.status_code,200,response.json);self.assertEqual(writer.call_args.args[0]['output_line_count'],4)
        self.assertEqual(writer.call_args.args[0]['document'],payload['document'])
        bad=[{'lines':['Another thought']*4}]*3
        with patch.object(module,'request_suggestions',return_value=bad) as writer:
            response=self.client.post('/api/lyric-desk/suggestions',headers=self.headers,json=payload)
        self.assertEqual(response.status_code,422);self.assertEqual(writer.call_count,2)

    def test_blank_hook_theme_allowed_and_bad_hook_shape_has_no_spend(self):
        hooks=[{'lines':['I let you go','I take it slow']}]*3
        payload={**self.payload,'document':'','start_line':0,'end_line':0,'action':'hook','hook_lines':2,'theme':'Letting go','keep_length':False}
        with patch.object(module,'request_suggestions',return_value=hooks):
            response=self.client.post('/api/lyric-desk/suggestions',headers=self.headers,json=payload)
        self.assertEqual(response.status_code,200,response.json)
        with patch.object(module,'request_suggestions') as writer:
            for change in [{'theme':''},{'hook_lines':3},{'hook_target':'replace'},{'hook_seed':'shit'}]:
                response=self.client.post('/api/lyric-desk/suggestions',headers=self.headers,json={**payload,**change})
                self.assertEqual(response.status_code,400,response.json)
            writer.assert_not_called()

    def test_history_roundtrip_coalesces_typing_and_protects_earlier_words(self):
        created=self.client.post('/api/lyric-desk/notebooks',headers=self.headers,json=self.payload).json['notebook']
        updated=self.client.put('/api/lyric-desk/notebooks/'+created['id'],headers=self.headers,json={**created,'document':'New lyrics','edit_kind':'ai'}).json['notebook']
        self.assertEqual(updated['history'][0]['document'],self.payload['document'])
        self.assertEqual(updated['history'][-1]['document'],'New lyrics')
        for i in range(3):
            updated=self.client.put('/api/lyric-desk/notebooks/'+created['id'],headers=self.headers,json={**updated,'document':'Typed '+str(i)}).json['notebook']
        self.assertEqual(len(updated['history']),3)
        milestone=self.client.put('/api/lyric-desk/notebooks/'+created['id'],headers=self.headers,json={**updated,'edit_kind':'checkpoint'}).json['notebook']
        self.assertEqual(len(milestone['history']),4)
        self.assertEqual(milestone['history'][-1]['label'],'Saved milestone')
        reopened=self.client.get('/api/lyric-desk/notebooks/'+created['id']).json['notebook']
        self.assertEqual(reopened['history'],milestone['history'])

    def test_history_has_bounded_size_and_ignores_client_supplied_history(self):
        from services.desk_tools import history_for
        old={'document':'original','history':[]}
        for n in range(70):
            new={'document':str(n),'locks':[]};new['history']=history_for(old,new,'ai');old=new
        self.assertEqual(len(old['history']),60);self.assertEqual(old['history'][0]['document'],'original')
        created=self.client.post('/api/lyric-desk/notebooks',headers=self.headers,json={**self.payload,'history':[{'document':'injected'}]}).json['notebook']
        self.assertEqual(len(created['history']),1);self.assertEqual(created['history'][0]['document'],self.payload['document'])

    def test_radar_is_private_and_uses_no_ai(self):
        with patch.object(module,'_client') as writer:
            response=self.client.post('/api/lyric-desk/radar',headers=self.headers,json={'document':'A light at night\nI hold it tight'})
        self.assertEqual(response.status_code,200);self.assertTrue(response.json['families']);writer.assert_not_called()
        self.assertEqual(self.client.post('/api/lyric-desk/radar',json={'document':'test'}).status_code,403)

    def test_hook_refrain_repair_and_direction_labels_survive_filtered_options(self):
        payload={**self.payload,'action':'hook','hook_lines':2,'hook_seed':'I let you go','keep_length':False}
        bad=[{'lines':['I let you go','I take it slow']},{'lines':['I let you go','I take it slow']},{'lines':['Where did we go','I let you go']}]
        with patch.object(module,'request_suggestions',return_value=bad) as writer:
            response=self.client.post('/api/lyric-desk/suggestions',headers=self.headers,json=payload)
        self.assertEqual(writer.call_count,2);self.assertEqual(response.status_code,200,response.json)
        self.assertEqual([r['direction'] for r in response.json['suggestions']],['Close & personal','Setup & payoff'])

    def test_seed_and_locks_are_saved_and_noop_saves_do_not_upload_again(self):
        payload={**self.payload,'hook_seed':'My own light','hook_lines':2,'hook_target':'append',
                 'locks':[{'id':'light','line':1,'start':11,'end':16,'text':'light'}]}
        created=self.client.post('/api/lyric-desk/notebooks',headers=self.headers,json=payload).json['notebook']
        self.assertEqual(created['hook_seed'],'My own light');self.assertEqual(created['hook_lines'],2)
        self.assertEqual(created['history'][0]['locks'],payload['locks'])
        with patch.object(module.cloud_storage,'queue_file') as upload:
            same=self.client.put('/api/lyric-desk/notebooks/'+created['id'],headers=self.headers,json=created).json['notebook']
        self.assertEqual(same['revision'],created['revision']);upload.assert_not_called()
