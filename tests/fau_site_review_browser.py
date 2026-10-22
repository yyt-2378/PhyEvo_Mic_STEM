"""Real browser QA in a NEW project-local directory; production data read-only.

Only synthetic review opinions enter the isolated test ledger. No live saves.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def serve():
    import streamlit as st
    sys.path.insert(0,str(ROOT/'apps'))
    from review_dataset_viewer import _require_access_code
    from tfree_detection.fau_site_review import render
    st.set_page_config(layout='wide')
    _require_access_code();render()


def main():
    from platform_browser_smoke import Browser
    from tfree_detection.fau_site_review import SiteLedger, MODEL
    import requests
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--browser',type=Path,required=True)
    parser.add_argument('--feedback',action='store_true',help='Also exercise the local feedback release without production writes')
    parser.add_argument('--feedback-edition',choices=('feedback-20261022','feedback-20261022-v2','feedback-20261022-v3'),default='feedback-20261022')
    args=parser.parse_args();run=args.run.resolve();binary=args.browser.resolve()
    if not run.is_relative_to(ROOT) or run.exists() or not binary.is_relative_to(ROOT):
        raise ValueError('Use NEW project-local run and existing project-local browser')
    run.mkdir(parents=True);children=[];logs=[]
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(ROOT/'src'),TFREE_REVIEW_ACCESS_CODE='fau-local-browser-only',
             TFREE_FAU_REVIEW_STORE_PATH=str(run/'reviews.sqlite3'),TMPDIR=str(run),XDG_CACHE_HOME=str(run/'cache'),XDG_CONFIG_HOME=str(run/'config'))
    def launch(cmd,log):
        stream=(run/log).open('x');logs.append(stream)
        child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=stream)
        children.append(child);return child
    b=None
    try:
        launch([sys.executable,'-B','-m','streamlit','run',str(Path(__file__).resolve()),'--server.address=127.0.0.1',
                '--server.port=8687','--server.headless=true','--browser.gatherUsageStats=false','--server.fileWatcherType=none','--','--serve'],'server.log')
        session=requests.Session();session.trust_env=False
        for _ in range(60):
            try:
                if session.get('http://127.0.0.1:8687/_stcore/health',timeout=1).ok:break
            except requests.RequestException:pass
            time.sleep(.2)
        launch([str(binary),'--remote-debugging-port=9337','--remote-debugging-address=127.0.0.1',
                '--user-data-dir='+str(run/'browser-profile'),'--window-size=1550,1050','about:blank'],'browser.log')
        b=Browser(9337);b.call('Page.enable')
        b.call('Page.addScriptToEvaluateOnNewDocument',source="window.__qaErrors=[];addEventListener('error',e=>window.__qaErrors.push(e.message));addEventListener('unhandledrejection',e=>window.__qaErrors.push(String(e.reason)));")
        b.call('Emulation.setDeviceMetricsOverride',width=1550,height=1050,deviceScaleFactor=1,mobile=False)
        b.call('Page.navigate',url='http://127.0.0.1:8687/?view=fau-sites&reviewer=browser-qa&edition=original')
        def input_react(selector,value):
            b.js(f"(()=>{{const el=document.querySelector({json.dumps(selector)});Object.getOwnPropertyDescriptor(HTMLInputElement.prototype,'value').set.call(el,{json.dumps(value)});el.dispatchEvent(new Event('input',{{bubbles:true}}));}})()")
        def login(code):
            b.wait("!!document.querySelector('input[type=password]')")
            input_react('input[type=password]',code)
            b.js("[...document.querySelectorAll('button')].find(b=>b.textContent==='进入审核').click()")
        login('wrong')
        b.wait("document.body.innerText.includes('访问码错误')")
        assert not (run/'reviews.sqlite3').exists()
        login('fau-local-browser-only')
        b.wait("document.querySelectorAll('#fau-picker option').length === 55",seconds=60)
        assert b.js("document.querySelector('#fau-model').checked")
        assert b.js("document.querySelectorAll('#fau-picker option').length")==55
        ledger=SiteLedger(run/'reviews.sqlite3')
        with ledger.connect() as con:assert con.execute('SELECT COUNT(*) FROM fau_events').fetchone()[0]==0
        # Existing model labels are editable immediately; no re-drawing or mandatory alignment.
        b.click('#fau-next-yellow')
        yellow_id=b.js("document.querySelector('#fau-picker').value")
        assert b.js("document.querySelector('#fau-current').dataset.decision")=='uncertain'
        assert b.js("document.querySelector('#fau-current').dataset.origin")=='model'
        assert b.js("document.querySelector('#fau-alignment').value")=='not_reviewed'
        b.shot(run/'01-desktop.png')
        b.click('button[data-decision=occupied]')
        assert b.js("document.querySelector('#fau-current').dataset.origin")=='draft'
        assert b.js("document.querySelector('#fau-color-counts').textContent.includes('含未保存预览')")
        b.click('#fau-save-stay')
        b.wait("document.querySelector('#fau-receipt')?.textContent.includes('第 1 版')",seconds=45)
        b.wait("document.querySelector('#fau-current')?.dataset.origin === 'human'")
        assert b.js("document.querySelector('#fau-picker').value")==yellow_id
        assert b.js("document.querySelector('#fau-current').dataset.decision")=='occupied'
        with ledger.connect() as con:first=json.loads(con.execute('SELECT payload FROM fau_events').fetchone()[0])
        assert first['alignment']=='not_reviewed' and first['decision']=='occupied' and first['model_status']=='uncertain'
        assert first['center']==first['model_center']
        b.shot(run/'02-yellow-to-red.png')
        b.click('button[data-decision=empty_like]')
        b.js("document.querySelector('#fau-alignment').value='bad';document.querySelector('#fau-alignment').dispatchEvent(new Event('change',{bubbles:true}));document.querySelector('#fau-note').value='隔离浏览器测试：不是生产审核';document.querySelector('#fau-note').dispatchEvent(new Event('input',{bubbles:true}));")
        b.click('#fau-recenter')
        pos=b.js("(()=>{const r=document.querySelector('#fau-detail').getBoundingClientRect();return {x:r.x+r.width*.55,y:r.y+r.height*.53};})()")
        b.call('Input.dispatchMouseEvent',type='mousePressed',button='left',clickCount=1,**pos)
        b.call('Input.dispatchMouseEvent',type='mouseReleased',button='left',clickCount=1,**pos)
        b.click('#fau-save-stay')
        b.wait("document.querySelector('#fau-receipt')?.textContent.includes('第 2 版')",seconds=45)
        b.wait("document.querySelector('#fau-save-stay')?.disabled === false")
        with ledger.connect() as con:payload=json.loads(con.execute('SELECT payload FROM fau_events ORDER BY seq DESC LIMIT 1').fetchone()[0])
        assert payload['decision']=='empty_like' and payload['alignment']=='bad'
        assert payload['center']!=payload['model_center']
        assert payload['revision']==2 and payload['model_status']=='uncertain'
        b.shot(run/'02-saved.png')
        # Reload starts a new Streamlit session: authenticate, then read saved progress.
        b.call('Page.navigate',url='http://127.0.0.1:8687/?view=fau-sites&reviewer=browser-qa&edition=original')
        # Page.navigate is asynchronous; do not mistake the preceding page's DOM for the new session.
        login('fau-local-browser-only')
        b.wait("document.querySelectorAll('#fau-picker option').length === 55",seconds=45)
        assert b.js("document.body.innerText.includes('1 / 101')")
        b.js(f"document.querySelector('#fau-picker').value={json.dumps(payload['site_id'])};document.querySelector('#fau-picker').dispatchEvent(new Event('change',{{bubbles:true}}));")
        b.wait("document.querySelector('button[data-decision=empty_like]').getAttribute('aria-pressed') === 'true'")
        assert b.js("document.querySelector('#fau-current').dataset.origin")=='human'
        assert b.js("document.querySelector('#fau-current').textContent.includes('原模型：黄色')")
        assert b.js("document.querySelector('#fau-note').value")=='隔离浏览器测试：不是生产审核'
        b.click('[data-sample=H2]')
        assert b.js("document.querySelectorAll('#fau-picker option').length")==46
        # Clicking a nearby existing ring in the detail image selects it, not a new point.
        rows=json.loads(MODEL.read_text())
        selected_id=b.js("document.querySelector('#fau-picker').value")
        selected=next(r for r in rows if r['id']==selected_id)
        nearby=sorted([r for r in rows if r['sample']=='H2' and r['id']!=selected_id],key=lambda r:(r['x']-selected['x'])**2+(r['y']-selected['y'])**2)[0]
        b.click('#fau-minus');b.click('#fau-minus');b.click('#fau-minus')
        point=json.dumps([nearby['x']-selected['x'],nearby['y']-selected['y']])
        pos=b.js(f"(()=>{{const r=document.querySelector('#fau-detail').getBoundingClientRect(),d={point},k=Math.min(r.width,r.height)/(2*37*1.35**3);return {{x:r.x+r.width/2+d[0]*k,y:r.y+r.height/2+d[1]*k}};}})()")
        b.call('Input.dispatchMouseEvent',type='mousePressed',button='left',clickCount=1,**pos)
        b.call('Input.dispatchMouseEvent',type='mouseReleased',button='left',clickCount=1,**pos)
        assert b.js("document.querySelector('#fau-picker').value")==nearby['id']
        assert b.js("document.querySelectorAll('#fau-picker option').length")==46
        b.js("document.querySelector('#fau-add').closest('details').open=true")
        b.click('#fau-add')
        b.js("document.querySelector('#fau-map').addEventListener('click',e=>{const c=e.currentTarget,r=c.getBoundingClientRect();window.__qaMapPoint=[(e.clientX-r.left)*c.width/r.width,(e.clientY-r.top)*c.height/r.height];},{once:true})")
        pos=b.js("(()=>{const r=document.querySelector('#fau-map').getBoundingClientRect();return {x:r.x+r.width*.5,y:r.y+r.height*.5};})()")
        b.call('Input.dispatchMouseEvent',type='mousePressed',button='left',clickCount=1,**pos)
        b.call('Input.dispatchMouseEvent',type='mouseReleased',button='left',clickCount=1,**pos)
        b.wait("document.querySelector('#fau-site-title')?.textContent.includes('手工补点')")
        b.click('button[data-decision=uncertain]')
        b.js("document.querySelector('#fau-alignment').value='uncertain';document.querySelector('#fau-alignment').dispatchEvent(new Event('change',{bubbles:true}));")
        b.click('#fau-save-stay')
        b.wait("document.querySelector('#fau-receipt')?.textContent.includes('H2-M')",seconds=45)
        with ledger.connect() as con:events=[json.loads(row[0]) for row in con.execute('SELECT payload FROM fau_events ORDER BY seq')]
        assert len(events)==3 and events[2]['manual'] and events[2]['sample']=='H2'
        expected=b.js('window.__qaMapPoint')
        assert all(abs(a-b)<1e-6 for a,b in zip(events[2]['center'],expected))
        assert abs(expected[0]-306)<4 and abs(expected[1]-298.5)<4  # one screen pixel at overview zoom
        if args.feedback:
            b.call('Page.navigate',url='http://127.0.0.1:8687/?view=fau-sites&reviewer=browser-qa&edition='+args.feedback_edition)
            login('fau-local-browser-only')
            b.wait("document.querySelectorAll('#fau-picker option').length === 55",seconds=45)
            assert b.js("document.body.innerText.includes('1 / 101')")
            b.js(f"document.querySelector('#fau-picker').value={json.dumps(payload['site_id'])};document.querySelector('#fau-picker').dispatchEvent(new Event('change',{{bubbles:true}}));")
            b.wait("document.querySelector('button[data-decision=empty_like]').getAttribute('aria-pressed') === 'true'")
            assert b.js("document.querySelector('#fau-current').dataset.origin")=='human'
            assert b.js("document.querySelector('#fau-current').textContent.includes('本版学习模型')")
            b.click('button[data-decision=occupied]');b.click('#fau-save-stay')
            b.wait("document.querySelector('#fau-receipt')?.textContent.includes('第 3 版')",seconds=45)
            with ledger.connect() as con:events=[json.loads(row[0]) for row in con.execute('SELECT payload FROM fau_events ORDER BY seq')]
            assert len(events)==4 and events[-1]['model_edition']==args.feedback_edition
            assert events[-1]['model_release']!=events[-1]['release']
            assert events[-1]['release']==events[0]['release'] and events[-1]['center']==payload['center']
            b.shot(run/'04-feedback-version.png')
        for width in (390,320):
            b.call('Emulation.setDeviceMetricsOverride',width=width,height=844,deviceScaleFactor=1,mobile=True)
            time.sleep(.4)
            assert b.js('document.documentElement.scrollWidth<=innerWidth'),width
            if width==390:b.shot(run/'03-mobile.png')
        errors=b.js('window.__qaErrors');assert errors==[],errors
        report=dict(status='passed',checks=['wrong-code-no-data','model-colors-visible-by-default','no-auto-confirmation','yellow-to-red-no-redraw-no-alignment-required','red-to-blue-revision-history','native-coordinate-correction','persistent-save','reload-resume','click-existing-ring-in-detail','manual-add','mobile-390-320-no-overflow'],test_reviews=len(events),production_ledger_touched=False,pids=[p.pid for p in children])
        if args.feedback:report['checks'].extend(['feedback-version-preserves-progress','feedback-save-records-display-model','human-correction-priority-across-versions'])
        with (run/'validation.json').open('x') as stream:json.dump(report,stream,ensure_ascii=False,indent=2)
        print(json.dumps(report,ensure_ascii=False),flush=True)
    except Exception:
        if b:
            b.shot(run/'failure.png')
            print(b.js('document.body.innerText.slice(-5000)'),flush=True)
        raise
    finally:
        if b:b.ws.close()
        for child in children:
            if child.poll() is None:child.terminate()
        for child in children:
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:child.kill();child.wait()
        for stream in logs:stream.close()


if __name__=='__main__':
    if '--serve' in sys.argv:serve()
    else:main()
