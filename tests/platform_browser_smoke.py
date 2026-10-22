"""Optional real-browser smoke test. Fresh synthetic data only; no production access.

Requires an existing Chromium headless shell and websocket-client / requests.
Pass a NEW workspace-local --run directory. Retains evidence, stops only its own PIDs.
"""
from __future__ import annotations
import argparse
import base64
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from PIL import Image
import requests
import websocket

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from tfree_detection.platform_store import PlatformStore


class Browser:
    def __init__(self, port):
        session = requests.Session()
        session.trust_env = False
        end = time.monotonic() + 15
        while True:
            try:
                pages = session.get(f"http://127.0.0.1:{port}/json/list").json()
                page = next(p for p in pages if p["type"] == "page")
                break
            except (requests.RequestException, StopIteration):
                if time.monotonic() > end:
                    raise RuntimeError("Browser did not start; inspect browser.log")
                time.sleep(.2)
        self.ws = websocket.create_connection(page["webSocketDebuggerUrl"], suppress_origin=True, timeout=20)
        self.seq = 0

    def call(self, method, **params):
        self.seq += 1
        self.ws.send(json.dumps({"id": self.seq, "method": method, "params": params}))
        while True:
            response = json.loads(self.ws.recv())
            if response.get("id") == self.seq:
                if "error" in response:
                    raise RuntimeError(response["error"])
                return response.get("result", {})

    def js(self, expression):
        r = self.call("Runtime.evaluate", expression=expression, returnByValue=True, awaitPromise=True)
        if "exceptionDetails" in r:
            raise RuntimeError(r["exceptionDetails"])
        return r.get("result", {}).get("value")

    def wait(self, expression, seconds=30):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            if self.js(expression):
                return
            time.sleep(.2)
        raise AssertionError("Timed out: " + expression)

    def shot(self, path):
        r = self.call("Page.captureScreenshot", format="png", captureBeyondViewport=False)
        with path.open("xb") as stream:
            stream.write(base64.b64decode(r["data"]))

    def click(self, selector):
        self.js(f"document.querySelector({json.dumps(selector)}).click()")


def check_evolution(b, store, run):
    """Exercise the historical replay without creating jobs or research requests."""
    assert b.js("document.querySelectorAll('[data-evo-stage]').length") == 6
    assert b.js("document.querySelector('.evo-history-label').textContent.includes('历史演进回放')")
    try:
        b.wait("document.querySelector('.evolution').dataset.stage === '1'", seconds=12)
    except AssertionError:
        b.shot(run/'evolution-playback-diagnostic.png')
        print(b.js("({visibility:document.visibilityState,reduced:matchMedia('(prefers-reduced-motion: reduce)').matches,replay:{...document.querySelector('.evolution').dataset},errors:window.__browserErrors})"), flush=True)
        raise
    b.click('[data-evo-toggle]')
    paused = b.js("document.querySelector('.evolution').dataset.progress")
    time.sleep(.3)
    assert b.js("document.querySelector('.evolution').dataset.progress") == paused
    b.click('[data-evo-stage="4"]')
    assert b.js("document.querySelector('[data-evo-stage=\"4\"]').getAttribute('aria-pressed')") == 'true'
    assert b.js("document.querySelector('.evo-stage-kept').textContent") == '分析分支与采用规则'
    assert b.js("document.documentElement.scrollWidth <= window.innerWidth")
    time.sleep(.7)  # Let node-color transitions finish before the static preview.
    b.shot(run/'05-evolution-desktop.png')
    b.click('[data-evo-replay]')
    assert b.js("document.querySelector('.evolution').dataset.stage") == '0'
    assert b.js("document.querySelector('.evolution').dataset.playing") == 'true'
    b.wait("document.querySelector('.evolution').dataset.progress === '1.0000' && document.querySelector('.evolution').dataset.playing === 'false'", seconds=45)
    assert b.js("document.querySelector('.evolution').dataset.playing") == 'false'
    assert b.js("document.querySelector('.evo-play-status').textContent") == '回放结束'
    b.click('#explore-evolution')
    b.wait("!!document.querySelector('.evolution-expanded')")
    assert b.js("document.querySelectorAll('.batch-stage').length") == 4
    assert b.js("document.querySelector('.evolution').dataset.playing") == 'false'
    assert b.js("document.querySelector('#request').disabled") is True
    b.click('[data-evo-stage="5"]')
    time.sleep(.7)
    b.shot(run/'06-evolution-route.png')
    b.call('Emulation.setEmulatedMedia', features=[{'name':'prefers-reduced-motion','value':'reduce'}])
    b.click('[data-nav=home]')
    b.wait("!!document.querySelector('#new-project')")
    assert b.js("document.querySelector('.evolution').dataset.playing") == 'false'
    assert b.js("getComputedStyle(document.querySelector('.evo-core')).animationName") == 'none'
    b.click('[data-evo-stage="2"]')
    b.click('[data-evo-replay]')
    assert b.js("document.querySelector('.evolution').dataset.stage") == '0'
    assert b.js("document.querySelector('.evolution').dataset.playing") == 'false'
    for width in [390, 320]:
        b.call('Emulation.setDeviceMetricsOverride',width=width,height=844,deviceScaleFactor=1,mobile=True)
        assert b.js('document.documentElement.scrollWidth <= window.innerWidth'), f'Replay overflow at {width}px'
        if width == 390:
            b.shot(run/'07-evolution-mobile.png')
    b.click('[data-nav=rsi]')
    b.wait("!!document.querySelector('.batch-cycle')")
    assert b.js('document.documentElement.scrollWidth <= window.innerWidth'), 'RSI route overflow at 320px'
    b.call('Emulation.setEmulatedMedia', features=[])
    b.call('Emulation.setDeviceMetricsOverride',width=1440,height=1000,deviceScaleFactor=1,mobile=False)
    b.click('[data-nav=home]')
    b.wait("!!document.querySelector('#new-project')")
    assert b.js("document.querySelector('.evolution').dataset.playing") == 'false'
    for table in ['projects', 'jobs', 'research_requests', 'feedback']:
        assert store.one(f'SELECT COUNT(*) AS n FROM {table}')['n'] == 0
    return {'stages':6,'pause_replay_and_finite_playback':True,'reduced_motion':True,
            'mobile_widths':[390,320],'replay_created_jobs_or_requests':False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--browser", type=Path, required=True)
    a = p.parse_args()
    run = a.run.resolve()
    if not run.is_relative_to(REPO) or run.exists():
        p.error("--run must be a NEW real workspace subdirectory")
    run.mkdir(parents=True)
    root = run / "data"
    store = PlatformStore(root)
    user = store.create_user("界面验证用户", "admin")
    y,x = np.mgrid[:512,:512]
    arr = 105 + 55*np.cos(x*.32)*np.cos(y*.35) + 15*np.cos(x*.8+y*.15)
    arr[220:260,220:290] = 115
    Image.fromarray(np.uint8(np.clip(arr,0,255))).save(run / "synthetic.png")
    processes, logs = [], []
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", MPLCONFIGDIR=str(run/"mpl"), TMPDIR=str(run), XDG_CACHE_HOME=str(run/"cache"), XDG_CONFIG_HOME=str(run/"config"))
    def launch(command, logname):
        log = (run/logname).open("x")
        logs.append(log)
        child = subprocess.Popen(command, cwd=REPO, env=env, stdout=log, stderr=log, stdin=subprocess.DEVNULL)
        processes.append(child)
        return child
    try:
        for command in ["serve", "worker"]:
            launch([sys.executable, str(REPO/"scripts/serve_platform.py"), command, "--root", str(root), "--local", "--origin", "http://127.0.0.1:8601", "--port", "8601"], command+".log")
        session = requests.Session(); session.trust_env = False
        for _ in range(50):
            try:
                if session.get("http://127.0.0.1:8601/healthz",timeout=1).ok:
                    break
            except requests.RequestException:
                pass
            time.sleep(.2)
        launch([str(a.browser.resolve()), "--remote-debugging-port=9226", "--remote-debugging-address=127.0.0.1", "--user-data-dir="+str(run/"browser-profile"), "--window-size=1440,1000", "about:blank"], "browser.log")
        b = Browser(9226)
        b.call('Page.enable')
        b.call('Page.addScriptToEvaluateOnNewDocument', source="window.__browserErrors=[];addEventListener('error',e=>window.__browserErrors.push(e.message));addEventListener('unhandledrejection',e=>window.__browserErrors.push(String(e.reason))); ")
        b.call("Emulation.setDeviceMetricsOverride", width=1440,height=1000,deviceScaleFactor=1,mobile=False)
        b.call("Page.navigate", url="http://127.0.0.1:8601/")
        b.wait("!!document.querySelector('#code')")
        b.shot(run/"01-login.png")
        b.js(f"document.querySelector('#code').value={json.dumps(user['activation_code'])}")
        b.js("document.querySelector('#login-form').requestSubmit()")
        b.wait("!!document.querySelector('#new-project')")
        evolution = check_evolution(b, store, run)
        assert b.js('window.__browserErrors') == []
        b.click("#new-project")
        b.js("document.querySelector('#project-name').value='合成晶格 · 端到端验证'")
        b.js("document.querySelector('#create-project').requestSubmit()")
        b.wait("!!document.querySelector('#upload')")
        b.click("#upload")
        doc = b.call("DOM.getDocument")
        node = b.call("DOM.querySelector", nodeId=doc["root"]["nodeId"], selector="#file")
        b.call("DOM.setFileInputFiles", nodeId=node["nodeId"], files=[str(run/"synthetic.png")])
        b.js("document.querySelector('#upload-form').requestSubmit()")
        b.wait("!!document.querySelector('#run')")
        b.click("#run")
        b.wait("!!document.querySelector('#save')", seconds=100)
        assert b.js("document.querySelector('#prediction').checked") is False
        b.click("[data-tool=point]")
        position = b.js("(()=>{const r=document.querySelector('#canvas').getBoundingClientRect();return {x:r.x+r.width/2,y:r.y+r.height/2};})()")
        b.call("Input.dispatchMouseEvent",type="mousePressed",button="left",clickCount=1,**position)
        b.call("Input.dispatchMouseEvent",type="mouseReleased",button="left",clickCount=1,**position)
        b.js("document.querySelector('#note').value='浏览器真实操作验证：中心点已标记。'")
        b.click("#save")
        b.wait("document.querySelector('#save-state')?.textContent.includes('第 1 版')")
        b.click("#prediction")
        b.shot(run/"02-workbench-desktop.png")
        record = store.one("SELECT * FROM feedback")
        geometry = json.loads(record["geometry"])
        assert abs(geometry[0]["points"][0][0] - 256) < 1
        assert abs(geometry[0]["points"][0][1] - 256) < 1
        b.call("Page.reload")
        b.wait("document.querySelector('#save-state')?.textContent.includes('第 1 版')")
        b.call("Emulation.setDeviceMetricsOverride",width=390,height=844,deviceScaleFactor=1,mobile=True)
        time.sleep(.4)
        assert b.js("document.documentElement.scrollWidth <= window.innerWidth"), "Mobile horizontal overflow"
        b.shot(run/"03-mobile.png")
        b.call("Emulation.setDeviceMetricsOverride",width=1440,height=1000,deviceScaleFactor=1,mobile=False)
        b.click("[data-nav=accounts]")
        b.wait("!!document.querySelector('#invite')")
        b.click("#invite")
        b.js("document.querySelector('#invite-name').value='受邀专家'")
        b.js("document.querySelector('#invite-role').value='expert'")
        b.js("document.querySelector('#invite-form').requestSubmit()")
        b.wait("!!document.querySelector('#new-code')")
        assert b.js("document.querySelector('#new-code').textContent.startsWith('PEM.')")
        expert_code = b.js("document.querySelector('#new-code').textContent")
        # Never screenshot activation codes.
        b.click("#code-done")
        b.wait("!!document.querySelector('#invite')")
        expert = requests.Session(); expert.trust_env=False
        expert.headers['Origin']='http://127.0.0.1:8601'
        login=expert.post('http://127.0.0.1:8601/api/login',json={'code':expert_code}).json()
        expert.headers['X-CSRF-Token']=login['csrf']
        pid=b.js('state.project.id'); jid=b.js('state.job.id')
        assert expert.get('http://127.0.0.1:8601/api/jobs/'+jid).status_code==404
        b.js('openProject(state.project.id,state.sample.id)')
        b.wait("!!document.querySelector('#share')")
        b.click('#share')
        b.wait("!!document.querySelector('[data-expert]')")
        b.click('[data-expert]')
        b.wait("!!document.querySelector('[data-expert][data-active=\"0\"]')")
        b.click('#modal [data-close]')
        result=expert.post('http://127.0.0.1:8601/api/jobs/'+jid+'/feedback',json={
            'action':'correct','geometry':[{'type':'point','points':[[100,150]]}],
            'note':'受邀专家独立修正','request_id':'browser-expert-review-001'})
        assert result.status_code==200,result.text
        b.call('Page.reload')
        b.wait("!!document.querySelector('#annotation-source')")
        b.js("document.querySelector('#annotation-source').selectedIndex=1; document.querySelector('#annotation-source').dispatchEvent(new Event('change'))")
        assert b.js('canvasState.geometry[0].points[0]')==[256,256]
        b.shot(run/'04-expert-shared-review.png')
        b.call('Emulation.setDeviceMetricsOverride',width=390,height=844,deviceScaleFactor=1,mobile=True)
        assert b.js('document.documentElement.scrollWidth <= window.innerWidth')
        assert b.js('window.__browserErrors') == []
        report={"browser_flow":"login/evolution/create/upload/real_engine/draw/save/reload/invite/share/expert_review/owner_view", "evolution":evolution, "mobile_no_horizontal_overflow":True, "raw_coordinates":geometry[0]["points"][0], "independent_expert_geometry":[100,150], "data_root":"synthetic-only", "processes_started":[p.pid for p in processes]}
        with (run/"validation.json").open("x") as stream:
            json.dump(report,stream,ensure_ascii=False,indent=2)
        print(json.dumps(report,ensure_ascii=False),flush=True)
        b.ws.close()
    finally:
        for child in processes:
            if child.poll() is None:
                child.terminate()
        for child in processes:
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill(); child.wait()
        for log in logs:
            log.close()


if __name__ == "__main__":
    main()
