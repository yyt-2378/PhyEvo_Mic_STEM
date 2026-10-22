"""Public platform isolation, consent, durable review and real-engine contract checks."""
import io
import json
from pathlib import Path
import sys
import time

import numpy as np
import pytest
pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tfree_detection.platform_api import create_app, decode_image
from tfree_detection.platform_store import PlatformStore, digest

ORIGIN = "http://testserver"


def png():
    y, x = np.mgrid[:128, :128]
    arr = np.uint8(np.clip(120 + 55*np.cos(x*.55)*np.cos(y*.6), 0, 255))
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def env(tmp_path):
    app = create_app(tmp_path / "platform", origin=ORIGIN, secure=False)
    store = app.state.store
    users = {name: store.create_user(name, role) for name, role in
             [("A", "researcher"), ("B", "researcher"), ("E", "expert"), ("admin", "admin")]}
    clients = {}
    for name, user in users.items():
        client = TestClient(app)
        client.headers["Origin"] = ORIGIN
        response = client.post("/api/login", json={"code": user["activation_code"]})
        assert response.status_code == 200
        client.headers["X-CSRF-Token"] = response.json()["csrf"]
        clients[name] = client
    yield app, store, users, clients
    for client in clients.values():
        client.close()


def setup_project(clients):
    a = clients["A"]
    p = a.post("/api/projects", json={"name": "Private specimen"}).json()
    r = a.post(f"/api/projects/{p['id']}/samples?name=synthetic.png", content=png(), headers={"Content-Type": "application/octet-stream"})
    assert r.status_code == 200, r.text
    s = r.json()
    j = a.post(f"/api/samples/{s['id']}/jobs", json={"task": "atomic_point_defect"}).json()
    return p, s, j


def test_private_by_default_and_explicit_expert_sharing(env):
    _, store, users, c = env
    p, s, j = setup_project(c)
    paths = [f"/api/projects/{p['id']}", f"/api/projects/{p['id']}/export",
             f"/api/projects/{p['id']}/research", f"/api/samples/{s['id']}/preview",
             f"/api/jobs/{j['id']}", f"/api/jobs/{j['id']}/artifacts/summary.json"]
    for name in ["B", "E", "admin"]:
        assert c[name].get("/api/projects").json() == []
        for path in paths:
            assert c[name].get(path).status_code == 404
    assert c["B"].post(f"/api/projects/{p['id']}/share", json={"expert_id": users["E"]["id"], "active": True}).status_code == 404
    grant = {"expert_id": users["E"]["id"], "active": True}
    assert c["A"].post(f"/api/projects/{p['id']}/share", json=grant).status_code == 200
    assert c["E"].get(f"/api/samples/{s['id']}/preview").status_code == 200
    assert c["E"].get(f"/api/jobs/{j['id']}").status_code == 200
    assert c["E"].post(f"/api/samples/{s['id']}/jobs", json={"task": "amorphous"}).status_code == 404
    assert c["E"].get(f"/api/projects/{p['id']}/export").status_code == 404
    c["A"].post(f"/api/projects/{p['id']}/share", json={**grant, "active": False})
    assert c["E"].get(f"/api/samples/{s['id']}/preview").status_code == 404
    assert c["E"].get("/api/projects").json() == []


def test_invalid_login_never_reaches_project_or_image_access(env, monkeypatch):
    app, store, _, _ = env
    def prohibited(*args, **kwargs):
        raise AssertionError("Private assets accessed before authentication")
    monkeypatch.setattr(store, "project", prohibited)
    monkeypatch.setattr(store, "sample_path", prohibited)
    client = TestClient(app, headers={"Origin": ORIGIN})
    assert client.post("/api/login", json={"code": "wrong"}).status_code == 401
    assert client.get("/api/projects").status_code == 401
    assert client.get("/api/samples/unknown/preview").status_code == 401
    assert client.get("/").status_code == 200


def test_csrf_cookie_revocation_and_admin_control(env):
    app, store, users, c = env
    assert c["A"].post("/api/projects", json={"name": "x"}, headers={"Origin": "https://attacker.invalid"}).status_code == 403
    assert c["A"].post("/api/projects", json={"name": "x"}, headers={"X-CSRF-Token": "wrong"}).status_code == 403
    assert c["A"].post("/api/users", json={"name": "intruder", "role": "admin"}).status_code == 403
    assert c["admin"].post("/api/users", json={"name": "E2", "role": "expert"}).status_code == 200
    assert c["admin"].post(f"/api/users/{users['B']['id']}/disable").status_code == 200
    assert c["B"].get("/api/me").status_code == 401
    assert c["B"].post("/api/login", json={"code": users["B"]["activation_code"]}).status_code == 401
    assert "activation_code" not in c["admin"].get("/api/users").text
    assert users["A"]["activation_code"] not in store.path.read_bytes().decode("utf8", "ignore")


def test_upload_validation_and_job_budget(env):
    _, _, _, c = env
    p, s, _ = setup_project(c)
    for name, content in [("bad.png", b"not an image"), ("../bad.png", png())]:
        assert c["A"].post(f"/api/projects/{p['id']}/samples", params={"name": name}, content=content).status_code == 422
    assert c["A"].post(f"/api/samples/{s['id']}/jobs", json={"task": "shell_exec"}).status_code == 422
    for _ in range(2):
        assert c["A"].post(f"/api/samples/{s['id']}/jobs", json={"task": "amorphous"}).status_code == 200
    assert c["A"].post(f"/api/samples/{s['id']}/jobs", json={"task": "amorphous"}).status_code == 409


def test_lost_code_renewal_keeps_identity_projects_and_history(env):
    _, store, users, c = env
    p, _, _ = setup_project(c)
    url = f"/api/users/{users['A']['id']}/renew"
    assert c["B"].post(url).status_code == 403
    r = c["admin"].post(url)
    assert r.status_code == 200
    assert r.json()["id"] == users["A"]["id"]
    assert c["A"].get("/api/me").status_code == 401
    assert c["A"].post("/api/login", json={"code": users["A"]["activation_code"]}).status_code == 401
    assert c["A"].post("/api/login", json={"code": r.json()["activation_code"]}).status_code == 200
    assert c["A"].get("/api/projects").json()[0]["id"] == p["id"]


def test_stack_color_and_oversized_image_rejected():
    for mode, size, options in [("RGB", (64,64), {}), ("L", (32,32), {}), ("L", (64,64), {"save_all": True, "append_images": [Image.new("L", (64,64))]})]:
        buf = io.BytesIO()
        Image.new(mode, size, color=(255,0,0) if mode == "RGB" else 0).save(buf, format="TIFF", **options)
        with pytest.raises(ValueError):
            decode_image(buf.getvalue())


def test_grayscale_rgb_content_is_not_silently_rejected_or_color_mixed():
    buf = io.BytesIO()
    Image.open(io.BytesIO(png())).convert("RGB").save(buf, format="PNG")
    array, _, _ = decode_image(buf.getvalue())
    original, _, _ = decode_image(png())
    assert np.array_equal(array, original)


def test_review_revisions_idempotency_and_reopen(env):
    _, store, users, c = env
    p,s,j = setup_project(c)
    store.execute("UPDATE jobs SET status='completed' WHERE id=?", (j["id"],))
    payload = {"action": "correct", "note": "synthetic validation", "geometry": [{"type": "point", "points": [[21.5, 38.25]]}], "prediction_seen": False, "request_id": "unique-request-0001"}
    url = f"/api/jobs/{j['id']}/feedback"
    for _ in range(2):
        response = c["A"].post(url, json=payload)
        assert response.status_code == 200, response.text
        assert response.json()["revision"] == 1
    assert c["A"].post(url, json={**payload, "note": "changed"}).status_code == 409
    assert c["A"].post(url, json={**payload, "request_id": "unique-request-0002"}).json()["revision"] == 2
    assert c["B"].post(url, json=payload).status_code == 404
    c["A"].post(f"/api/projects/{p['id']}/share", json={"expert_id": users["E"]["id"], "active": True})
    assert c["E"].post(url, json=payload).json()["revision"] == 1
    reopened = PlatformStore(store.root)
    assert len(reopened.all("SELECT * FROM feedback")) == 3
    assert len(reopened.all("SELECT * FROM feedback WHERE user_id=?", (users["A"]["id"],))) == 2
    exported = c["A"].get(f"/api/projects/{p['id']}/export").json()
    assert len(exported["revisions"]) == 3
    c["A"].post(f"/api/projects/{p['id']}/share", json={"expert_id": users["E"]["id"], "active": False})
    assert c["E"].post(url, json={**payload, "request_id": "unique-request-0003"}).status_code == 404
    assert len(reopened.all("SELECT * FROM feedback")) == 3


@pytest.mark.parametrize("change", [
    {"action": "accept", "geometry": []}, {"action": "correct", "geometry": []},
    {"geometry": [{"type": "point", "points": [[128, 2]]}]},
    {"geometry": [{"type": "line", "points": [[2,3]]}]},
])
def test_feedback_semantics_reject_invalid(env, change):
    _, store, _, c = env
    _,_,j = setup_project(c)
    store.execute("UPDATE jobs SET status='completed' WHERE id=?", (j["id"],))
    payload = {"action": "correct", "geometry": [{"type": "point", "points": [[10,20]]}], "request_id": "validation-00001", **change}
    assert c["A"].post(f"/api/jobs/{j['id']}/feedback", json=payload).status_code == 422


def test_research_requires_owner_consent_and_is_not_fake_execution(env):
    _, _, _, c = env
    p,_,_ = setup_project(c)
    url = f"/api/projects/{p['id']}/research"
    assert c["A"].post(url, json={"reason": "missed interface"}).status_code == 422
    assert c["B"].post(url, json={"reason": "missed interface", "consent": True}).status_code == 404
    r = c["A"].post(url, json={"reason": "missed interface", "consent": True})
    assert r.json()["executed"] is False
    assert c["A"].get(url).json()[0]["status"] == "requested"


@pytest.mark.parametrize("task", ["atomic_point_defect", "amorphous", "stacking_fault", "phase_boundary", "symmetry_relation"])
def test_real_engine_output_and_queue_survive_reopen(env, task):
    from tfree_detection.platform_worker import execute_job
    _, store, _, c = env
    _,s,j = setup_project(c)
    store.execute("UPDATE jobs SET task=? WHERE id=?", (task, j["id"]))
    reopened = PlatformStore(store.root)
    claimed = reopened.claim_job()
    assert claimed["id"] == j["id"]
    assert reopened.claim_job() is None
    execute_job(store.root, j["id"])
    reopened.execute("UPDATE jobs SET status='completed',finished=? WHERE id=?", (time.time(), j["id"]))
    r = c["A"].get(f"/api/jobs/{j['id']}")
    assert r.status_code == 200
    assert r.json()["summary"]["image_shape"] == [128,128]
    assert r.json()["summary"]["platform_provenance"]["original_sha256"] == s["sha256"]
    for artifact in ["outline.png", "heatmap.png", "summary.json"]:
        assert c["A"].get(f"/api/jobs/{j['id']}/artifacts/{artifact}").status_code == 200
        assert c["B"].get(f"/api/jobs/{j['id']}/artifacts/{artifact}").status_code == 404
    assert Image.open(store.job_path(j["id"])/"outline.png").size == (128,128)
    assert c["A"].get(f"/api/jobs/{j['id']}/artifacts/platform.sqlite3").status_code == 404


def test_secure_cookie_and_security_headers(tmp_path):
    app = create_app(tmp_path / "secure", origin="https://testserver", secure=True)
    u = app.state.store.create_user("safe")
    c = TestClient(app, base_url="https://testserver", headers={"Origin": "https://testserver"})
    r = c.post("/api/login", json={"code": u["activation_code"]})
    header = r.headers["set-cookie"]
    assert "Secure" in header and "HttpOnly" in header and "SameSite=strict" in header
    assert "__Host-phyevo_session" in header
    assert c.get("/api/me").status_code == 200
    assert "frame-ancestors 'none'" in r.headers["content-security-policy"]
    assert r.headers["cache-control"] == "no-store"
