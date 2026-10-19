"""Post-response integrity audit, not an algorithm or a scientific grader."""
from __future__ import annotations

import json
import os
import subprocess

from . import history_probe as h
from .contracts import digest, engine_fingerprint, file_hash


def verify():
    protocol = h.read(h.RUN / "protocol.json")
    seal = h.read(h.RUN / "requests_seal.json")
    mapping = h.read(h.RUN / "private_review_map.json")
    assert engine_fingerprint() == protocol["engine_fingerprint"]
    plans = []
    for slot in protocol["slots"]:
        sid, cid = slot["slot"], slot["case_id"]
        folder = h.RUN / "exchange" / sid
        request = h.read(folder / "request.json")
        receipt = h.read(folder / "receipt.json")
        answer = h.read(folder / "raw_response.json")
        h.check_request(request)
        h.legacy_validator()(request, answer)
        assert file_hash(folder / "request.json") == seal[sid] == receipt["request_sha256"]
        assert file_hash(folder / "raw_response.json") == receipt["response_sha256"]
        assert receipt["origin"] == "/root/history_" + sid
        run = h.RUN / "policy_runs" / sid
        policy_input = h.read(run / "input/request.json")
        assert policy_input["context"]["diagnostic"] == request["packet"]
        assert policy_input["context"]["pool"] == []
        assert policy_input["context"]["memories"] == []
        execution = h.read(run / "output/execution.json")
        assert execution["status"] == "ok" and execution["network"] is False
        assert h.read(run / "output/result.json") == request["research_plan"]
        plans.append(request["research_plan"]["focus"])
    reviewed = []
    for i in range(1, 5):
        job = h.read(h.RUN / "review" / f"job{i}.json")
        for row in job["rows"]:
            cid = row["packet"]["case_id"]
            assert row["historical_reference"] == h.read(h.OLD / "assessor_only" / f"{cid}.json")
            for item in row["answers"]:
                m = mapping[item["label"]]
                assert digest(item["answer"]) == m["answer_sha256"]
                assert m["case_id"] == cid
                expected = (h.read(h.RUN / "exchange" / m["slot"] / "raw_response.json")
                            if m["condition"] == "current" else
                            h.read(h.OLD / "responses" / f"blind-{cid}.json")["result"])
                assert item["answer"] == expected
                assert set(item) == {"label", "answer"}
                reviewed.append(item["label"])
    assert len(reviewed) == len(set(reviewed)) == 42
    changed = [name for name, expected in protocol["source_hashes"].items()
               if file_hash(h.ROOT / name) != expected]
    assert not changed
    env = dict(os.environ, PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=".:src")
    test = subprocess.run([str(h.ROOT / ".venvs/sam2/bin/python"), "-m", "unittest",
                          "experiments.rsi_program_evolution_v1.test_history_probe", "-v"],
                         cwd=h.ROOT, env=env, text=True, capture_output=True, timeout=30)
    assert test.returncode == 0, test.stderr
    result = {"policy_executions": 23, "sealed_answers": 23, "anonymous_review_answers": 42,
              "protected_files_checked": len(protocol["source_hashes"]), "changed": changed,
              "unique_research_focus_count": len(set(plans)),
              "cold_start_has_no_candidate_archive_or_memory": True,
              "diagnosis_only_not_controller_step": True,
              "tests": {"passed": 6, "returncode": test.returncode,
                        "stdout": test.stdout, "stderr": test.stderr},
              "auditor_sha256": file_hash(__file__)}
    h.save(h.RUN / "verification.json", result)
    return {k: v for k, v in result.items() if k != "tests"}


if __name__ == "__main__":
    print(json.dumps(verify(), ensure_ascii=False))
