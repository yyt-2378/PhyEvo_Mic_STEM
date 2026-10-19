"""Transport tests use labelled fixtures, never count as discovery experiments."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import json
import time

import pytest

from experiments.rsi_program_evolution_v1.contracts import file_hash, write_new
from experiments.rsi_program_evolution_v1.provider import (
    FileExchangeModel, ModelUnavailable, proposer_protocol, publish_response)
from experiments.rsi_program_evolution_v1.test_core import protocol, proposal


def wait_request(folder):
    end = time.monotonic() + 5
    path = folder / "call-001/request.json"
    while not path.exists():
        if time.monotonic() >= end:
            raise TimeoutError("No fixture request")
        time.sleep(.01)
    # Writer may still be closing the file; polling doesn't modify evidence.
    while time.monotonic() < end:
        try:
            return json.loads(path.read_text())
        except ValueError:
            time.sleep(.01)
    raise TimeoutError("Incomplete fixture request")


def test_broker_roundtrip_preserves_raw_response_and_distinguishes_model_attestation(tmp_path):
    directory = tmp_path / "exchange"
    model = FileExchangeModel(directory, model="test-fixture", maximum_calls=1, timeout=5)
    raw = json.dumps(asdict(proposal()))
    with ThreadPoolExecutor(max_workers=1) as pool:
        task = pool.submit(model.propose, {"diagnostic": {"coverage": .3}}, protocol())
        request = wait_request(directory)
        assert "audit_groups" not in request["protocol"]
        assert "train_groups" not in request["protocol"]
        publish_response(directory / "call-001", raw_text=raw, model="test-fixture", origin="unit-test")
        value, receipt = task.result(timeout=5)
    assert value == proposal()
    assert receipt["status"] == "ok" and receipt["returned_model"] is None
    assert receipt["usage"] is None and receipt["origin"] == "unit-test"
    saved = json.loads((directory / "call-001/response.json").read_text())
    assert saved["raw_text"] == raw
    with pytest.raises(ModelUnavailable, match="budget"):
        model.propose({}, protocol())
    with pytest.raises(FileExistsError):
        publish_response(directory / "call-001", raw_text=raw, model="test-fixture", origin="unit-test")


def test_proposer_protocol_contains_no_source_identity():
    public = proposer_protocol(protocol())
    assert not {"train_groups", "validation_groups", "audit_groups", "data_version"} & public.keys()
    assert public["area_fraction"] == protocol().area_fraction


@pytest.mark.parametrize("corruption", ["hash", "model", "raw", "binding"])
def test_corrupt_external_response_is_a_recorded_failure(tmp_path, corruption):
    directory = tmp_path / "exchange"
    model = FileExchangeModel(directory, model="test-fixture", maximum_calls=1, timeout=5)
    with ThreadPoolExecutor(max_workers=1) as pool:
        task = pool.submit(model.propose, {}, protocol())
        wait_request(directory)
        folder = directory / "call-001"
        p = asdict(proposal())
        response = {"request_sha256": file_hash(folder / "request.json"), "model": "test-fixture",
                    "origin": "unit-test", "proposal": p, "raw_text": json.dumps(p)}
        if corruption == "model":
            response["model"] = "wrong-model"
        elif corruption == "raw":
            response["raw_text"] = json.dumps({**p, "hypothesis": "changed by broker"})
        elif corruption == "binding":
            response["request_sha256"] = "0" * 64
        write_new(folder / "response.json", response)
        write_new(folder / "response.ready.json", {
            "response_sha256": "0" * 64 if corruption == "hash" else file_hash(folder / "response.json")})
        value, receipt = task.result(timeout=5)
    assert value is None and receipt["status"] == "invalid_external_response"


def test_timeout_is_not_a_generated_proposal(tmp_path):
    model = FileExchangeModel(tmp_path / "exchange", model="test-fixture", maximum_calls=1, timeout=.01)
    value, receipt = model.propose({}, protocol())
    assert value is None and receipt["status"] == "external_model_timeout"


def test_publish_refuses_wrong_dispatch_model_before_writing(tmp_path):
    folder = tmp_path / "call"
    folder.mkdir()
    write_new(folder / "request.json", {"model": "test-fixture"})
    with pytest.raises(ValueError, match="Dispatch model"):
        publish_response(folder, raw_text=json.dumps(asdict(proposal())), model="wrong", origin="unit-test")
    assert not (folder / "response.json").exists()
