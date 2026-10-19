"""Handoff checks use tiny synthetic records, never overwrite real evidence."""
from dataclasses import asdict
import json

import pytest

from experiments.rsi_program_evolution_v1.broker import stage, publish
from experiments.rsi_program_evolution_v1.contracts import digest, write_new
from experiments.rsi_program_evolution_v1.final_report import observation_payload, tests_receipt as read_tests, verify_packet
from experiments.rsi_program_evolution_v1.test_core import proposal


def test_report_requires_executed_passing_tests(tmp_path):
    path = tmp_path / "tests.xml"
    path.write_text('<testsuites><testsuite><testcase classname="unit" name="one"/></testsuite></testsuites>')
    assert read_tests(path)["passed"] == 1
    for content in ('<testsuites/>', '<testsuites><testcase classname="unit" name="one"><skipped/></testcase></testsuites>'):
        path.write_text(content)
        with pytest.raises(ValueError):
            read_tests(path)


def test_report_rechecks_original_model_reply(tmp_path):
    source, packet = tmp_path / "source", tmp_path / "packet"
    source.mkdir()
    write_new(source / "request.json", {"model": "gpt-5.6-sol", "context": {"synthetic": True}})
    stage(source, packet)
    write_new(packet / "raw_response.json", asdict(proposal()))
    publish(packet, "/synthetic/test/session")
    result = verify_packet(packet)
    assert result["requested_model"] == "gpt-5.6-sol"
    assert result["usage"] is None and result["returned_model"] is None
    # Dataclass ancestry is a tuple, JSON ancestry a list: canonical data must
    # agree without falsely treating serialization as an algorithm change.
    assert digest(asdict(proposal())) == digest(result["proposal"])
    # This file is a fresh test fixture, not an archived scientific reply.
    (packet / "raw_response.json").write_text(json.dumps(asdict(proposal(hypothesis="changed"))))
    with pytest.raises(ValueError, match="Raw model reply changed"):
        verify_packet(packet)


def test_comparison_ignores_only_trial_locator_not_observed_numbers():
    a = {"diagnostic": {"trial": "a", "candidate": "same", "coverage": .5},
         "budget": {"calls": 3}, "field_contract": {"image": "normalized"}}
    b = json.loads(json.dumps(a))
    b["diagnostic"]["trial"] = "b"
    assert observation_payload(a) == observation_payload(b)
    b["diagnostic"]["coverage"] = .6
    assert observation_payload(a) != observation_payload(b)
    assert a["diagnostic"]["trial"] == "a"
