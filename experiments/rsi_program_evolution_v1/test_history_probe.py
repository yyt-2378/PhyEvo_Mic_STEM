"""Read-only contract tests for the historical diagnosis adapter."""
import copy
import unittest

from . import history_probe as h


class HistoryProbeTests(unittest.TestCase):
    def setUp(self):
        old = h.read(h.OLD / "requests/blind-R07.json")
        self.request = {"slot": "q07", "case_id": "R07", "system": old["system"],
                        "packet": old["packet"], "validated_memory": [],
                        "research_plan": {"action": "tune", "parents": [], "focus": "generic"},
                        "adapter_scope": "diagnosis only"}

    def test_original_evidence_passes(self):
        h.check_request(self.request)

    def test_answer_injection_is_rejected(self):
        self.request["historical_outcome"] = "answer"
        with self.assertRaises(ValueError):
            h.check_request(self.request)

    def test_old_memory_is_rejected(self):
        self.request["validated_memory"] = [{"memory_id": "M-old"}]
        with self.assertRaises(ValueError):
            h.check_request(self.request)

    def test_observation_mutation_is_rejected(self):
        self.request["packet"] = copy.deepcopy(self.request["packet"])
        self.request["packet"]["observations"][0]["value"] = "later answer"
        with self.assertRaises(ValueError):
            h.check_request(self.request)

    def test_original_response_contract_is_reused(self):
        answer = h.read(h.OLD / "responses/blind-R07.json")["result"]
        h.legacy_validator()(self.request, answer)
        answer["hypotheses"][0]["evidence_ids"] = ["invented"]
        with self.assertRaises(ValueError):
            h.legacy_validator()(self.request, answer)

    def test_registered_case_and_repeat_counts(self):
        self.assertEqual(len(h.CASES), 19)
        self.assertEqual(len(h.REPEATS), 4)
        self.assertEqual(len(set(h.CASES)), len(h.CASES))
        self.assertTrue(set(h.REPEATS) <= set(h.CASES))


if __name__ == "__main__":
    unittest.main(verbosity=2)
