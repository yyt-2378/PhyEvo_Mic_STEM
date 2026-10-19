from copy import deepcopy

import pytest

from tfree_detection.interface_release import select_retention_step


def fixture_runs():
    protocol={"groups":["outer","a","b"],"checkpoints":[50,100]}
    runs=[]
    for group in ["a","b"]:
        for seed in [23,41,67]:
            records=[]
            for step in [50,100]:
                records.extend([
                    {"step":step,"case_number":1,"group":"b" if group=="a" else "a",
                     "is_training_diagnostic":True,"health":{"eligible":True},"coverage":.99},
                    {"step":step,"case_number":2,"group":group,"is_training_diagnostic":False,
                     "health":{"eligible":False if seed==41 else True},"coverage":.001,
                     "routed_coverage":.7 if step==50 else .6}])
            runs.append({"job":{"index":len(runs),"outer_group":"outer","validation_group":group,
                "stage":"inner","seed":seed,"train_groups":["b" if group=="a" else "a"]},"records":records})
    return protocol,runs


def test_selects_guarded_inner_result_not_outer_results_or_raw_bad_student():
    p,r=fixture_runs()
    result=select_retention_step(p,r,"outer")
    assert result["selected"]["step"]==50
    assert any(v["student_output_vetoes"] for v in result["selected"]["receipts"])
    r.append({"job":{"stage":"outer","outer_group":"outer","seed":23},
              "records":[{"coverage":100000,"step":100}]})
    assert select_retention_step(p,r,"outer")==result


def test_saturated_training_checkpoint_still_rejected():
    p,r=fixture_runs()
    for rec in r[0]["records"]:
        if rec["step"]==50 and rec["is_training_diagnostic"]:rec["health"]["eligible"]=False
    assert select_retention_step(p,r,"outer")["selected"]["step"]==100


def test_heldout_source_in_training_is_rejected():
    p,r=fixture_runs();r[0]["job"]["train_groups"].append("outer")
    with pytest.raises(ValueError,match="leakage"):select_retention_step(p,r,"outer")


def test_tie_break_is_earliest_and_limited_identifiability_is_explicit():
    p,r=fixture_runs()
    for run in r:
        for rec in run["records"]:
            if not rec["is_training_diagnostic"]:rec["routed_coverage"]=.8
    result=select_retention_step(p,r,"outer")
    assert result["selected"]["step"]==50
    assert not result["inner_objective_distinguishes_steps"]
