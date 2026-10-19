import hashlib
import json

import pytest

from scripts.verify_rsi_cycle import Audit, disjoint_training, junit_counts


def test_hash_and_changed_bytes_are_rejected(tmp_path):
    path = tmp_path / "evidence.json"
    path.write_text('{"a": 1}')
    audit = Audit(tmp_path)
    assert audit.read(path) == {"a": 1}
    audit.check(path, hashlib.sha256(path.read_bytes()).hexdigest())
    path.write_text('{"a": 2}')
    with pytest.raises(ValueError, match="changed during audit"):
        audit.read(path)


def test_wrong_expected_digest_rejected(tmp_path):
    (tmp_path / "artifact").write_bytes(b"test")
    with pytest.raises(ValueError, match="Hash mismatch"):
        Audit(tmp_path).check("artifact", "0" * 64)


def test_json_rehashes_when_metadata_cache_cannot_distinguish_rewrite(tmp_path):
    path = tmp_path / "evidence.json"
    path.write_text('{"a": 1}')
    audit = Audit(tmp_path)
    audit.read(path)
    path.write_text('{"a": 2}')
    stat = path.stat()
    # Simulate indistinguishable timestamps without depending on FS resolution.
    audit.stats[path.name] = (stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    with pytest.raises(ValueError, match="changed during audit"):
        audit.read(path)


def test_project_escape_and_external_symlink_rejected(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.write_bytes(b"data")
    (root / "link").symlink_to(outside)
    audit = Audit(root)
    for path in [root, outside, "../outside", "link"]:
        with pytest.raises(ValueError, match="outside project"):
            audit.path(path)


def test_both_inner_and_outer_groups_excluded():
    entries = {1: {"group": "inner"}, 2: {"group": "outer"}, 3: {"group": "train"}}
    disjoint_training(entries, {"train_case_numbers": [3]}, {"inner", "outer"}, [3])
    for number in (1, 2):
        with pytest.raises(ValueError, match="leaked"):
            disjoint_training(entries, {"train_case_numbers": [number]}, {"inner", "outer"}, [number])
    with pytest.raises(ValueError, match="inventory"):
        disjoint_training(entries, {"train_case_numbers": [3]}, {"inner", "outer"}, [2])


@pytest.mark.parametrize("xml,valid", [
    ('<testsuites><testsuite tests="2" failures="0" errors="0" skipped="0"/></testsuites>', True),
    ('<testsuite tests="1" failures="0" errors="0"/>', True),
    ('<testsuite tests="1" failures="1" errors="0"/>', False),
    ('<testsuite tests="0" failures="0" errors="0"/>', False),
    ('<testsuites/>', False),
])
def test_junit_requires_actual_nonzero_passing_tests(tmp_path, xml, valid):
    path = tmp_path / "junit.xml"
    path.write_text(xml)
    if valid:
        assert junit_counts(path)["tests"] > 0
    else:
        with pytest.raises(ValueError):
            junit_counts(path)
