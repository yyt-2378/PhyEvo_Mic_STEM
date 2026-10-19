"""Append-only candidate/experiment archive, separate from the incumbent."""
from __future__ import annotations

from dataclasses import asdict
import json
import math
from pathlib import Path
import sqlite3

from .contracts import canonical, digest, file_hash, inside, Proposal, write_new


class Archive:
    def __init__(self, directory, protocol, *, create=False):
        self.root = Path(directory).resolve()
        if create:
            self.root.mkdir(parents=True, exist_ok=False)
            write_new(self.root / "protocol.json", asdict(protocol))
            (self.root / "programs").mkdir()
        stored = json.loads((self.root / "protocol.json").read_text())
        if digest(stored) != protocol.identity:
            raise ValueError("Protocol changed: start a separate experiment")
        self.protocol = protocol
        database = inside(self.root, "archive.sqlite3")
        if not create and not database.is_file():
            raise ValueError("Missing archive; not a resumable experiment")
        self.db = sqlite3.connect(database, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys=ON")
        if create:
            self.db.executescript("""
                CREATE TABLE candidates(id TEXT PRIMARY KEY, lineage TEXT NOT NULL,
                    action TEXT NOT NULL, parents TEXT NOT NULL, proposal TEXT NOT NULL,
                    code_hash TEXT NOT NULL, policy_hash TEXT NOT NULL);
                CREATE TABLE trials(id TEXT PRIMARY KEY, candidate TEXT NOT NULL REFERENCES candidates(id),
                    split TEXT NOT NULL, outcome TEXT NOT NULL, receipt TEXT NOT NULL);
                CREATE TABLE events(seq INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT NOT NULL,
                    body TEXT NOT NULL, previous_hash TEXT NOT NULL, hash TEXT NOT NULL);
                CREATE TABLE memories(id TEXT PRIMARY KEY, candidate TEXT NOT NULL REFERENCES candidates(id),
                    trial TEXT NOT NULL REFERENCES trials(id), body TEXT NOT NULL);
            """)
            for table in ("candidates", "trials", "events", "memories"):
                for operation in ("UPDATE", "DELETE"):
                    self.db.execute(f"CREATE TRIGGER protect_{table}_{operation} BEFORE {operation} ON {table} "
                                    "BEGIN SELECT RAISE(ABORT, 'append-only archive'); END")
        self.verify()

    def close(self):
        self.db.close()

    def event(self, kind, body):
        last = self.db.execute("SELECT hash FROM events ORDER BY seq DESC LIMIT 1").fetchone()
        previous = last[0] if last else self.protocol.identity
        h = digest({"kind": kind, "body": body, "previous": previous})
        self.db.execute("INSERT INTO events(kind,body,previous_hash,hash) VALUES(?,?,?,?)",
                        (kind, canonical(body), previous, h))
        return h

    def events(self, kind=None):
        query = "SELECT * FROM events" + (" WHERE kind=?" if kind else "") + " ORDER BY seq"
        return [{**dict(r), "body": json.loads(r["body"])}
                for r in self.db.execute(query, (kind,) if kind else ())]

    def add(self, proposal):
        proposal.validate(self.protocol)
        existing = self.db.execute("SELECT id FROM candidates WHERE id=?", (proposal.identity,)).fetchone()
        if existing:
            self.candidate(proposal.identity)  # Verify existing artifacts before reuse.
            return proposal.identity
        parents = [self.candidate(p) for p in proposal.parents]
        lineage = parents[0]["lineage"] if proposal.action in {"tune", "repair"} else proposal.identity
        target = inside(self.root, Path("programs") / proposal.identity)
        target.mkdir(exist_ok=False)
        for name, content in (("pipeline.py", proposal.pipeline_code), ("policy.py", proposal.policy_code)):
            with (target / name).open("x") as f:
                f.write(content)
        write_new(target / "proposal.json", asdict(proposal))
        self.db.execute("BEGIN IMMEDIATE")
        try:
            self.db.execute("INSERT INTO candidates VALUES(?,?,?,?,?,?,?)", (
                proposal.identity, lineage, proposal.action, canonical(proposal.parents),
                canonical(asdict(proposal)), file_hash(target / "pipeline.py"), file_hash(target / "policy.py")))
            self.event("candidate_registered", {"candidate": proposal.identity, "lineage": lineage})
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise  # Preserve incomplete artifacts, never silently overwrite them.
        return proposal.identity

    def candidate(self, identity):
        r = self.db.execute("SELECT * FROM candidates WHERE id=?", (identity,)).fetchone()
        if r is None:
            raise ValueError("Unknown parent/candidate")
        item = dict(r)
        directory = inside(self.root, Path("programs") / identity)
        if (file_hash(directory / "pipeline.py") != item["code_hash"] or
                file_hash(directory / "policy.py") != item["policy_hash"]):
            raise ValueError("Candidate artifact changed")
        item["proposal"] = Proposal.from_dict(json.loads(item["proposal"]))
        item["parents"] = json.loads(item["parents"])
        item["directory"] = str(directory)
        return item

    def record_trial(self, identity, split, outcome, receipt):
        self.candidate(identity)
        if split not in {"train", "validation", "audit", "smoke", "meta_validation"}:
            raise ValueError("Unknown evaluation split")
        if outcome not in {"ok", "failed", "timeout"}:
            raise ValueError("Unknown execution outcome")
        value = {"candidate": identity, "split": split, "outcome": outcome, "receipt": receipt}
        trial = digest(value)
        self.db.execute("BEGIN IMMEDIATE")
        try:
            self.db.execute("INSERT INTO trials VALUES(?,?,?,?,?)", (trial, identity, split, outcome, canonical(receipt)))
            self.event("trial_recorded", {"trial": trial, "candidate": identity, "split": split, "outcome": outcome})
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise
        return trial

    def trials(self, identity=None, *, visible_only=True):
        rows = self.db.execute("SELECT * FROM trials ORDER BY rowid").fetchall()
        return [{**dict(r), "receipt": json.loads(r["receipt"])} for r in rows
                if (identity is None or r["candidate"] == identity)
                and (not visible_only or r["split"] != "audit")]

    def remember(self, candidate, trial, observation, conditions):
        r = self.db.execute("SELECT * FROM trials WHERE id=?", (trial,)).fetchone()
        if r is None or r["candidate"] != candidate or r["split"] == "audit":
            raise ValueError("Memory must reference this candidate's non-audit experiment")
        body = {"observation": observation, "conditions": conditions,
                "outcome": r["outcome"], "verified_by_trial": trial}
        key = digest({"candidate": candidate, **body})
        self.db.execute("INSERT INTO memories VALUES(?,?,?,?)", (key, candidate, trial, canonical(body)))
        self.event("memory_added", {"memory": key})
        return key

    def memories(self):
        return [{"id": r["id"], "candidate": r["candidate"], **json.loads(r["body"])}
                for r in self.db.execute("SELECT * FROM memories ORDER BY rowid")]

    def incumbent(self):
        states = self.events("selection")
        return states[-1]["body"]["incumbent"] if states else None

    def select_policy(self, candidate, trial, *, minimum_gain=0.0):
        """Meta improvement needs its own paired experiment, not task score alone."""
        self.candidate(candidate)
        row = self.db.execute("SELECT * FROM trials WHERE id=?", (trial,)).fetchone()
        if row is None or row["candidate"] != candidate or row["split"] != "meta_validation" or row["outcome"] != "ok":
            raise ValueError("A separate paired meta-validation trial is required")
        receipt = json.loads(row["receipt"])
        pairs = receipt.get("pairs", [])
        if not pairs or not receipt.get("actual_execution"):
            raise ValueError("No executed meta-policy comparison")
        gains = []
        for pair in pairs:
            for key in ("model", "input_sha256", "starting_program_sha256", "budget"):
                if pair["control"][key] != pair["candidate"][key]:
                    raise ValueError("Unmatched meta-evaluation condition: " + key)
            if not pair["control"]["execution_ids"] or not pair["candidate"]["execution_ids"]:
                raise ValueError("Missing executed descendants")
            measured = []
            for side in ("control", "candidate"):
                run = pair[side]
                baseline_id = run.get("baseline_trial")
                base = self.db.execute("SELECT * FROM trials WHERE id=?", (baseline_id,)).fetchone()
                if base is None or base["split"] != "validation" or base["outcome"] != "ok":
                    raise ValueError("Missing meta starting-point evaluation")
                if self.candidate(base["candidate"])["code_hash"] != run["starting_program_sha256"]:
                    raise ValueError("Meta starting program hash does not match its evaluated baseline")
                base_receipt = json.loads(base["receipt"])
                base_rows = base_receipt.get("rows", [])
                if not base_rows or not base_receipt.get("actual_execution"):
                    raise ValueError("Starting point was not executed")
                from .evaluation import summarize
                reference = summarize(base_rows)
                actual_gains = []
                for identity in run["execution_ids"]:
                    descendant = self.db.execute("SELECT * FROM trials WHERE id=?", (identity,)).fetchone()
                    if descendant is None or descendant["split"] != "validation" or descendant["outcome"] != "ok":
                        raise ValueError("Unresolved or non-validation meta descendant")
                    result = json.loads(descendant["receipt"])
                    if not result.get("actual_execution"):
                        raise ValueError("Meta descendant was not actually executed")
                    rows = result.get("rows", [])
                    keys = lambda values: {(r["group"], r["case"], r["seed"], r["metric"], r["budget"]) for r in values}
                    if len(rows) != len(base_rows) or keys(rows) != keys(base_rows):
                        raise ValueError("Meta descendant evaluation cohort changed")
                    outcome = summarize(rows)
                    # Research utility, not an 'overall detection accuracy'.
                    actual_gains.append(sum(outcome[g]["mean"] - reference[g]["mean"] for g in reference) / len(reference))
                calls = run["budget"].get("model_calls", 0)
                if not isinstance(calls, int) or calls <= 0 or len(run["execution_ids"]) > calls:
                    raise ValueError("Invalid meta model-call budget")
                utility = max(0., max(actual_gains)) / calls
                if not math.isfinite(run["gain_per_call"]) or abs(utility - run["gain_per_call"]) > 1e-10:
                    raise ValueError("Meta gain disagrees with recorded executions")
                measured.append(utility)
            gains.append(measured[1] - measured[0])
        gain = sum(gains) / len(gains)
        accepted = gain > minimum_gain and all(x >= 0 for x in gains)
        previous = self.events("policy_selection")
        incumbent = (previous[-1]["body"]["incumbent"] if previous else
                     self.events("baseline_registered")[0]["body"]["candidate"])
        self.event("policy_selection", {"candidate": candidate, "trial": trial,
                   "accepted": accepted, "incumbent": candidate if accepted else incumbent,
                   "paired_gain_per_call": gain, "pairs": len(pairs),
                   "statistical_significance_claimed": False})
        return accepted

    def select(self, candidate, accepted, reason, trial):
        self.candidate(candidate)
        row = self.db.execute("SELECT * FROM trials WHERE id=?", (trial,)).fetchone()
        if row is None or row["candidate"] != candidate or row["split"] != "validation" or row["outcome"] != "ok":
            raise ValueError("Selection requires a successful validation trial")
        incumbent = candidate if accepted else self.incumbent()
        if incumbent is None:
            raise ValueError("A baseline must be registered first")
        self.event("selection", {"candidate": candidate, "accepted": accepted,
                                "incumbent": incumbent, "reason": reason, "trial": trial})

    def pool(self):
        """Rejected but successfully evaluated candidates remain research parents."""
        ids = [r[0] for r in self.db.execute("SELECT id FROM candidates ORDER BY rowid")]
        return [{"id": i, "lineage": self.candidate(i)["lineage"],
                 "trials": self.trials(i)} for i in ids]

    def verify(self):
        previous = self.protocol.identity
        for row in self.events():
            if row["previous_hash"] != previous or row["hash"] != digest({
                    "kind": row["kind"], "body": row["body"], "previous": previous}):
                raise ValueError("Archive event chain damaged")
            previous = row["hash"]
        for row in self.db.execute("SELECT id FROM candidates"):
            self.candidate(row[0])
        return previous
