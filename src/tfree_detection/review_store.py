"""Durable, append-only expert-review events with one current projection."""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping


REVIEW_STORE_SCHEMA_VERSION = "1"
REVIEW_PROJECT_ID = "phyevo-mic-2d-unlabeled-v1"
_IDENTIFIER = re.compile(r"^[A-Za-z0-9_.:@-]{1,160}$")
_DIGEST = re.compile(r"^[a-f0-9]{64}$")


class CaseReviewAction(str, Enum):
    """Case-level decisions exposed by the online review application."""

    ACCEPT_PREDICTION = "accept_prediction"
    CORRECT_GEOMETRY = "correct_geometry"
    REJECT_PREDICTION = "reject_prediction"
    NO_DEFECT = "no_defect"
    ARTIFACT = "artifact"
    UNKNOWN = "unknown"


TRAINING_READY_ACTIONS = {
    CaseReviewAction.ACCEPT_PREDICTION,
    CaseReviewAction.CORRECT_GEOMETRY,
    CaseReviewAction.NO_DEFECT,
}


_GEOMETRY_TYPE = {
    "point": "point_set",
    "line": "line_trace",
    "interface": "boundary_trace",
    "region": "region_contour",
}
_MINIMUM_POINTS = {"point": 1, "line": 2, "interface": 2, "region": 3}


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _payload_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _validate_identifier(value: str, field_name: str) -> None:
    if not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"{field_name} contains unsupported characters")


def validate_review_geometry(
    geometry: Mapping[str, Any],
    *,
    geometry_hint: str,
    image_shape: tuple[int, int],
) -> None:
    """Validate expert geometry in the source image coordinate system."""

    if geometry_hint not in _GEOMETRY_TYPE:
        raise ValueError(f"Unsupported geometry hint {geometry_hint!r}")
    if geometry.get("coordinate_space") != "source_pixels_xy":
        raise ValueError("Review geometry must use source_pixels_xy")
    if geometry.get("geometry_type") != _GEOMETRY_TYPE[geometry_hint]:
        raise ValueError("Review geometry type does not match the case")
    recorded_shape = geometry.get("image_shape_yx")
    if recorded_shape != list(image_shape):
        raise ValueError("Review geometry image shape does not match the case")
    points = geometry.get("points_xy")
    if not isinstance(points, list):
        raise TypeError("Review geometry points_xy must be a JSON array")
    minimum = _MINIMUM_POINTS[geometry_hint]
    if len(points) < minimum:
        raise ValueError(
            f"{geometry_hint} geometry requires at least {minimum} points"
        )
    if len(points) > 2048:
        raise ValueError("Review geometry exceeds the 2048-point safety limit")
    height, width = image_shape
    for point in points:
        if not isinstance(point, list) or len(point) != 2:
            raise TypeError("Each geometry point must be [x, y]")
        x, y = point
        if isinstance(x, bool) or isinstance(y, bool):
            raise TypeError("Geometry coordinates must be numeric")
        try:
            x_value, y_value = float(x), float(y)
        except (TypeError, ValueError) as error:
            raise TypeError("Geometry coordinates must be numeric") from error
        if not math.isfinite(x_value) or not math.isfinite(y_value):
            raise ValueError("Geometry coordinates must be finite")
        if not (0.0 <= x_value <= width - 1):
            raise ValueError("Geometry x coordinate lies outside the image")
        if not (0.0 <= y_value <= height - 1):
            raise ValueError("Geometry y coordinate lies outside the image")


@dataclass(frozen=True)
class ReviewSubmission:
    """One proposed current state before immutable revision metadata is added."""

    project_id: str
    reviewer_id: str
    case_id: str
    dataset_id: str
    review_split: str
    selected_sample_id: str
    applies_to_sample_ids: tuple[str, ...]
    geometry_hint: str
    annotation_scope: str
    image_shape: tuple[int, int]
    action: CaseReviewAction
    defect_labels: tuple[str, ...]
    geometry: Mapping[str, Any] | None
    model_assistance_shown: bool
    prediction_sha256: str | None
    case_manifest_sha256: str
    annotation_duration_seconds: float
    notes: str
    client_submission_id: str

    def validate(self) -> None:
        for field_name, value in (
            ("project_id", self.project_id),
            ("reviewer_id", self.reviewer_id),
            ("case_id", self.case_id),
            ("dataset_id", self.dataset_id),
            ("selected_sample_id", self.selected_sample_id),
            ("client_submission_id", self.client_submission_id),
        ):
            _validate_identifier(value, field_name)
        if self.review_split not in {"train", "validation"}:
            raise ValueError("Expert review may only write train/validation cases")
        if not self.applies_to_sample_ids:
            raise ValueError("Review must apply to at least one sample")
        if self.selected_sample_id not in self.applies_to_sample_ids:
            raise ValueError("Selected sample must belong to the review case")
        if len(set(self.applies_to_sample_ids)) != len(
            self.applies_to_sample_ids
        ):
            raise ValueError("Review case sample ids must be unique")
        for sample_id in self.applies_to_sample_ids:
            _validate_identifier(sample_id, "applies_to_sample_ids")
        if self.geometry_hint not in _GEOMETRY_TYPE:
            raise ValueError("Unsupported review geometry hint")
        if len(self.image_shape) != 2 or min(self.image_shape) < 1:
            raise ValueError("Review image shape must be positive and two-dimensional")
        if not isinstance(self.action, CaseReviewAction):
            raise TypeError("Review action must be a CaseReviewAction")
        labels = tuple(label.strip() for label in self.defect_labels)
        if any(not label for label in labels) or len(set(labels)) != len(labels):
            raise ValueError("Defect labels must be non-empty and unique")
        if self.action in {
            CaseReviewAction.ACCEPT_PREDICTION,
            CaseReviewAction.CORRECT_GEOMETRY,
        } and not labels:
            raise ValueError("Accepted or corrected defects require a label")
        if self.action == CaseReviewAction.NO_DEFECT and labels:
            raise ValueError("no_defect cannot carry positive defect labels")
        if self.action == CaseReviewAction.CORRECT_GEOMETRY:
            if self.geometry is None:
                raise ValueError("correct_geometry requires expert geometry")
            validate_review_geometry(
                self.geometry,
                geometry_hint=self.geometry_hint,
                image_shape=self.image_shape,
            )
        elif self.geometry is not None:
            raise ValueError("Only correct_geometry may carry expert geometry")
        if self.action in {
            CaseReviewAction.ACCEPT_PREDICTION,
            CaseReviewAction.REJECT_PREDICTION,
        } and not self.model_assistance_shown:
            raise ValueError("Prediction decisions require model assistance to be shown")
        if self.action == CaseReviewAction.ACCEPT_PREDICTION:
            if self.prediction_sha256 is None:
                raise ValueError("accept_prediction requires a prediction digest")
        if self.prediction_sha256 is not None and not _DIGEST.fullmatch(
            self.prediction_sha256
        ):
            raise ValueError("prediction_sha256 must be a lowercase SHA-256")
        if not _DIGEST.fullmatch(self.case_manifest_sha256):
            raise ValueError("case_manifest_sha256 must be a lowercase SHA-256")
        if not math.isfinite(self.annotation_duration_seconds):
            raise ValueError("Annotation duration must be finite")
        if self.annotation_duration_seconds <= 0:
            raise ValueError("Annotation duration must be positive")
        if len(self.notes) > 8000:
            raise ValueError("Review notes exceed the 8000-character limit")

    def semantic_payload(self) -> dict[str, Any]:
        """Return content that defines a revision, excluding click identity/time."""

        self.validate()
        payload = asdict(self)
        payload["action"] = self.action.value
        payload["defect_labels"] = list(self.defect_labels)
        payload["applies_to_sample_ids"] = list(self.applies_to_sample_ids)
        payload["image_shape"] = list(self.image_shape)
        payload["geometry"] = (
            None if self.geometry is None else dict(self.geometry)
        )
        payload.pop("client_submission_id")
        payload.pop("annotation_duration_seconds")
        return payload


@dataclass(frozen=True)
class ReviewRevision:
    revision_id: str
    revision_number: int
    submitted_at_utc: str
    client_submission_id: str
    payload_sha256: str
    annotation_duration_seconds: float
    submission: Mapping[str, Any]

    @property
    def case_id(self) -> str:
        return str(self.submission["case_id"])

    @property
    def reviewer_id(self) -> str:
        return str(self.submission["reviewer_id"])

    @property
    def action(self) -> str:
        return str(self.submission["action"])

    @property
    def training_ready(self) -> bool:
        return CaseReviewAction(self.action) in TRAINING_READY_ACTIONS

    def to_dict(self) -> dict[str, Any]:
        return {
            "review_revision_format": "phyevo-mic-review-revision-v1",
            "revision_id": self.revision_id,
            "revision_number": self.revision_number,
            "submitted_at_utc": self.submitted_at_utc,
            "client_submission_id": self.client_submission_id,
            "payload_sha256": self.payload_sha256,
            "annotation_duration_seconds": self.annotation_duration_seconds,
            "training_ready": self.training_ready,
            **dict(self.submission),
        }


class ReviewStore:
    """SQLite owner of immutable review events and their current projection."""

    def __init__(self, path: str | Path, *, project_id: str = REVIEW_PROJECT_ID):
        self.path = str(path)
        self.project_id = project_id
        _validate_identifier(project_id, "project_id")
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=15.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 15000")
        if self.path != ":memory:":
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA synchronous = FULL")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS review_store_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS review_events (
                    revision_id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    reviewer_id TEXT NOT NULL,
                    case_id TEXT NOT NULL,
                    revision_number INTEGER NOT NULL CHECK (revision_number > 0),
                    submitted_at_utc TEXT NOT NULL,
                    client_submission_id TEXT NOT NULL,
                    payload_sha256 TEXT NOT NULL,
                    annotation_duration_seconds REAL NOT NULL
                        CHECK (annotation_duration_seconds > 0),
                    payload_json TEXT NOT NULL,
                    UNIQUE (project_id, reviewer_id, case_id, revision_number),
                    UNIQUE (project_id, client_submission_id)
                );
                CREATE TABLE IF NOT EXISTS current_reviews (
                    project_id TEXT NOT NULL,
                    reviewer_id TEXT NOT NULL,
                    case_id TEXT NOT NULL,
                    revision_id TEXT NOT NULL UNIQUE,
                    PRIMARY KEY (project_id, reviewer_id, case_id),
                    FOREIGN KEY (revision_id) REFERENCES review_events(revision_id)
                );
                CREATE INDEX IF NOT EXISTS review_events_case_idx
                    ON review_events(project_id, reviewer_id, case_id);
                """
            )
            existing = connection.execute(
                "SELECT value FROM review_store_meta WHERE key = 'schema_version'"
            ).fetchone()
            if existing is None:
                connection.execute(
                    "INSERT INTO review_store_meta(key, value) VALUES (?, ?)",
                    ("schema_version", REVIEW_STORE_SCHEMA_VERSION),
                )
            elif existing["value"] != REVIEW_STORE_SCHEMA_VERSION:
                raise RuntimeError("Unsupported expert review store schema")

    @staticmethod
    def _revision(row: sqlite3.Row) -> ReviewRevision:
        return ReviewRevision(
            revision_id=str(row["revision_id"]),
            revision_number=int(row["revision_number"]),
            submitted_at_utc=str(row["submitted_at_utc"]),
            client_submission_id=str(row["client_submission_id"]),
            payload_sha256=str(row["payload_sha256"]),
            annotation_duration_seconds=float(
                row["annotation_duration_seconds"]
            ),
            submission=json.loads(str(row["payload_json"])),
        )

    def save(self, submission: ReviewSubmission) -> ReviewRevision:
        submission.validate()
        if submission.project_id != self.project_id:
            raise ValueError("Submission project does not match the review store")
        semantic = submission.semantic_payload()
        payload_json = _canonical_json(semantic)
        payload_digest = _payload_sha256(semantic)
        submitted_at = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            duplicate_click = connection.execute(
                """
                SELECT * FROM review_events
                WHERE project_id = ? AND client_submission_id = ?
                """,
                (self.project_id, submission.client_submission_id),
            ).fetchone()
            if duplicate_click is not None:
                if duplicate_click["payload_sha256"] != payload_digest:
                    raise ValueError(
                        "client_submission_id was reused for different content"
                    )
                return self._revision(duplicate_click)
            current = connection.execute(
                """
                SELECT e.* FROM current_reviews AS c
                JOIN review_events AS e ON e.revision_id = c.revision_id
                WHERE c.project_id = ? AND c.reviewer_id = ? AND c.case_id = ?
                """,
                (self.project_id, submission.reviewer_id, submission.case_id),
            ).fetchone()
            if current is not None and current["payload_sha256"] == payload_digest:
                return self._revision(current)
            latest = connection.execute(
                """
                SELECT COALESCE(MAX(revision_number), 0) AS latest
                FROM review_events
                WHERE project_id = ? AND reviewer_id = ? AND case_id = ?
                """,
                (self.project_id, submission.reviewer_id, submission.case_id),
            ).fetchone()
            revision_number = int(latest["latest"]) + 1
            revision_id = f"revision:{uuid.uuid4()}"
            connection.execute(
                """
                INSERT INTO review_events(
                    revision_id, project_id, reviewer_id, case_id,
                    revision_number, submitted_at_utc, client_submission_id,
                    payload_sha256, annotation_duration_seconds, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    revision_id,
                    self.project_id,
                    submission.reviewer_id,
                    submission.case_id,
                    revision_number,
                    submitted_at,
                    submission.client_submission_id,
                    payload_digest,
                    submission.annotation_duration_seconds,
                    payload_json,
                ),
            )
            connection.execute(
                """
                INSERT INTO current_reviews(
                    project_id, reviewer_id, case_id, revision_id
                ) VALUES (?, ?, ?, ?)
                ON CONFLICT(project_id, reviewer_id, case_id)
                DO UPDATE SET revision_id = excluded.revision_id
                """,
                (
                    self.project_id,
                    submission.reviewer_id,
                    submission.case_id,
                    revision_id,
                ),
            )
            row = connection.execute(
                "SELECT * FROM review_events WHERE revision_id = ?",
                (revision_id,),
            ).fetchone()
        return self._revision(row)

    def get_current(
        self, reviewer_id: str, case_id: str
    ) -> ReviewRevision | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT e.* FROM current_reviews AS c
                JOIN review_events AS e ON e.revision_id = c.revision_id
                WHERE c.project_id = ? AND c.reviewer_id = ? AND c.case_id = ?
                """,
                (self.project_id, reviewer_id, case_id),
            ).fetchone()
        return None if row is None else self._revision(row)

    def history(self, reviewer_id: str, case_id: str) -> list[ReviewRevision]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM review_events
                WHERE project_id = ? AND reviewer_id = ? AND case_id = ?
                ORDER BY revision_number
                """,
                (self.project_id, reviewer_id, case_id),
            ).fetchall()
        return [self._revision(row) for row in rows]

    def list_current(
        self, *, reviewer_id: str | None = None
    ) -> list[ReviewRevision]:
        query = """
            SELECT e.* FROM current_reviews AS c
            JOIN review_events AS e ON e.revision_id = c.revision_id
            WHERE c.project_id = ?
        """
        parameters: list[str] = [self.project_id]
        if reviewer_id is not None:
            query += " AND c.reviewer_id = ?"
            parameters.append(reviewer_id)
        query += " ORDER BY c.reviewer_id, c.case_id"
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [self._revision(row) for row in rows]

    def list_history(
        self, *, reviewer_id: str | None = None
    ) -> list[ReviewRevision]:
        query = "SELECT * FROM review_events WHERE project_id = ?"
        parameters: list[str] = [self.project_id]
        if reviewer_id is not None:
            query += " AND reviewer_id = ?"
            parameters.append(reviewer_id)
        query += " ORDER BY reviewer_id, case_id, revision_number"
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [self._revision(row) for row in rows]

    def progress(
        self,
        case_ids: Iterable[str],
        *,
        reviewer_id: str,
    ) -> dict[str, Any]:
        expected = tuple(dict.fromkeys(case_ids))
        allowed = set(expected)
        current = [
            item
            for item in self.list_current(reviewer_id=reviewer_id)
            if item.case_id in allowed
        ]
        actions: dict[str, int] = {}
        for item in current:
            actions[item.action] = actions.get(item.action, 0) + 1
        completed = len(current)
        return {
            "reviewer_id": reviewer_id,
            "case_count": len(expected),
            "completed_count": completed,
            "remaining_count": len(expected) - completed,
            "completion_fraction": (
                0.0 if not expected else completed / len(expected)
            ),
            "training_ready_count": sum(item.training_ready for item in current),
            "action_counts": dict(sorted(actions.items())),
        }

    def integrity_report(self) -> dict[str, Any]:
        with self._connect() as connection:
            quick = connection.execute("PRAGMA quick_check").fetchone()[0]
            orphan_current = connection.execute(
                """
                SELECT COUNT(*) FROM current_reviews AS c
                LEFT JOIN review_events AS e ON e.revision_id = c.revision_id
                WHERE e.revision_id IS NULL
                """
            ).fetchone()[0]
            event_count = connection.execute(
                "SELECT COUNT(*) FROM review_events WHERE project_id = ?",
                (self.project_id,),
            ).fetchone()[0]
            current_count = connection.execute(
                "SELECT COUNT(*) FROM current_reviews WHERE project_id = ?",
                (self.project_id,),
            ).fetchone()[0]
        return {
            "schema_version": REVIEW_STORE_SCHEMA_VERSION,
            "sqlite_quick_check": str(quick),
            "event_count": int(event_count),
            "current_count": int(current_count),
            "orphan_current_count": int(orphan_current),
            "valid": quick == "ok" and int(orphan_current) == 0,
        }


def revisions_jsonl(revisions: Iterable[ReviewRevision]) -> bytes:
    """Serialize an ordered export without modifying the authoritative store."""

    lines = [
        json.dumps(item.to_dict(), ensure_ascii=False, sort_keys=True)
        for item in revisions
    ]
    return (("\n".join(lines) + "\n") if lines else "").encode("utf-8")
