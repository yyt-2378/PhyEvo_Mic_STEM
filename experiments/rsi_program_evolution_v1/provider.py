"""Small, explicitly budgeted JSON-model adapter. No credential files are read."""
from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
import time
import urllib.error
import urllib.parse
import urllib.request

from .contracts import digest, file_hash, inside, Proposal, write_new


PROPOSAL_SCHEMA = {"type": "object", "additionalProperties": False, "properties": {
    "action": {"type": "string", "enum": ["generate", "tune", "evolve", "repair"]},
    "parents": {"type": "array", "items": {"type": "string"}},
    **{k: {"type": "string"} for k in ("observation", "hypothesis", "intervention", "expected_effect",
                                      "falsifier", "pipeline_code", "policy_code")},
    "evidence_ids": {"type": "array", "items": {"type": "string"}},
    "training_steps": {"type": "integer"},
    "target": {"type": "string", "enum": ["analysis", "learning", "routing", "research_policy"]}},
    "required": ["action", "parents", "observation", "hypothesis", "intervention", "expected_effect",
                 "falsifier", "pipeline_code", "policy_code", "evidence_ids", "training_steps", "target"]}


class ModelUnavailable(RuntimeError):
    pass


def proposer_protocol(protocol):
    """Expose budgets, never actual source names or held-out audit identities."""
    return {k: v for k, v in asdict(protocol).items()
            if k not in {"train_groups", "validation_groups", "audit_groups", "data_version"}}


def publish_response(directory, *, raw_text, model, origin, usage=None,
                     provider_returned_model=None):
    """Trusted transport broker: preserve model text, never edit its algorithm.

    ``model`` is the requested dispatch model. Backend model attestation and
    usage are separate optional fields; a broker cannot infer them from text.
    """
    root = Path(__file__).resolve().parents[2]
    folder = inside(root, directory)
    request_path = inside(root, folder / "request.json")
    request = json.loads(request_path.read_text())
    if model != request["model"] or not isinstance(origin, str) or not origin.strip():
        raise ValueError("Dispatch model or origin does not match sealed request")
    value = json.loads(raw_text)
    # Construction only; validation against the private full protocol occurs
    # at the receiving adapter. No generated source is executed by the broker.
    Proposal.from_dict(value)
    response = inside(root, folder / "response.json")
    marker = inside(root, folder / "response.ready.json")
    if response.exists() or marker.exists():
        raise FileExistsError("A response has already been published")
    write_new(response, {"request_sha256": file_hash(request_path), "model": model,
                         "origin": origin, "proposal": value, "raw_text": raw_text,
                         "provider_returned_model": provider_returned_model, "usage": usage})
    write_new(marker, {"response_sha256": file_hash(response)})
    return file_hash(response)


class JsonModel:
    """The controller stores requests/responses; candidates never see credentials.

    Third-party OpenAI-compatible services are recorded as such. A 4xx is not
    bypassed by replacing hosts or models. Optional mode is explicit, not fallback.
    """
    def __init__(self, model, *, maximum_calls, maximum_output_tokens=6000,
                 mode="responses", timeout=90):
        if not model or mode not in {"responses", "chat"} or maximum_calls <= 0:
            raise ValueError("Explicit model, API mode and call budget required")
        self.model = model
        self.maximum_calls = maximum_calls
        self.maximum_output_tokens = maximum_output_tokens
        self.mode, self.timeout, self.calls = mode, timeout, 0

    def request(self, system, content, schema):
        if self.calls >= self.maximum_calls:
            raise ModelUnavailable("Model-call budget exhausted")
        base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        parts = urllib.parse.urlsplit(base)
        if parts.scheme != "https" or parts.username or parts.password or parts.query or parts.fragment:
            raise ModelUnavailable("Expected a credential-free HTTPS API base URL")
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ModelUnavailable("OPENAI_API_KEY is not configured; no credential file fallback")
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": json.dumps(content, ensure_ascii=False, allow_nan=False)}]
        fmt = {"name": "program_proposal", "schema": schema, "strict": True}
        if self.mode == "responses":
            payload = {"model": self.model, "input": messages, "store": False,
                       "max_output_tokens": self.maximum_output_tokens,
                       "text": {"format": {"type": "json_schema", **fmt}}}
            endpoint = "/responses"
        else:
            payload = {"model": self.model, "messages": messages,
                       "max_completion_tokens": self.maximum_output_tokens,
                       "response_format": {"type": "json_schema", "json_schema": fmt}}
            endpoint = "/chat/completions"
        request = urllib.request.Request(base + endpoint, data=json.dumps(payload).encode(), headers={
            "Authorization": "Bearer " + key, "Content-Type": "application/json",
            "User-Agent": "PhyEvo-Mic-RSI-Experiment/1.0"})
        self.calls += 1
        started = time.monotonic()
        receipt = {"requested_model": self.model, "api_mode": self.mode,
                   "provider_host": parts.hostname, "official_openai_host": parts.hostname == "api.openai.com",
                   "call_index": self.calls, "request_sha256": digest(payload),
                   "max_output_tokens": self.maximum_output_tokens}
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                result = json.load(response)
        except urllib.error.HTTPError as e:
            # Do not print response bodies: gateways can echo credentials/URLs.
            receipt.update(status="http_error", http_status=e.code,
                           seconds=time.monotonic() - started)
            try:
                error = json.loads(e.read(65536)).get("error", {})
                known = {"model_not_found", "insufficient_quota", "invalid_api_key", "rate_limit_exceeded",
                         "invalid_request_error", "server_error", "no_available_channel", "channel_not_found"}
                receipt["provider_error_code"] = error.get("code") if error.get("code") in known else "unclassified"
                receipt["provider_error_type"] = error.get("type") if error.get("type") in known else "unclassified"
            except (ValueError, AttributeError, TypeError):
                receipt["provider_error_code"] = "non_json_error"
            return None, receipt
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            receipt.update(status="transport_error", error_type=type(e).__name__,
                           seconds=time.monotonic() - started)
            return None, receipt
        receipt.update(seconds=time.monotonic() - started, response_id=result.get("id"),
                       returned_model=result.get("model"), usage=result.get("usage", {}))
        if self.mode == "responses":
            text = "".join(part.get("text", "") for item in result.get("output", [])
                           if item.get("type") == "message" for part in item.get("content", [])
                           if part.get("type") == "output_text")
            complete = result.get("status") == "completed"
        else:
            choices = result.get("choices", [])
            text = choices[0].get("message", {}).get("content", "") if choices else ""
            complete = bool(choices and choices[0].get("finish_reason") == "stop")
        receipt["response_text"] = text
        if not complete:
            receipt["status"] = "incomplete_or_refused"
            return None, receipt
        try:
            parsed = json.loads(text)
        except (ValueError, TypeError):
            receipt["status"] = "invalid_json"
            return None, receipt
        receipt["status"] = "ok"
        return parsed, receipt

    def propose(self, context, protocol):
        system = ("You are developing an executable microscopy analysis workflow. Return JSON only. "
                  "Use the supplied development observations, never invent results. Change code, not the evaluator. "
                  "pipeline_code must define predict(sample, state, context), returning a finite 2D score "
                  "and boolean support matching image shape. Optional fit(samples, context) returns a dict of "
                  "numeric numpy checkpoint arrays. Only fit receives human/known training labels; inference "
                  "never receives labels or source names. Available libraries: numpy, scipy, skimage, torch. "
                  "No network, host filesystem, new packages or evaluation code is available. "
                  "policy_code must define plan(context) to propose future research actions from actual history. "
                  "Choose generate with no parents, tune/repair with one parent, evolve with two distinct parents. "
                  "Provide the specific mechanism, code intervention, expected result and falsifier. "
                  "Unknown/unannotated pixels are not confirmed negatives. Report training_steps=0 when not training. "
                  "Do not hard-code case identities, coordinates or answers; use observable image evidence. "
                  "All source is complete executable Python, not markdown or pseudocode.")
        value, receipt = self.request(system, context, PROPOSAL_SCHEMA)
        if value is None:
            return None, receipt
        try:
            proposal = Proposal.from_dict(value).validate(protocol)
        except (TypeError, ValueError, SyntaxError) as e:
            receipt.update(status="invalid_proposal", validation_error=str(e))
            return None, receipt
        return proposal, receipt


class FileExchangeModel:
    """External-model bridge with exactly one sealed input and response per call.

    A trusted broker may use a separately authorized isolated Codex session or
    another text-model service. It must preserve the raw response and its origin.
    This adapter does not pretend its waiting time or test fixtures are LLM work.
    """
    def __init__(self, directory, *, model, maximum_calls, timeout=1200):
        if not model or maximum_calls <= 0 or timeout <= 0:
            raise ValueError("Explicit model and positive call/time budgets required")
        self.directory = inside(Path(__file__).resolve().parents[2], directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.model, self.maximum_calls, self.timeout = model, maximum_calls, timeout
        self.calls = 0

    def propose(self, context, protocol):
        if self.calls >= self.maximum_calls:
            raise ModelUnavailable("External-model call budget exhausted")
        self.calls += 1
        folder = self.directory / f"call-{self.calls:03d}"
        folder.mkdir(exist_ok=False)
        request = {"model": self.model, "context": context,
                   "protocol": proposer_protocol(protocol), "response_schema": PROPOSAL_SCHEMA,
                   "instructions": "Return a single Proposal JSON object matching the schema. Implement complete "
                   "Python pipeline_code and policy_code. Follow required_action and required_parents. "
                   "predict(sample,state,context) returns score and boolean support of the original image shape. "
                   "fit(samples,context), if needed, returns numeric-array checkpoint state; only fit receives "
                   "human/known labels. plan(context) returns action, parents and focus. Libraries available: "
                   "numpy/scipy/skimage/torch. Do not import repository files or read any other tasks/history. "
                   "Use only this sealed request. No raw images or full labels have been supplied. Do not invent "
                   "observations/results. Preserve physically valid support and the frozen evaluation contract. "
                   "Use observable image features, not source names/case IDs/fixed annotation coordinates."}
        write_new(folder / "request.json", request)
        request_hash = file_hash(folder / "request.json")
        print(json.dumps({"event": "external_model_request", "directory": str(folder),
                          "request_sha256": request_hash, "model": self.model}), flush=True)
        started = time.monotonic()
        response = folder / "response.json"
        ready = folder / "response.ready.json"
        while not ready.exists():
            if time.monotonic() - started > self.timeout:
                return None, {"status": "external_model_timeout", "request_sha256": request_hash,
                              "requested_model": self.model, "call_index": self.calls}
            time.sleep(.25)
        # The marker is published only after the complete response is closed.
        try:
            marker = json.loads(ready.read_text())
            if marker["response_sha256"] != file_hash(response):
                raise ValueError("External response changed after publication")
            result = json.loads(response.read_text())
            if file_hash(folder / "request.json") != request_hash or result["request_sha256"] != request_hash:
                raise ValueError("External response is not bound to the sealed model input")
            if not result.get("origin") or result.get("model") != self.model:
                raise ValueError("Missing or mismatched external model provenance")
            if json.loads(result["raw_text"]) != result["proposal"]:
                raise ValueError("Broker proposal differs from the raw model response")
            proposal = Proposal.from_dict(result["proposal"]).validate(protocol)
        except (ValueError, KeyError, TypeError, OSError, SyntaxError) as error:
            return None, {"status": "invalid_external_response", "request_sha256": request_hash,
                          "requested_model": self.model, "call_index": self.calls,
                          "error_type": type(error).__name__, "error": str(error)[:1000]}
        receipt = {"status": "ok", "requested_model": self.model,
                   "returned_model": result.get("provider_returned_model"),
                   "transport": "sealed_external_model_exchange",
                   "origin": result["origin"], "request_sha256": request_hash,
                   "response_sha256": file_hash(response), "seconds": time.monotonic() - started,
                   "usage": result.get("usage"), "call_index": self.calls,
                   "raw_response_artifact": str(response)}
        return proposal, receipt
