"""Shared numerical training-trace check, independent of experiment drivers."""
import numpy as np


def trace_check(state, steps):
    required = {"audit_initial_parameters", "audit_final_parameters", "audit_loss_trace", "audit_steps"}
    if not required <= set(state):
        raise ValueError("Learner checkpoint lacks a checkable training trace")
    before = np.asarray(state["audit_initial_parameters"], dtype=float).ravel()
    after = np.asarray(state["audit_final_parameters"], dtype=float).ravel()
    losses = np.asarray(state["audit_loss_trace"], dtype=float).ravel()
    actual = np.asarray(state["audit_steps"]).reshape(-1)
    if actual.size != 1 or actual[0] != steps or losses.size != steps:
        raise ValueError("Training trace does not match requested steps")
    if not before.size or before.shape != after.shape or not all(np.isfinite(a).all() for a in (before, after, losses)):
        raise ValueError("Invalid training parameters or loss trace")
    delta = float(np.linalg.norm(after - before))
    if delta <= 1e-9:
        raise ValueError("No measured parameter update")
    return {"steps": steps, "parameters": before.size, "parameter_delta_l2": delta,
            "first_loss": float(losses[0]), "last_loss": float(losses[-1]),
            "note": "Trace is checked alongside the preserved executable fit code; it is not a backend attestation."}
