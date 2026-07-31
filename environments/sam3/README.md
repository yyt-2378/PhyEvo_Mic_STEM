# Isolated SAM 3 runtime

SAM 3 is not installed into the main `tfree-detection` environment. The main
process owns physics candidates and acceptance gates; this Python 3.12
environment only returns boundary proposals through `sam_request.json` and
`sam_response.json`.

Protocol v2 requires one output for every
`(candidate_id, prompt_role, ensemble_index)` request. The worker must not
merge candidates into an image-wide mask. For interface tasks, `prompt_role`
is `domain_a` or `domain_b`; the main process—not the worker—derives their
shared interface and applies candidate-local fallback.

Create a project-local environment (example):

```bash
python3.12 -m venv .venvs/sam3
.venvs/sam3/bin/python -m pip install --upgrade pip
# Install the PyTorch build appropriate for the local CUDA runtime first.
.venvs/sam3/bin/python -m pip install \
  -r environments/sam3/requirements-sam3.txt
```

The official checkpoint is access-gated. Accept its terms and authenticate in
the isolated environment, or place an explicitly authorized checkpoint inside
the project and set `stage2.sam3.checkpoint_path`. Do not commit credentials or
checkpoints.

Then set:

```yaml
stage2:
  sam3:
    enabled: true
    inference_mode: isolated_process
    worker_python: .venvs/sam3/bin/python
```

The pinned repository revision is recorded in `requirements-sam3.txt`.
