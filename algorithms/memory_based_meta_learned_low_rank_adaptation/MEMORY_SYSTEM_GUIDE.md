# Memory System Implementation Guide (No Code)

This guide captures the agreed design for a simple memory system that stores and retrieves LoRA parameters when tasks recur. It is intentionally minimal and is meant to be implemented later without changing the high level behavior.

## Agreed Decisions
- Persist memory to disk as a single file: `memory.pt`.
- Memory stores and retrieves LoRA parameters only (not full model weights).
- Retrieval scores each stored LoRA by running the base prior model with that LoRA on collected steps and choosing the lowest loss.
- Accept a retrieved LoRA only if it improves over the base prior by a margin:
  - `base_loss - best_loss > max(rel_improve * abs(base_loss), abs_improve)`
  - Default: `rel_improve = 0.02`, `abs_improve = 0.01`
  - Minimum steps to evaluate: `min_steps = 50`
- No eviction policy for now (memory grows unbounded).

## Proposed File Layout
- New module: `memory.py` in this directory.
- New persistence file: `memory.pt` in this directory.

## Proposed API (Conceptual)
Create a small class that encapsulates persistence, storage, and retrieval.

- `save(lora_params, metadata)`
  - Inputs:
    - `lora_params`: dict of LoRA parameter tensors (A/B weights only)
    - `metadata`: dict with optional episode info (return, length, timestamp, env_id, etc.)
  - Behavior:
    - Append entry to memory list
    - Persist to `memory.pt` immediately or on a schedule

- `retrieve(steps, dynamics_model, base_parameters, min_steps, rel_improve, abs_improve)`
  - Inputs:
    - `steps`: collected transitions so far (see next section)
    - `dynamics_model`: for `compute_loss_with_parameters`
    - `base_parameters`: `dynamics_model.get_parameter_dict()` for the current prior
  - Output:
    - `None` if no match, otherwise a LoRA parameter dict (A/B weights only)
  - Behavior:
    - If `len(steps) < min_steps`: return `None`
    - Compute `base_loss` on the steps using `base_parameters`
    - For each stored LoRA:
      - Create `candidate_parameters = base_parameters` plus that LoRA
      - Compute `candidate_loss`
    - Select the lowest `candidate_loss`
    - Apply acceptance threshold vs `base_loss`

## Steps Format (Minimal, Compatible with Current Code)
Use the existing episode transition format already used in `trainer.py`:
- `self.episode_transitions` is a list of tuples: `(obs, action, next_obs)`
- For retrieval, take the most recent `N` transitions (e.g., last 50 or 100)
- Stack them into tensors:
  - `obs`: shape `(N, obs_dim)`
  - `act`: shape `(N, act_dim)`
  - `next_obs`: shape `(N, obs_dim)`
  - `delta = next_obs - obs`

This aligns with `DynamicsModel.compute_loss_with_parameters` which expects `(observation, action, delta)`.

## Persistence Format (`memory.pt`)
Recommend a single dict serialized by `torch.save`, for simplicity.

Suggested structure:
- `version`: int (start at 1)
- `items`: list of entries
  - Each entry:
    - `id`: unique string
    - `lora_state`: dict of tensors (A/B weights only)
    - `metadata`: dict (optional episode stats)
- `model_signature`:
  - `observation_dim`, `action_dim`, `lora_rank`, `lora_alpha`
  - Optional: parameter shapes for sanity checks

Rationale: a single file reduces bookkeeping and is faster to load.

## Integration Points (Trainer)
These are recommended touchpoints in `trainer.py`:

1. Initialization
- Create memory instance and load `memory.pt` if it exists.
- Keep `self.memory` on the trainer.

2. Retrieval Trigger
- During `_collect_steps` or in the step loop where transitions are appended:
  - Once `len(self.episode_transitions) >= min_steps`, call `retrieve` once.
  - If a LoRA is returned, set:
    - `self.online_adapted_parameters = base_parameters + retrieved_lora`
    - Keep `self.online_num_updates = 0` (so online adaptation can continue)
  - Add a boolean flag like `self._memory_retrieved` to avoid repeated retrieval in the same episode.

3. Save Trigger
- At episode end (inside `_reset_episode_state`), if a LoRA was adapted:
  - Save if `self.online_adapted_parameters` is not `None` and `self.online_num_updates > 0`.
  - Optionally gate on helpfulness (e.g., last update improved loss), but keep it simple initially.

## Scoring Details
- Use `torch.no_grad()` during evaluation for speed.
- Use the same normalization stats already set on `dynamics_model`.
- If normalization stats are missing, skip retrieval.
- Skip any stored LoRA whose parameter shapes do not match current model.

## Suggested Config Knobs (Optional)
If you want these configurable, add to `memory_meta_lora.yaml`:
- `memory_file`: default `memory.pt`
- `memory_min_steps`: default `50`
- `memory_rel_improve`: default `0.02`
- `memory_abs_improve`: default `0.01`

## Logging (Optional but Helpful)
When retrieval is attempted, log:
- `base_loss`, `best_loss`, `improvement`, `selected_id`
- Whether accepted or rejected by threshold

## Edge Cases
- No `memory.pt`: start empty, do nothing.
- Corrupted `memory.pt`: catch and warn, start empty.
- Unbounded growth: known issue for now, track file size in logs.

## Minimal Acceptance Test (Manual)
- Run one episode to ensure `memory.pt` is created and contains at least one entry.
- Start a new episode, collect 50 steps, and confirm retrieval attempts are logged.
- Verify that retrieved LoRA reduces loss vs base prior by the configured margin.

