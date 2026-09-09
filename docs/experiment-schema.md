# Experiment Record Schema

Each line in `experiments.jsonl` describes one model-optimization experiment rather than one video-generation task. Records are append-only, fsynced, and redact credential-like keys.

Required fields are `experiment_id`, `session_id`, `target_profile_id`, controller provider/model, parent and optional child model IDs, ModelState digest, diagnosis, full ExperimentPlan, operator execution result, training-log/artifact paths, measured cost, independent evaluation, stable failure type, keep/drop decision, Pareto update and UTC creation time.

Model files are not embedded. A successful model modification always produces `parent_model_id → experiment_id → child_model_id`. Failed validation or execution has no child ID and leaves the active pointer unchanged. Candidate files are created exclusively and cannot be overwritten; `active.json`, `front.json` and `session.json` use atomic replacement.
