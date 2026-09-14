# Experiment Record Schema

Each line in `experiments.jsonl` describes one model-optimization experiment rather than one video-generation task. Records are append-only, fsynced, and redact credential-like keys.

Required fields are `experiment_id`, `session_id`, `target_profile_id`, controller provider/model, parent and optional child model IDs, ModelState digest, diagnosis, full ExperimentPlan, operator execution result, training-log/artifact paths, measured cost, independent evaluation, stable failure type, keep/drop decision, Pareto update and UTC creation time.

Model files are not embedded. A successful model modification always produces `parent_model_id → experiment_id → child_model_id`. Failed validation or execution has no child ID and leaves the active pointer unchanged. Candidate files are created exclusively and cannot be overwritten; `active.json`, `front.json` and `session.json` use atomic replacement.

## Remote experience records

`var/remote-h3/experience.jsonl` is separate append-only memory for remote
trainer and benchmark evidence. Each `ExperienceRecord` includes source URI
and SHA-256, parent/child IDs, operator arguments, training metrics,
independent evaluation, reward terms, decision, and provenance. Checkpoint
bytes are never embedded.

`training_only_unvalidated` means Q/L/M/E evidence is incomplete and the child
cannot be active. `evaluated_candidate` means a benchmark was recorded but the
child was not promoted. `accepted` and `rejected` are explicit gate decisions;
`failed` is retained training/infrastructure negative evidence. Missing metrics
are `null`, never zero. The default `structural_proxy` quality scope is an
engineering signal, not semantic video-quality evidence.
