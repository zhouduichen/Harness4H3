# Wan RSI Hybrid Run Recipes

These recipes are portable within the authorized server directory. Stage the code snapshot, model, CLIP model, Wan configuration, validation manifest, and complete real-CUDA execution profile under the `ai4mbse` root. Put SparseVideo at `ai4mbse/third_party/SparseVideo` at commit `8cd70fd3315a32bc9ac6e6ad37f57ad7cefeda0d`; its Wan 1.3B SVOO sparsity profile is loaded relative to that package. Run the commands from this recipe directory with `PYTHONPATH=../..:../../third_party/SparseVideo`; model assets, the SVOO package/profile, and experiment outputs then resolve inside `ai4mbse`. Do not substitute a path outside `ai4mbse`.

The comparison recipe uses Wan2.1-T2V-1.3B at 832×480, 81 frames, and 16 denoising steps. The validation set has six fixed prompt/seed pairs. The evaluator uses one warmup and three timed repeats per pair.

Prepare `../../models/wan21-t2v-1.3b`, `../../models/clip-vit-large-patch14-336`, and `../../experiments/wan-rsi-hybrid-20261018/profile.json` inside `ai4mbse` before running. Verify that the profile checkpoint SHA matches the model snapshot. The model and profile are not copied or inspected by these local recipes.

Example from `ai4mbse/configs/wan-rsi-hybrid-20261018`:

```sh
PYTHONPATH=../..:../../third_party/SparseVideo python ../../tools/run_wan_evaluator_once.py --config dense.yaml
```

Run `svoo.yaml`, `magcache.yaml`, and `svoo-magcache.yaml` sequentially on the same idle GPU. Each writes to a distinct path under `experiments/wan-rsi-hybrid-20261018/`. `rsi.yaml` is the matched evolutionary run configuration; run it only after the local source snapshot and evaluator dependencies have been staged under this root. The recipes do not launch remote jobs.

```sh
PYTHONPATH=../..:../../third_party/SparseVideo python -m harness4h3 evolve-compression --config rsi.yaml --rounds 2
```
