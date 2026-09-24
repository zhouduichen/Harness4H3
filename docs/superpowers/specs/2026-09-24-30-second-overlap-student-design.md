# 30-Second Overlap-Window Student Design

## Goal

Upgrade the existing real-H3 1–2B Student from a stable five-second clip
contract to a continuous approximately 30-second video contract. The final
artifact must be 708–732 decoded frames at 24 FPS, 256x256, decodable, and
generated from overlapping latent windows rather than independent clips.

The F1 -> F2 -> F3 training and checkpoint lineage remains the source of truth.
Thor/Orin deployment, phone real-time execution, architecture search, and LLM
code editing are outside this design.

## Approaches considered

1. **One 216-latent Student window.** This is the smallest code change, but it
   multiplies attention, activation, and VAE memory while training remains
   five-second-shaped. It also offers no explicit boundary mechanism, so it is
   rejected for the current 1–2B memory and continuity requirements.

2. **Fixed Student window with latent continuation (recommended).** Keep the
   existing 36-latent Student graph and real H3 Teacher, generate seven windows
   with a deterministic six-latent overlap, anchor each new window to the
   previous window's tail during every denoising step, decode each window with
   the real H3 VAE, and crossfade only the decoded overlap. This keeps the
   architecture/checkpoint lineage intact while making continuity explicit and
   testable.

3. **Add a recurrent/global-memory Student architecture.** This could improve
   long-range semantic persistence, but it changes the graph and checkpoint
   contract, requiring a new model lineage and a separate architecture search.
   It is deferred until the overlap-window path has measured semantic drift.

## Contract and temporal geometry

The H3 VAE temporal mapping is authoritative and is shared by cache generation,
runtime, and verification. For a 36-latent window it emits 120 frames; for a
six-latent overlap it emits 18 frames. Seven windows therefore compose as:

```text
120 + 6 * (120 - 18) = 732 decoded frames = 30.5 seconds at 24 FPS
```

The model-facing target remains `latent_frames=36`. The long-video contract
adds `window_count=7`, `overlap_latent_frames=6`, and `stride=30`. The final
video contract accepts 708–732 frames, 29.5–30.5 seconds, 24 FPS within the
existing tolerance, and 256x256. The contract digest includes all window
parameters.

## Runtime data flow

1. Load one existing Student checkpoint and one prompt conditioning tensor.
2. Sample window zero with the existing Student flow sampler.
3. For each later window, initialize deterministic noise from a window-specific
   seed. During each Student step, blend the first six latent frames toward the
   previous window's final six clean latent frames. The final step clamps that
   prefix exactly, so the next window sees an actual continuation state rather
   than an unrelated clip.
4. Decode every window through the existing tiled H3 VAE decoder. Blend the
   first 18 decoded frames of each new window with the preceding 18 output
   frames, then append only the non-overlapping suffix.
5. Persist window count, stride, overlap, per-window seeds, prefix hashes, and
   a continuity report next to the MP4. A long-video evaluation is invalid if
   this continuation trace is absent or marked simulated.

The same continuation sampler is used by the real Teacher cache worker. Teacher
cache items remain authenticated with `teacher_generated=true`,
`synthetic_latent=false`, and `offline_simulation=false`. Student training
cycles over all authenticated prompt/seed/window cache items instead of the
legacy single `00000000.pt` item; this is the direct semantic-quality change.

## Semantic quality and evaluation

The evaluation manifest remains multi-prompt and multi-seed and is shared by
Teacher and Student. Long-video CLIP evaluation uses stratified samples across
the full 30 seconds rather than four short-clip samples. The semantic score is
the mean prompt similarity with a lower-quantile drift penalty, and the
baseline uses exactly the same sampler and metric. F1 stays structural, F2
adds semantic verification, and F3 keeps full hardware/quality evidence.

The evaluator rejects black frames, non-finite pixels, decode truncation,
wrong resolution/FPS/duration, and all-frame repetition. It also requires the
long-video continuation metadata for a multi-window contract. `offline_simulation`
is false on every real Teacher, training, and Student result path.

## Lineage and failure handling

No parent fallback is introduced. F1, F2, and F3 continue to pass the selected
checkpoint and its SHA-256 into the next fidelity. A failed fidelity cannot
start the next fidelity. A failed long-video decode or semantic gate cannot
become Parent. Edge/Thor/Orin checks are not part of this acceptance contract;
the normal model remains promotable based on server-side long-video evidence.

Memory pressure may change scheduling, tiled decode, or window serialization,
but it must never lower the window count, overlap, frame range, or real-Teacher
requirement.

## Verification

Unit tests cover H3 temporal geometry, overlap planning, prefix anchoring,
decoded crossfade composition, metadata requirements, and 708–732 frame
validation. Worker contract tests cover multi-prompt cache cycling and real
flag propagation. Integration tests cover the 30-second config and remote
command arguments. A real remote smoke must run multiple prompts and seeds
before any full F1 -> F2 -> F3 campaign is promoted.
