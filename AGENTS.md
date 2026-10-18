# PrismMoves contribution instructions

PrismMoves owns evidence organisation, visual-language reporting, CLI ergonomics and PixelCue composition. Image decoding, statistics, model adapters and inference belong to PixelCue. Improve PixelCue's supported APIs rather than duplicating those capabilities here.

Read README.md and current PixelCue analysis documentation before changes. Update README.md alongside every changed feature. Run pytest, build wheel/sdist, validate versions and immutable assets, and verify release downloads. Use synthetic fixtures. Do not commit private inputs, generated reports, credentials, weights or coordination records. No model download is needed for tests. Model interpretations and unknown evidence must remain explicit.

All downloaded weights and model-specific caches belong in the owning project's models/ directory. Add models/ to .gitignore and exclude it from distributions. Honour explicit download/licence permission; do not silently move existing models or change references without verification.

## Full ecosystem and release preflight

Before every new workflow run, check the latest stable official release and observe the actual runtime used by the job. Follow [the shared preflight](https://github.com/kieransimkin/DanceFlow/blob/main/docs/workflows/release-preflight.md) and this project's [people-and-agent guide](docs/ECOSYSTEM.md). Review the complete relevant capability inventory rather than only a familiar command. Consider every DanceFlow component against the actual task; use and compose all applicable supported capabilities, and improve a concrete reusable gap in its canonical repository. Record why existing capabilities suffice when no change is justified. No catalogue checklist expands the task, authorises model downloads/training, or proves integration. Preserve exact source/package identity, clocks, model-local storage, privacy, costs, immutable releases and separate deployment gates.
