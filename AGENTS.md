# PrismMoves contribution instructions

PrismMoves owns evidence organisation, visual-language reporting, CLI ergonomics and PixelCue composition. Image decoding, statistics, model adapters and inference belong to PixelCue. Improve PixelCue's supported APIs rather than duplicating those capabilities here.

Read README.md and current PixelCue analysis documentation before changes. Update README.md alongside every changed feature. Run pytest, build wheel/sdist, validate versions and immutable assets, and verify release downloads. Use synthetic fixtures. Do not commit private inputs, generated reports, credentials, weights or coordination records. No model download is needed for tests. Model interpretations and unknown evidence must remain explicit.

All downloaded weights and model-specific caches belong in the owning project's models/ directory. Add models/ to .gitignore and exclude it from distributions. Honour explicit download/licence permission; do not silently move existing models or change references without verification.

## Full ecosystem and release preflight

Before every new workflow run, check the latest stable official release and observe the actual runtime used by the job. Follow [the shared preflight](https://github.com/kieransimkin/DanceFlow/blob/main/docs/workflows/release-preflight.md) and this project's [people-and-agent guide](docs/ECOSYSTEM.md). Review the complete relevant capability inventory rather than only a familiar command. Consider every DanceFlow component against the actual task; use and compose all applicable supported capabilities, and improve a concrete reusable gap in its canonical repository. Record why existing capabilities suffice when no change is justified. No catalogue checklist expands the task, authorises model downloads/training, or proves integration. Preserve exact source/package identity, clocks, model-local storage, privacy, costs, immutable releases and separate deployment gates.

## Preflight temporary-directory access

Before any long, costly, model-backed, media-rendering, build, test, packaging or release command that creates private or atomic temporary directories, run a disposable preflight with the exact runtime, user or sandbox, and intended output parent. The probe must create a unique directory, write and read a file, atomically rename it, then remove the file and directory.

Keep temporary data and caches under the owning project on `Z:`; model weights and model-specific caches remain under that project's ignored `models/`. If the probe returns `WinError 5`, access denied, a rename failure or a cleanup failure, stop before the expensive command. Preserve the exact error and use the documented narrowly approved unrestricted recovery from the outset only when the task requires it. Use a fresh output directory and do not repeat the already-failing full job in the restricted shell.

After any unrestricted recovery, verify that the final output is readable and writable from the normal managed environment. If it is not, copy the completed bytes into a pre-created project-local directory that passed the probe, verify hashes, and retain the original failure and recovery evidence. Do not edit global ACLs, switch caches to `C:`, terminate unrelated processes or claim the underlying permission problem is fixed. Repositories that own temporary-directory creation should provide and test a reusable preflight helper rather than relying only on this instruction.