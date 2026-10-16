---
name: prismmoves
description: Generate an evidence-labelled visual-language profile of local images by calling PixelCue through PrismMoves. Use for broad image understanding, measured colour/texture/composition evidence, and explicit optional visual-model analysis; use assess-visual-identity for collection-level identity conclusions.
---

# PrismMoves

![Synthetic input and actual PixelCue measurement figure showing the reporting handoff](assets/visual-profile-example.png)

Generated measurement figure from PixelCue 0.9.0; explanatory output, not an interface screenshot or semantic-accuracy evaluation.

Use PrismMoves to organise PixelCue analysis into JSON, Markdown and self-contained HTML. PixelCue owns image decoding, statistics, model adapters and inference. Improve PixelCue's supported APIs when reusable analysis is missing; do not duplicate its engine in PrismMoves or campaign scripts.

Verify installed/released identities and read the current README. PixelCue 0.9.0+ is required. Run `prismmoves IMAGE --out NEW_DIRECTORY` using the in-process PixelCue API, or add `--pixelcue-url http://127.0.0.1:8765` for the local service. Inspect `/health.visual_analysis` before service calls. My Songs uses `Tools/PrismMoves.ps1` and `Workflows/prismmoves-visual-analysis.md`.

Keep all downloaded weights and model-specific caches in the owning project's `models/`, ignored with `models/` in `.gitignore`. Use `--models-dir PROJECT/models` or the configured project model root. PixelCue's explicit installer can populate selected profiles with revision/hash manifests. Do not silently download to a global/user cache or migrate/delete existing weights without checking consumers.

Select only useful optional `--models clip dinov2 blip ocr`. Weights load locally by default; no download is needed for statistics. `--pixelcue-models PROFILE` delegates to existing PixelCue taggers and requires explicit `--allow-download`, since their current loader can download on first use. Preserve applicable model/licence/privacy/cost permission. Never enable downloads just to eliminate unavailable receipts. OCR needs pytesseract and a Tesseract executable.

Retain the source hash, analysis dimensions/matte/ICC treatment, individual model receipts and complete failure ledger. Exit 2 means a report contains unavailable inputs or requested models; review those receipts. Original images stay unchanged. Reports and model paths may be private; no upload/publication is implied.

Inspect the actual image alongside the report. Distinguish measured observations, model interpretations, proposals and unknowns. CLIP cosine is prompt-relative, caption/OCR/tags can hallucinate, spectral residual is signal novelty, and palette contrast cannot certify actual text contrast. No analysis establishes intent, rights, provenance, consent, audience response or aesthetic quality.

One image supports an individual profile. For identity rules across a collection use assess-visual-identity: preserve supplied roles, canonical authority, dates and independent lineages; review clusters before outliers. Same-raster flags and measured distances are evidence, not proof of independent authorship or a shared identity.
