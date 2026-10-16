# PrismMoves

PrismMoves turns **PixelCue analysis** into a broad, evidence-labelled visual-language profile of an image. It belongs to the [DanceFlow ecosystem](https://kieransimkin.co.uk/danceflow/).

PixelCue owns decoding, measurements, model loading, inference and tag extraction. PrismMoves calls PixelCue's Python API or local HTTP service, organises its receipts along the visual-identity assessment axes, and writes JSON, Markdown and a self-contained HTML report. It contains no competing image-analysis engine or model implementation.

## Install and run

Requires Python 3.13. Install PrismMoves from the wheel on its GitHub release. The HTTP client has no third-party dependencies. For an in-process backend install `prismmoves[local]`, which requires PixelCue 0.9.0 or later and its dependencies. If using release wheels before a new package is available in a registry, install the downloaded wheel with `python -m pip install path/to/prismmoves-0.1.0-py3-none-any.whl`.

With an existing PixelCue service:

```powershell
pixelcue-server --host 127.0.0.1 --port 8765
prismmoves cover.png --pixelcue-url http://127.0.0.1:8765 --out analysis/cover
```

Or use the installed PixelCue Python API:

```powershell
prismmoves cover.png --out analysis/cover
prismmoves cover.png --models clip dinov2 blip ocr --out analysis/cover-models
```

`--out` must be a new directory. Originals are never modified. Inputs are explicit files, with a maximum of 64 per invocation; there is no filesystem scan. Multiple images receive individual profiles and a shared evidence ledger. Recurrence, independent lineages, clusters and outliers require reviewed comparison evidence; PrismMoves does not invent a unified brand identity.

## What PixelCue contributes

| Evidence | Mechanism and scope |
|---|---|
| Colour | Deterministic sampled Lab k-means palette; hue histogram; HSV saturation; linear sRGB luminance distribution and palette-pair contrast |
| Texture | Grayscale entropy; 8-neighbour local binary patterns; symmetric 16-level grey-level co-occurrence statistics |
| Geometry/composition | Edge density and orientation histogram; horizontal/vertical mirror error; spatial luminance/saturation/edge grid |
| Frequency and signal novelty | Mean-removed 2D Fourier power bands; spectral residual centroid and spatial grid |
| CLIP | Prompt-relative medium, material, light, tone and setting hypotheses; image embedding and exact prompt bank |
| DINOv2 | Visual embedding for later same-model reference comparison; no labels invented from the vector |
| BLIP | Caption hypothesis |
| Tesseract OCR | Text hypotheses, engine confidence and boxes in analysis-raster coordinates |
| Existing PixelCue VLMs | Calls the existing keyword service, e.g. SmolVLM/JoyCaption; no copied tagger |

All optional models are selected explicitly. Missing dependencies, absent cached weights and failed inference produce individual `unavailable` receipts; statistical measurements survive. The default run uses no neural model. `--models clip dinov2 blip` uses locally cached checkpoints only. `--model-path clip=PATH` selects a local model directory; no remote custom code is enabled. `--device cpu` is the default, with explicit CUDA/MPS options.

```powershell
prismmoves cover.png --pixelcue-url http://127.0.0.1:8765 --models clip dinov2 --out analysis/local-models
prismmoves cover.png --pixelcue-url http://127.0.0.1:8765 --pixelcue-models smolvlm-256m --allow-download --out analysis/tagged
```

**Download consent:** `--allow-download` explicitly permits downloads for the selected models. Existing PixelCue VLM taggers may download on first use, so they are not invoked without this flag. No hosted inference or image upload is performed. HTTP is restricted to a loopback origin; redirects and environment proxies are disabled. Paths refer to files visible to that local PixelCue service. Heavy model work can exceed the client's 300-second request timeout; a timeout is not proof that server work stopped.

OCR requires PixelCue's `analysis` extra (`pytesseract`) and a separately installed Tesseract executable. Neural adapters use PixelCue's Torch/Transformers installation. Checkpoint licences and suitability require review before downloading or using weights.

## Evidence contract

`analysis.json` retains PixelCue's source SHA-256, original and analysed dimensions, EXIF orientation handling, alpha matte, ICC handling, measurement methods, model checkpoint/revision where available, dependency versions, prompt bank and per-model completion/failure state. `report.md` and `report.html` distinguish observations, model interpretations, unknowns and proposed next evidence. `SHA256SUMS.txt` covers all three outputs. `--role` and `--lineage` retain supplied labels; they never imply verified approval or independent authorship. Local source paths are not copied into the source ledger; model paths and service errors may still be private and require review before sharing reports.

Exit codes: `0` all inputs and requested models completed; `2` usable report with missing inputs/models; `1` backend/configuration failure. Invalid CLI usage exits with argparse's code `2` and creates no report.

## Interpretation limits

Measurements describe a bounded raster (default longest edge 768, maximum 2048), with alpha composited over `--background` (default white). Palette centres are approximations. Resizing, ICC handling and matte choice can change the measurements. Animated inputs use frame zero only. Spectral residual is a signal novelty proxy, not an eye-tracking or subject detector. Palette-pair contrast cannot certify actual text/background contrast. CLIP similarity is prompt-relative, not probability or calibrated confidence. Captions, OCR and tags can be wrong. No model establishes artist intent, provenance, rights, personal identity, audience response or aesthetic quality.

The assessment axes also include typography, narrative, representation, motion, accessibility and provenance. Where no adequate evidence exists, the report records unknowns. Visual review remains necessary.

## Development and Potential problems

The portable `skills/prismmoves/SKILL.md` is included in the source and wheel under `share/prismmoves/skills/prismmoves`. It routes image analysis to PixelCue and collection conclusions to the visual-identity workflow. Install it into your agent's skill directory when useful; package installation does not automatically activate an agent skill.

```powershell
python -m pip install -e '.[dev]'
python -m pytest -q
python -m build
python -m twine check --strict dist/*
```

Tests use synthetic protocol fixtures, including a loopback HTTP server; they do not download weights. Real PixelCue engine/service tests live in PixelCue's repository. In a restricted Windows environment set `TEMP`/`TMP` and pytest's `--basetemp` to writable workspace paths and disable pytest's cache provider if its atomic cache directories are blocked. A service without `/health.visual_analysis` is an old PixelCue and must be upgraded. Partial results require checking each model receipt before making semantic claims.
