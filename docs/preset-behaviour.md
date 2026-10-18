# Preset behaviour: temporal first, semantics selectively

The single-image PrismMoves CLI is unchanged. The new `prismmoves-preset` entry
point composes PixelCue's reusable sequence and model-major batch APIs. It does
not implement a second image decoder, optical-flow engine or neural adapter.

## Recommended path

FrameRudiments captures ordered frames with input/output SHA-256 bindings and
separate conditions. PixelCue first measures every bounded raster and reliable
adjacent motion. PrismMoves then selects representative images and, only then,
requests expensive interpretation. A provisional numerical description is saved
before model inference. A final short paragraph combines measured motion,
supported visual interpretations and observed artefacts; every claim keeps its
frame/transition references and evidence class.

Defaults are a maximum of **24 unique semantic images per preset**, CLIP and
BLIP only, CPU, 256-pixel temporal measurements and 384-pixel semantic rasters.
The budget is a ceiling, not a quota. Uniform white/black/flat-colour frames stay
in numerical evidence but are not captioned. Exact duplicates reuse inference;
near-duplicates do not inherit another frame's caption. Coverage is allocated
across conditions before visual novelty. A warm-up allowance de-prioritises
startup-only frames. An exhausted budget or unrepresented condition is explicit.
DINOv2, OCR and keyword tagging remain opt-in, not mandatory costs for a
visualiser description. Models are processed model-major: one load per selected
model for the bounded batch, rather than one load for every image.

```powershell
python -m prismmoves.preset_pipeline analyse --captures captures.json --out analysis/preset --models-dir models
python -m prismmoves.preset_pipeline analyse --captures captures.json --out analysis/numeric-only --models-dir models --budget 0 --models
```

`captures.json` is FrameRudiments' retained capture manifest. An authored suite
may contain multiple covers and programme/silence controls; conditions are
never concatenated into one false motion sequence. The files must be available
locally and their hashes must match. No hosted model or image upload is used.
Neural weights are local-only unless `--allow-download` is explicitly given.
This API requires the coordinated PixelCue sequence update (0.9.1.dev1 here).

## Measurements and limits

The initial motion model uses forward/backward Lucas-Kanade feature tracking,
a robust similarity fit and forward/backward-checked dense flow residuals.
It can establish supported rotation, translation, expansion/contraction and
local deformation; low texture, failed tracks, dimension changes and gaps above
0.25 seconds produce unknowns, not guessed motion. Centre-relative translation
avoids calling a rotating image a drifting image. Brightness periodicity needs
regular dense sampling, at least two seconds and at least three observed cycles.
A single fade is not a pulse train. Static source-raster controls take precedence
when describing preset motion; otherwise the description explicitly refers to
output motion whose moving-source contribution is not isolated.

Captions and CLIP scores remain hypotheses. The visualiser prompt bank describes
spatial motifs, light and surface rather than forcing every image into generic
'outer space', '3D render' or 'ominous' categories. A spatial spiral does not
prove rotation. One-off captions and irrelevant safety/booru tags do not become
style facts. White washouts, near-black intervals and measured loss of source
edge detail are reported separately from intent. This update does not certify
banding, aliasing, feedback trails, beat synchronisation, aesthetic quality,
flash safety or motion comfort. Those require additional suitable evidence.

The analysis is bounded to 32,768 frame records and 512 conditions; semantic
batches are at most 64 images. Raster memory is adjacent-frame bounded, while
compact descriptors/receipts remain proportional to the explicit frame budget.
These limits are not a guarantee that every hardware configuration is fast.

## Caching and interrupted work

`temporal.json`, its digest and `semantic-plan.json` are durable before models
start. Matching captures are checked again on resume; missing/changed bytes are
not concealed by caches. Semantic caches live under project-local
`models/analysis-cache/preset-semantics/`. Keys include exact image hashes,
preprocessing, prompt-bank/code identity, package versions and model manifests.
Manifest file sizes/mtimes guard ordinary local checkpoint mutations; weights
are verified by PixelCue's explicit installer rather than rehashed per preset.
Unpinned checkpoints and heterogeneous keyword/OCR backends do not receive
indefinite cross-run caching. Failed or incomplete model receipts are retried.

Changing analysis settings in the same direct-API output is rejected rather
than mixing experiments. FrameRudiments' driver uses separate analysis attempts
and shared semantic caches, while retaining matching capture runs. A limited
but completed description is different from an inference process failure.

## Reuse the reports you already paid to generate

```powershell
python -m prismmoves.preset_pipeline legacy --reports ../FrameRudiments/visualiser-analysis/full-catalogue/reports --out analysis/existing-descriptions
```

This scans every per-preset JSON, not the potentially stale catalogue index,
verifies compressed PixelCue receipt hashes, selects from existing evidence and
writes a new searchable HTML atlas. It performs **no new model inference** and
**does not invent dense temporal data from four still frames**. Earlier originals
are never overwritten. A selection count in this mode is a retrospective plan,
not a measured live speed-up. Inspect legacy model paths and rights before
redistributing original receipts; new FrameRudiments exports redact local paths.

## Development

`python -m pytest -q tests` runs portable PrismMoves protocol tests. With the
coordinated PixelCue update and its sequence dependencies installed, also run
`python -m pytest -q integration`. Native temporal checks use synthetic fixtures
and fake semantic adapters; pretrained-model accuracy needs separate validation.
