# Feasibility: YOLO11 support (starting with `facen.onnx`)

**Status: research only. Nothing in this document is implemented.**

Verdict: **straightforward — roughly 100 lines of new decode code.** Most of the
machinery already exists in the addon, and the dynamic-shape work done for
`yolo26n-obb` turns out to be exactly what makes this family fast.

## What `facen.onnx` actually is

It is not a YOLO26 model. Its ONNX metadata reads:

```
description = Ultralytics YOLO11n model trained on .../dataset.yaml
task        = detect
names       = {0: 'face'}
imgsz       = [640, 640]
args        = {..., 'dynamic': True, 'nms': False, ...}
input        images  float32  ['batch', 3, 'height', 'width']
output       output0 float32  ['batch', 5, 'anchors']
```

So: YOLO11n, one class, **no NMS in the graph**, dynamic input shape, and the
classic raw YOLO detection head. That last point is the whole story — it is a
different decode from every model currently supported.

## Two model families

Surveying everything bundled in the reference project splits cleanly in two:

| model | task | NMS in graph | output |
|---|---|---|---|
| yolo26n, yolo26n-v | detect | yes | `[1, 300, 6]` |
| yolo26n-obb | obb | yes | `[batch, 300, 7]` |
| yolo26n-pose | pose | yes | `[1, 300, 57]` |
| yolo26n-seg | segment | yes | `[1, 300, 38]`, `[1, 32, 160, 160]` |
| yolo26n-depth | depth | – | `[1, 1, 640, 640]` |
| **facen** | **detect** | **no** | **`[batch, 5, anchors]`** |
| yolo11n, yolo11s | detect | no | `[batch, 84, anchors]` |
| yolo11n-v, yolo11s-v | detect | no | `[batch, 14, anchors]` |
| yolo11n-pose, yolo11s-pose | pose | no | `[batch, 56, anchors]` |
| yolo11n-obb | obb | no | `[batch, 20, anchors]` |

The implemented (YOLO26) family is row-major with one row per surviving
detection. The YOLO11 family is **channel-major over every anchor**, with no
suppression applied:

| | YOLO26 | YOLO11 |
|---|---|---|
| shape | `[1, 300, K]` | `[batch, C, anchors]` |
| indexing | `data[row * K + field]` | `data[channel * anchors + anchor]` |
| box form | corner `x1,y1,x2,y2` (detect) | centre `cx,cy,w,h` |
| NMS | in-graph | **caller's job** |
| decode | threshold + sort | argmax + threshold + NMS |

`facen` is therefore not a special case; it is one instance of the legacy raw
head, and the same decoder serves the rest of the family.

## Verified layout

Confirmed by running the model and inspecting the tensor, not by reading docs:

- Shape `[1, 5, 8400]`. The anchor count is `80² + 40² + 20² = 8400`, the usual
  strides 8/16/32 at 640x640 — so it varies with input size and must be read
  from the tensor rather than assumed.
- Channels 0–3 are `cx, cy, w, h` in model input-plane pixels.
- Channel 4 is the class score, **already in [0,1]** — Ultralytics applies the
  activation during export, so no sigmoid is needed on this path.
- `numClasses == C - 4` holds across the whole detect family: facen 1,
  yolo11n/yolo11s 80, yolo11n-v/yolo11s-v 10 (each cross-checked against the
  count of entries in the `names` metadata).

Decoded manually with a 0.5 score threshold and 0.45 IoU NMS, `facen` finds
**16 faces** in the bundled 1024x680 group photo, top scores 0.852 / 0.851 /
0.849, every box tight on a face. The decode is correct; only the C++ port
remains.

## What already exists and can be reused

| piece | status |
|---|---|
| Letterbox preprocessing (NCHW, RGB, [0,1]) | identical, reuse as-is |
| Dynamic input shape pinning (`AddFreeDimensionOverrideByName`) | done for OBB, applies unchanged |
| `nmsPerClassIndices` / `nmsPerClass` | already ported from the reference for segmentation |
| `Detection`, class names from ONNX metadata, `classColor` | reuse |
| `drawDetections()` | reuse |
| `ModelThread`, CoreML backend, per-configuration model cache | reuse |

## What is genuinely new

1. Channel-major indexing — `data[c * numAnchors + a]` rather than row-major.
2. Centre form to corner form, `cx,cy,w,h` → `x1,y1,x2,y2`.
3. Argmax across the `C - 4` class channels per anchor, keeping the best class
   and its score.
4. Wiring the existing NMS (score threshold, IoU threshold, top-k).

Estimate: ~100 lines in `ofxYolo26Detector.cpp`, plus a handful in the header.

## Performance

The dimension pinning added for `yolo26n-obb` is what unlocks this family.
Median of 10 inferences at 640x640, Apple Silicon:

| model | CPU | CoreML, axes free | CoreML, axes pinned |
|---|---|---|---|
| facen | 26.8 ms | 27.6 ms | **2.8 ms (9.6x)** |
| yolo11n | 28.0 ms | 28.3 ms | **4.4 ms (6.4x)** |

Left dynamic, CoreML gives nothing — the same trap `yolo26n-obb` fell into.
Pinned, `facen` would be **the fastest model in the addon at 2.8 ms**. The
reason is the mirror image of the decode work: having no NMS in the graph is
what CoreML likes, and it is exactly what pushes the NMS cost onto us.

## Recommended design

Extend the existing `Detector` to dispatch on output shape rather than
introducing a separate class. The two shapes are unambiguous — `[1, 300, 6]`
versus `[1, C, N]` with `N ≥ 8400` — so a single class can serve both families
and `Detector::setup()` would simply work with either model. This also matches
how the reference app handles it, with one detect session and a shape check.

Doing so delivers the reference README's "backwards compatible" claim for
`yolo11n`, `yolo11s`, `yolo11n-v` and `yolo11s-v` at the same time, since they
differ only in class count.

## Risks

All minor, all with known mitigations:

- **Layout ambiguity.** Some Ultralytics exports ship `[1, anchors, C]` rather
  than `[1, C, anchors]`. Resolve by treating the smaller of the two dimensions
  as channels — `C ≤ 84` and `anchors ≥ 8400` never overlap in practice. The
  reference's `decodeYOLO_or_OBB` makes the same check.
- **Anchor count depends on input size.** Read it from the output tensor at
  runtime; never hardcode 8400.
- **Decode cost moves to the CPU.** The argmax is `C × anchors` float reads per
  frame — about 706k for an 80-class model, well under a millisecond, but no
  longer free the way the end2end path is. Negligible for `facen` at 5 × 8400.
- **Single-class degeneracy.** With one class, per-class NMS reduces to plain
  NMS. Correct, but worth a test so the class-bucketing path is not silently
  untested.

## Out of scope for face support, but adjacent

The same channel-major decoder extends to the remaining YOLO11 variants with
modest additions, both layouts confirmed against the models:

- `yolo11n-pose`, `yolo11s-pose` — `C = 56 = 4 + 1 + 17 × 3` (box, objectness,
  17 keypoints). Needs the keypoint block and would feed the existing
  `PoseResult`.
- `yolo11n-obb` — `C = 20 = 4 + 15 + 1` (box, 15 DOTA classes, angle). Needs
  rotated-box NMS to be strictly correct, since the current axis-aligned IoU
  would over-suppress crossed boxes.

These are separate work from face detection and are listed only so the shape of
the remaining effort is on record.
