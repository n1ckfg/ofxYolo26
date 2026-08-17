# ofxYolo26 — build report

Depth, pose and instance segmentation are all implemented, build, and are
verified against the Python onnxruntime reference. GPU/ANE acceleration is on
by default via CoreML, worth 2.6x–9.3x.

## Structure

`libs/ofxYolo26/`, layered so each new task is a decoder on top of shared plumbing:

| file | role |
|---|---|
| `ofxYolo26Types` | `Letterbox` (source ↔ model-input coordinate mapping), `Settings`, `ResizeMode` |
| `ofxYolo26Preprocessor` | `ofPixels` → planar RGB NCHW float32 `[0,1]`, the layout every YOLO26 export wants |
| `ofxYolo26Detection` | `Detection`, per-class NMS, class colours |
| `ofxYolo26Model` | subclasses `ofxOnnxRuntime::BaseHandler`; session options, IO introspection, ONNX metadata, class names, timing |
| `ofxYolo26ModelThread` | generic background runner with frame dropping |
| `ofxYolo26Depth` | `Depth` + `DepthMap` + display helpers |
| `ofxYolo26Pose` | `Pose` + `PoseResult` + COCO skeleton + `drawPoses()` |
| `ofxYolo26Segmentation` | `Segmentation` + `SegmentationResult` + `InstanceMask` + overlay helpers |

Examples: `example-depth`, `example-pose`, `example-seg`.

## Notable decisions

- **Raw values passed through untouched** for depth, matching what the reference
  ships to TouchDesigner. The bundled model self-describes as
  `YOLO26n-depth-log` in its ONNX metadata while Ultralytics documents metric
  metres — flagged in the README rather than guessing at a conversion.
- **The resampler picks its filter per axis** — box average when shrinking,
  bilinear when growing. The reference gets this free from canvas `drawImage()`;
  naive bilinear would alias badly on a 1280x720 camera.
- **Depth min/max and auto-scaling use the content region only.** The letterbox
  bars get their own hallucinated depth, which would otherwise wreck every
  auto-scaled visualisation.
- **Segmentation instances are kept separate**, not composited into the
  reference's packed float map (class id in the integer part, alpha in the
  fraction). That encoding exists to push masks through a WebSocket into a
  single-channel TouchDesigner TOP — a transport constraint that does not exist
  here. `toColorPixels()` still produces a composited overlay for the cases
  where one texture is what you want.
- **Masks are stored cropped to their box**, which is both the only region where
  the prototype combination is meaningful and far cheaper than 100 full-plane
  masks per frame.
- **Class names are parsed from ONNX metadata**, so results read "person" and
  "bus" rather than 0 and 5.
- **`classColor()` uses van der Corput hue placement**, not golden-angle
  stepping. The first version put classes 0 and 5 only 33° apart on the wheel,
  which rendered the buses and people in a street scene as two nearly identical
  pinks; bit-reversal puts them at red and blue.

## Verification

Compared against Python onnxruntime running the same models with the same
letterbox preprocessing.

- **Preprocessing** — lossless source at exact 2:1 downscale: zero pixels differ
  from PIL's box filter by more than one 8-bit level. The residual is PIL
  quantising back to uint8 where this keeps the average in float.
- **Depth** — agrees to ≤0.37%. Letterbox geometry matches exactly, and
  `getValueAtSource()` on source corners returns exactly the content-region
  corners.
- **Pose** — same letterbox, same 11 detections on a group photo, top scores
  0.8994 / 0.8880 / 0.8152 vs 0.8994 / 0.8879 / 0.8335, nose of pose 0 at
  (562.03, 363.21) vs (562.09, 363.09). Rendering the skeletons confirms the
  bone table and the source mapping.
- **Segmentation** — same 8 instances and classes, boxes agreeing to ~0.1 px,
  instance 0's mask 59.3% above alpha 0.5 within its box vs 59.4%. Person-only
  filtering drops the run from 8 to 6 as expected. Rendered masks follow the bus
  silhouettes rather than their boxes.

Residual differences are JPEG decoder noise (FreeImage vs libjpeg) plus that
one-level preprocessing rounding.

All three examples launch, run, and shut their worker threads down cleanly.

## GPU acceleration (onnxruntime 1.29.0 + CoreML)

onnxruntime has no Metal/MPS provider; the Apple path is CoreML. The bundled
1.10.0 had no CoreML compiled in, so the runtime was replaced with 1.29.0 and
a `Backend` / `CoreMLComputeUnits` option added, defaulting to CoreML with
automatic CPU fallback.

Median steady-state inference, 640x640:

| task | CPU | CoreML | speedup |
|---|---|---|---|
| depth | 85.6 ms | 9.2 ms | 9.3x |
| pose | 29.1 ms | 11.3 ms | 2.6x |
| segmentation | 38.2 ms | 8.5 ms | 4.5x |

Accuracy is not traded away: depth and segmentation agree with CPU to within
1e-5, and pose keypoints land on the same pixel with a worst-case score
difference of 4e-5.

Four findings worth recording, none of them predictable from the docs:

- **`CoreMLComputeUnits::All` beats Neural-Engine-only** — 5.6 ms vs 22.9 ms on
  pose. Operators the ANE refuses fall back to CPU rather than to the GPU, so
  pinning to the ANE is the worst accelerated option, not the best.
- **The shipped `coreml_provider_factory.h` documents the wrong option values.**
  It lists `MLComputeUnitsAll`; the runtime only accepts bare `ALL`, `CPUOnly`,
  `CPUAndGPU`, `CPUAndNeuralEngine`. I probed the runtime for the accepted set
  rather than trusting the header.
- **CoreML compiles at first `Run`, not at session creation** — up to 1.9 s.
  `coreMLCacheDirectory` persists the compiled model and cuts that to ~60 ms on
  later launches. Warm the model up before feeding it live video.
- **A quarantined dylib hangs the process silently.** A browser-downloaded
  onnxruntime carries `com.apple.quarantine`; the app then sits at 0% CPU and
  192 KB RSS forever, with no crash and no log. `cp` propagates it to every
  `bin/` copy.

Two platform regressions from the upgrade, both documented in the README: the
1.29.0 macOS build is **arm64-only** (1.10.0 was universal) and requires
**macOS 14+**.

The CPU-fallback path proved itself during development — when the compute-units
value was still wrong, both models loaded on CPU with a warning and produced
correct results instead of failing.

## Changes outside the addon

`ofxOnnxRuntime` now carries onnxruntime 1.29.0 in place of 1.10.0, and two
calls in `ofxOnnxRuntime.cpp` were updated: `GetInputName`/`GetOutputName`
returned raw `char*` and were removed after 1.11 in favour of the `...Allocated`
forms. Those return owning pointers, so the names are copied into members that
outlive them — the old code leaked them, which is why the raw pointers stayed
valid. The 1.10.0 tree and the original `.cpp` are backed up in `~/Downloads`.

Separately, the examples would not link at all. `ofxOnnxRuntime`'s
`addon_config.mk` used
`-Xlinker -rpath -Xlinker @executable_path`, and openFrameworks runs addon
LDFLAGS through `$(call uniq,...)` — which collapses the repeated `-Xlinker` and
leaves `@executable_path` as a stray argument. Its own example ships only an
Xcode project, so this path was never exercised. It now uses the single-token
`-Wl,-rpath,...` form, plus a second rpath, since the makefile build puts the
dylib in `bin/` while Xcode puts it inside the bundle. Two lines in a dependency
owned locally, noted in the README, easy to revert.

`example*/bin/data/models/*.onnx` is gitignored — the models are 10–20 MB each,
so they are copied into place but will not be committed unless you want them to
be.
