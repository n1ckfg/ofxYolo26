# ofxYolo26

<img src="ofxaddons_thumbnail.png"><br>

Ultralytics YOLO26 ONNX models in openFrameworks.

A C++ port of the inference half of [yolo-touchdesigner](https://github.com/torinmb/yolo-touchdesigner).
It runs the same `.onnx` exports, but through onnxruntime via
[ofxOnnxRuntime](https://github.com/hanasaan/ofxOnnxRuntime) instead of
onnxruntime-web/WebGPU in a browser. The reference project's server half — the
Web Server DAT, the WebSocket transport, the binary framing protocol — is
deliberately not ported; in openFrameworks you already have the pixels.

**Implemented**

| task | class | model | example |
|---|---|---|---|
| object detection | `ofxYolo26::Detector` | `yolo26n.onnx` | `example-detect` |
| oriented boxes (OBB) | `ofxYolo26::Obb` | `yolo26n-obb.onnx` | `example-obb` |
| monocular depth | `ofxYolo26::Depth` | `yolo26n-depth.onnx` | `example-depth` |
| multi-person pose | `ofxYolo26::Pose` | `yolo26n-pose.onnx` | `example-pose` |
| instance segmentation | `ofxYolo26::Segmentation` | `yolo26n-seg.onnx` | `example-seg` |

## Dependencies

- [ofxOnnxRuntime](https://github.com/hanasaan/ofxOnnxRuntime), with its
  bundled onnxruntime replaced by **1.29.0** — see Upgrading onnxruntime below.

**Platform:** the onnxruntime 1.29.0 macOS build is **arm64 only** and requires
**macOS 14+**. Apple Silicon, in other words. If you need Intel or an older
macOS, stay on the 1.10.0 that ofxOnnxRuntime ships (universal, CPU-only) — the
addon still builds against it apart from the two API calls noted below.

## Setup

Copy the model your project needs into its data folder:

```
bin/data/models/yolo26n.onnx
bin/data/models/yolo26n-obb.onnx
bin/data/models/yolo26n-depth.onnx
bin/data/models/yolo26n-pose.onnx
bin/data/models/yolo26n-seg.onnx
```

Each example expects its model at that path. Models live in the reference
project under `reference/yolo-touchdesigner/public/models/`.

## Usage

Every task follows the same shape: `setup()` with a model path, `update()` with
pixels, read a result object back.

```cpp
ofxYolo26::Pose pose;
pose.setup("models/yolo26n-pose.onnx");

const ofxYolo26::PoseResult & result = pose.update(pixels);
for (size_t i = 0; i < result.size(); i++) {
    ofRectangle box = result.getBoxInSource(i);
    glm::vec2 nose = result.getKeypointInSource(i, ofxYolo26::KP_NOSE);
}
```

Live video, with inference on a background thread — `DepthThread`, `PoseThread`
and `SegmentationThread` all work this way:

```cpp
// setup()
poseThread.setup("models/yolo26n-pose.onnx");
poseThread.start();

// update()
if (grabber.isFrameNew()) poseThread.setInput(grabber.getPixels());
if (poseThread.isFrameNew()) { /* poseThread.getPoseResult() */ }

// draw()
ofxYolo26::drawPoses(poseThread.getPoseResult(), frameRect);
```

Frames handed to `setInput()` while the worker is busy are dropped rather than
queued, so results stay as close to live as the model allows.

## Backends

onnxruntime has no Metal or MPS execution provider; on Apple hardware the
accelerated path is **CoreML**, which spreads the graph across the Neural
Engine, GPU and CPU. It is the default here, and it is worth 2.6x–9.3x (see
Performance below).

```cpp
ofxYolo26::Settings settings;
settings.backend = ofxYolo26::Backend::CoreML;   // the default on macOS
settings.coreMLComputeUnits = ofxYolo26::CoreMLComputeUnits::All;
settings.fallbackToCPU = true;                   // the default

depth.setup("models/yolo26n-depth.onnx", settings);
depth.getActiveBackend();   // what you actually got
```

A few things worth knowing:

- **Requesting an unavailable provider throws** rather than degrading, so the
  loader catches it and retries on CPU with a warning. `getActiveBackend()`
  reports what is actually running; set `fallbackToCPU = false` to fail loudly
  instead.
- **`CoreMLComputeUnits::All` is the fastest.** Restricting to the Neural
  Engine is *slower* than letting CoreML choose — the operators the ANE will
  not take fall back to CPU rather than to the GPU.
- **The first inference is slow.** CoreML compiles the model on the first
  `Run`, not at session creation: cold, that is 0.3–1.9 s depending on the
  model. `Settings::coreMLCacheDirectory` (default `coreml-cache` in your data
  folder) persists the compiled model, which cuts a cold first inference of
  1.9 s to about 60 ms on later launches. Either way, run one throwaway frame
  through the model during setup rather than letting a live source hit it.
- **Dynamic input shapes get pinned.** `yolo26n-obb` declares
  `[batch, 3, height, width]`, and CoreML cannot do much with a graph whose
  shapes are unknown — left free it runs at CPU speed (23.8 ms vs 23.2 ms).
  The addon pins the free axes with `AddFreeDimensionOverrideByName`, which
  takes it to 7.2 ms. Static models are unaffected; the override is ignored.
  Size comes from `Settings::dynamicInputWidth/Height`, or the export's `imgsz`
  metadata, defaulting to 640x640.
- **The CoreML cache is keyed per configuration.** onnxruntime keys its own
  cache on the model path alone, so changing anything that alters graph
  partitioning — input size, compute units, model format — would silently reuse
  a mismatched compiled model and fail at inference with
  `Feature ... is required but not specified`. The addon gives each
  configuration its own subdirectory, named for the model, size, compute units,
  format, and the model file's size and mtime, so replacing a `.onnx` in place
  is also handled.
- `Backend::CUDA` and `Backend::TensorRT` are for platforms where you have
  swapped in a runtime that supports them; the macOS build has neither.

## Coordinates and letterboxing

Source frames are fitted into the model's 640x640 input with aspect ratio
preserved and the remainder padded black — the same thing the reference app's
webcam path does via canvas `drawImage()`. `ResizeMode::Stretch` is available
if you would rather fill the input, matching the reference app's TouchDesigner
binary path where the host has already sized the frame.

**Every model reports coordinates in that 640x640 input plane, not in your
frame.** Each result carries the `Letterbox` that maps between the two, and the
`...InSource()` helpers do the conversion for you:

```cpp
depthMap.getValueAtSource(x, y);
poseResult.getBoxInSource(i);
poseResult.getKeypointInSource(i, k);
segResult.getBoxInSource(i);
segResult.getAlphaAtSource(x, y, &instance);
letterbox.destToSource({dx, dy});   // the general case
```

## Object detection

Output is `[1, 300, 6]`: per proposal `x1, y1, x2, y2, score, class`. End2end,
so decoding is a score threshold and a sort, exactly as `decodeYOLOv26()` does
in the reference.

```cpp
ofxYolo26::Detector detector;
detector.setup("models/yolo26n.onnx");
detector.setScoreThreshold(0.4f);   // DET_SCORE_T
detector.setMaxDetections(100);     // DET_TOPK
detector.setClassFilter({0});       // people only, by COCO id

const ofxYolo26::DetectionResult & result = detector.update(pixels);
for (size_t i = 0; i < result.size(); i++) {
    ofRectangle box = result.getBoxInSource(i);
    result.detections[i].labelName;   // "person", "bus", ...
}

ofxYolo26::drawDetections(result, frameRect);
```

## Oriented bounding boxes

Output is `[batch, 300, 7]`: per proposal `cx, cy, w, h, score, class, angle`.
Note the box is **centre form** here, unlike the detect head's corner form, and
`angle` is radians in image coordinates (+Y down), so a positive angle turns
clockwise on screen.

```cpp
ofxYolo26::Obb obb;
obb.setup("models/yolo26n-obb.onnx");

const ofxYolo26::ObbResult & result = obb.update(pixels);
for (size_t i = 0; i < result.size(); i++) {
    std::array<glm::vec2, 4> corners = result.getCornersInSource(i);
    glm::vec2 centre = result.getCenterInSource(i);
    float angle = result.getAngleInSource(i);
    ofRectangle aabb = result.boxes[i].getBoundingBox();
}

ofxYolo26::drawOrientedBoxes(result, frameRect);
```

Two things to know about this model:

- **Its classes are aerial.** The stock export is trained on DOTA: plane, ship,
  storage tank, harbour, roundabout, large/small vehicle and so on. On an
  ordinary photograph it scores near zero, which is correct behaviour rather
  than a broken decode. `example-obb` takes a dropped image so you can try a
  real aerial or satellite frame.
- **Its input shape is dynamic** — `[batch, 3, height, width]`, unlike every
  other export here. The addon pins the free axes at load; see below.

The reference web app's header comment claims the column order is
`cx, cy, w, h, angle, score, cls`. That is wrong — its own code, and the model's
actual output, put score at 4, class at 5 and angle at 6. This port follows the
measured layout.

## Depth

```cpp
ofxYolo26::Depth depth;
depth.setup("models/yolo26n-depth.onnx");
const ofxYolo26::DepthMap & map = depth.update(pixels);

ofxYolo26::toFloatPixels(map, floatPixels);        // raw values, 1 channel
ofxYolo26::toNormalizedPixels(map, floatPixels);   // rescaled to 0..1
ofxYolo26::toColorPixels(map, pixels);             // through a colour ramp
```

`DepthMap::values` holds the model's output untouched — no normalisation, no
inversion — matching what the reference ships to TouchDesigner. Larger values
are further away.

Ultralytics documents the YOLO26 depth export as metric depth in metres. The
model bundled with the reference project describes itself as
`YOLO26n-depth-log` in its ONNX metadata, so treat the absolute scale as the
model's business and rely on the relative ordering unless you have calibrated
against something known. On a 640x640 input the raw range typically lands
somewhere around 0.8–2.5.

Two consequences of letterboxing worth knowing:

- The padding bars get their own hallucinated depth, so `minValue`/`maxValue`
  and the conversion helpers' auto-scaling are computed over the **content
  region only**, and `cropToContent` defaults to true.
- The model bleeds slightly across the content/padding boundary, leaving a thin
  edge artifact on the padded sides. This is inherent to letterboxing and the
  reference app has it too. `ResizeMode::Stretch` avoids it at the cost of
  distorting the image.

## Pose

Output is `[1, 300, 57]`: per proposal `x1, y1, x2, y2, score, class`, then
17 * `(x, y, score)`. The export is end2end — NMS runs inside the graph — so
decoding is a score threshold and a sort, exactly as `decodeYOLOv26Pose()` does
in the reference.

```cpp
pose.setScoreThreshold(0.35f);   // POSE_SCORE_T
pose.setMaxPoses(50);            // POSE_TOPK

ofxYolo26::drawPoses(result, frameRect, 0.3f);
```

`getCocoKeypointNames()` and `getCocoSkeleton()` give the COCO 17-point layout
and its bone pairs; `KP_NOSE`, `KP_LEFT_WRIST` and friends index into a pose's
keypoints. Check the model's `kpt_shape` metadata before assuming COCO for a
custom export — `Pose::getNumKeypoints()` reports what the loaded model
actually has.

## Segmentation

Outputs are `[1, 300, 38]` (`x1, y1, x2, y2, score, class`, then 32 prototype
coefficients) and `[1, 32, 160, 160]` (the prototypes). A mask is
`sigmoid(sum of coefficient * prototype)` evaluated inside the detection's box,
the same combination the reference's compute shader performs.

```cpp
seg.setScoreThreshold(0.2f);      // SEG_SCORE_T
seg.setIouThreshold(0.45f);       // duplicate suppression; 1.0 disables
seg.setMaxInstances(100);         // SEG_TOPK
seg.setClassFilter({0});          // PERSON_SEG_ONLY

const ofxYolo26::SegmentationResult & result = seg.update(pixels);
for (size_t i = 0; i < result.size(); i++) {
    result.detections[i].labelName;   // "person", "bus", ... from ONNX metadata
    result.masks[i].alpha;            // cropped to the box, in [0,1]
}

ofxYolo26::toColorPixels(result, pixels);       // RGBA overlay, class-coloured
ofxYolo26::toAlphaPixels(result, floatPixels);  // union alpha
ofxYolo26::toAlphaPixels(result, floatPixels, i);
```

Unlike the reference, instances are kept separate rather than composited into
one float map with the class id in the integer part and alpha in the fraction.
That encoding exists to squeeze masks through a WebSocket into a
single-channel TouchDesigner TOP; here you can just iterate the instances.
`toColorPixels()` still produces a composited overlay when a single texture is
what you want, painting largest boxes first so small foreground objects stay on
top.

Two things to watch:

- Masks are stored **cropped to their detection box**, because that is the only
  region where the prototype combination means anything — the same coefficients
  light up unrelated parts of the image outside it.
- Mask textures span the whole padded 640x640 input, so drawing one straight
  over your video will be offset and squashed. Use
  `SegmentationResult::getContentRectInProto()` with
  `ofTexture::drawSubsection()`, as `example-seg` does.

The reference applies its own NMS pass to segmentation results even though the
export is end2end, because near-duplicate boxes that survive the in-graph NMS
end up fighting over the same mask pixels. That is ported and on by default.

## Examples

All five take `space` to switch between a bundled still and the webcam, `d` to
dump results to the console, and `h` to toggle the help line.

- **example-detect** — boxes and labels. `-`/`=` score threshold, `f` cycles
  all / people / vehicles, `l` labels. Drop an image on the window to use it.
- **example-obb** — rotated boxes with a heading spur. `-`/`=` score
  threshold, `l` labels, `o` heading. Drop an aerial image on the window; the
  bundled street photo is a stand-in and will show little or nothing, which the
  footer says explicitly.
- **example-depth** — depth map beside the frame, mouse probe reading raw depth
  in source coordinates. `c` colour ramp, `i` invert, `x` show/crop padding.
- **example-pose** — skeletons over the frame. `-`/`=` score threshold,
  `[`/`]` keypoint threshold, `b` boxes, `l` labels.
- **example-seg** — `v` cycles overlay / mask / cutout, `-`/`=` score
  threshold, `c` colour by class or instance, `p` person-only filter,
  `b` boxes. The mouse reports which instance owns the pixel under it.

## Performance

Steady-state inference, 640x640, Apple Silicon laptop, onnxruntime 1.29.0.
Median of 5 runs, each the median of 10 inferences:

| task | CPU | CoreML | speedup |
|---|---|---|---|
| detect | 25.6 ms | 10.2 ms | **2.5x** |
| obb | 25.4 ms | 7.2 ms | **3.5x** |
| depth | 85.2 ms | 9.6 ms | **8.9x** |
| pose | 30.0 ms | 7.8 ms | **3.9x** |
| segmentation | 36.9 ms | 12.8 ms | **2.9x** |

Every task clears 60 fps on CoreML; depth goes from ~12 fps to ~104. The
threaded runners still matter, since even 10 ms on the render thread is a third
of a 30 fps budget.

CoreML timings are bimodal run to run — segmentation lands on either ~8.7 ms or
~12.8 ms depending on how CoreML partitions that session, and the others vary
similarly. The medians above are honest middles, not best cases.

Session load with the cache warm is about 40–90 ms per model; cold, roughly
300–380 ms while CoreML compiles.

## Verification

Every task was checked against the Python onnxruntime reference running the
same model with the same letterbox preprocessing.

**Preprocessing** — on a lossless 1280x720 source at an exact 2:1 downscale, no
pixel of the tensor differs from PIL's box filter by more than one 8-bit level
(mean 0.28 levels). The residual is PIL quantising its resampled image back to
uint8, where this keeps the box average in float.

**Depth**, over the content region:

| | python | c++ | diff |
|---|---|---|---|
| content min | 0.8291 | 0.8322 | 0.37% |
| content max | 1.4835 | 1.4839 | 0.03% |
| centre | 1.0705 | 1.0737 | 0.30% |

**Pose**, on a 1024x680 group photo: same letterbox (640x425, pad 0,107), same
11 detections, top scores 0.8994 / 0.8880 / 0.8152 against Python's 0.8994 /
0.8879 / 0.8335, nose keypoint of pose 0 at (562.03, 363.21) against
(562.09, 363.09).

**Segmentation**, on a street scene: same 8 instances, same classes, boxes
agreeing to ~0.1 px; instance 0's mask covers 59.3% of its box above alpha 0.5
against Python's 59.4%.

**Detection and OBB** were checked by handing Python the exact tensor the addon
built, which removes the image decoder from the comparison entirely. Every value
matches to 4 decimal places — all 6 detections and all 3 OBB rows including
angles:

```
detect #0  C++  0.9336 cls5 [279.7271 237.2336 524.2201 396.6292]
           py   0.9336 cls5 [279.7271 237.2336 524.2201 396.6292]
obb    #0  C++  0.0225 cls10 c=(326.3159,337.6576) wh=(15.9558,11.6423) ang=0.5287
           py   0.0225 cls10 c=(326.3159,337.6576) wh=(15.9558,11.6423) ang=0.5287
```

Feeding the same *image* instead yields 6 detections against Python's 5, because
one box scores 0.4157 here and just under the 0.4 threshold under PIL's JPEG
decoder. That is the decoder, not the decode.

The OBB corner maths is checked geometrically as well: opposite edges come back
exactly `w, h, w, h`, adjacent edges are perpendicular to within 1e-7, the
corner centroid lands on the reported centre, and the corner heading recovers
the reported angle.

Remaining differences are JPEG decoder noise (FreeImage vs libjpeg) plus the
one-level preprocessing rounding above.

**CoreML vs CPU**, same addon, same inputs — the accelerated paths may compute
at reduced precision, so this matters:

| task | agreement |
|---|---|
| depth | mean and max \|diff\| below 1e-5 over all 409,600 pixels |
| pose | same 11 detections; max score diff 4e-5; every keypoint on the same pixel |
| segmentation | same 8 instances, identical classes; max score and mask-alpha diff below 1e-5 |

In other words, the backend is a performance choice, not an accuracy trade-off,
on these models.

## Upgrading onnxruntime

ofxOnnxRuntime ships onnxruntime 1.10.0, which is CPU-only on macOS — no CoreML
provider is compiled in. Getting GPU/ANE acceleration means replacing it:

1. Drop a current `onnxruntime-osx-arm64` release into
   `ofxOnnxRuntime/libs/onnxruntime`: headers into `include/`, and the dylib
   into `lib/osx/` named **`libonnxruntime.1.dylib`** (its `install_name` is
   `@rpath/libonnxruntime.1.dylib`, so the filename has to match).

2. **Strip the quarantine attribute**, or the app will hang on launch:

   ```bash
   xattr -d com.apple.quarantine libs/onnxruntime/lib/osx/libonnxruntime.1.dylib
   ```

   A browser-downloaded dylib carries `com.apple.quarantine`, and Gatekeeper
   stalls the process at load — it does not crash or log, it just sits there at
   0% CPU forever. `cp` propagates the attribute, so any copy already in a
   `bin/` folder needs the same treatment.

3. Patch two calls in `ofxOnnxRuntime.cpp` — `GetInputName`/`GetOutputName`
   returned a raw `char*` and were removed after 1.11 in favour of
   `GetInputNameAllocated`/`GetOutputNameAllocated`. The replacements return
   owning pointers, so the names must be copied into storage that outlives
   them; the old code leaked them, which is why it worked. Both changes are in
   this repo's copy of the addon.

## Notes

- Model paths are resolved with `ofToDataPath()`.
- Class names come from the export's `names` ONNX metadata, so
  `detections[i].labelName` reads "person" rather than "0".
- `example*/bin/data/models/*.onnx` and `example*/bin/data/coreml-cache/` are
  gitignored — the models are 10–20 MB each and the cache is build output.
- The `coreml_provider_factory.h` shipped with 1.29.0 documents `MLComputeUnits`
  values as `MLComputeUnitsAll`, `MLComputeUnitsCPUOnly` and so on. **The
  runtime rejects those.** The accepted values are the bare tokens `ALL`,
  `CPUOnly`, `CPUAndGPU` and `CPUAndNeuralEngine`, case-sensitive.
- ofxOnnxRuntime's `addon_config.mk` needed a fix to link under the openFrameworks
  makefiles: openFrameworks runs addon LDFLAGS through `$(call uniq,...)`, which
  collapsed the repeated `-Xlinker` in `-Xlinker -rpath -Xlinker @executable_path`
  and left `@executable_path` as a stray linker argument. It now uses the
  single-token `-Wl,-rpath,...` form, plus a second rpath for the makefile
  build's bundle layout.
- Linking warns `building for macOS-11.0, but linking with dylib ... built for
  newer version 14.0`. That is openFrameworks' deployment target versus the
  runtime's; harmless as long as you are on macOS 14+.

## License

AGPL-3.0, see LICENSE.txt — matching yolo-touchdesigner, which this ports from,
and the Ultralytics YOLO models themselves.
