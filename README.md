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
| monocular depth | `ofxYolo26::Depth` | `yolo26n-depth.onnx` | `example-depth` |
| multi-person pose | `ofxYolo26::Pose` | `yolo26n-pose.onnx` | `example-pose` |
| instance segmentation | `ofxYolo26::Segmentation` | `yolo26n-seg.onnx` | `example-seg` |

Object detection and OBB share the same plumbing and are the natural next
additions — their heads are already the simplest case of the pose decoder.

## Dependencies

- [ofxOnnxRuntime](https://github.com/hanasaan/ofxOnnxRuntime)

## Setup

Copy the model your project needs into its data folder:

```
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

All three take `space` to switch between a bundled still and the webcam, `d` to
dump results to the console, and `h` to toggle the help line.

- **example-depth** — depth map beside the frame, mouse probe reading raw depth
  in source coordinates. `c` colour ramp, `i` invert, `x` show/crop padding.
- **example-pose** — skeletons over the frame. `-`/`=` score threshold,
  `[`/`]` keypoint threshold, `b` boxes, `l` labels.
- **example-seg** — `v` cycles overlay / mask / cutout, `-`/`=` score
  threshold, `c` colour by class or instance, `p` person-only filter,
  `b` boxes. The mouse reports which instance owns the pixel under it.

## Performance

The onnxruntime build shipped with ofxOnnxRuntime is 1.10.0, CPU-only on macOS
— it has no CoreML execution provider compiled in. On an Apple Silicon laptop,
per 640x640 inference: depth ~80 ms, pose ~35 ms, segmentation ~38 ms. Hence
the threaded runners. `Settings::inferType` exposes CUDA and TensorRT for
platforms where you have swapped in a runtime that supports them.

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

Remaining differences are JPEG decoder noise (FreeImage vs libjpeg) plus the
one-level preprocessing rounding above.

## Notes

- Model paths are resolved with `ofToDataPath()`.
- Class names come from the export's `names` ONNX metadata, so
  `detections[i].labelName` reads "person" rather than "0".
- `example*/bin/data/models/*.onnx` is gitignored — the models are 10–20 MB each.
- ofxOnnxRuntime's `addon_config.mk` needed a fix to link under the openFrameworks
  makefiles: openFrameworks runs addon LDFLAGS through `$(call uniq,...)`, which
  collapsed the repeated `-Xlinker` in `-Xlinker -rpath -Xlinker @executable_path`
  and left `@executable_path` as a stray linker argument. It now uses the
  single-token `-Wl,-rpath,...` form, plus a second rpath for the makefile
  build's bundle layout.

## License

AGPL-3.0, see LICENSE.txt — matching yolo-touchdesigner, which this ports from,
and the Ultralytics YOLO models themselves.
