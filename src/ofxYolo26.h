#pragma once

// ofxYolo26 - Ultralytics YOLO26 ONNX models in openFrameworks.
//
// A C++ port of the inference half of yolo-touchdesigner
// (https://github.com/torinmb/yolo-touchdesigner), running the same .onnx
// exports through onnxruntime via ofxOnnxRuntime instead of a browser and
// WebSockets. The web app's server/transport layers are deliberately not ported.
//
// Currently implemented: monocular depth (yolo26n-depth.onnx).

#include "ofxYolo26Types.h"
#include "ofxYolo26Preprocessor.h"
#include "ofxYolo26Model.h"
#include "ofxYolo26Depth.h"
#include "ofxYolo26DepthThread.h"
