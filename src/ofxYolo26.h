#pragma once

// ofxYolo26 - Ultralytics YOLO26 ONNX models in openFrameworks.
//
// A C++ port of the inference half of yolo-touchdesigner
// (https://github.com/torinmb/yolo-touchdesigner), running the same .onnx
// exports through onnxruntime via ofxOnnxRuntime instead of a browser and
// WebSockets. The web app's server/transport layers are deliberately not ported.
//
// Implemented: monocular depth, multi-person pose, instance segmentation.

#include "ofxYolo26Types.h"
#include "ofxYolo26Preprocessor.h"
#include "ofxYolo26Detection.h"
#include "ofxYolo26Model.h"
#include "ofxYolo26ModelThread.h"
#include "ofxYolo26Depth.h"
#include "ofxYolo26DepthThread.h"
#include "ofxYolo26Pose.h"
#include "ofxYolo26Segmentation.h"
