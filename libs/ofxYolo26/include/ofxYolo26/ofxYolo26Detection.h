#pragma once

#include "ofxYolo26Types.h"

namespace ofxYolo26 {

/// One detected object.
///
/// `box` is in model input-plane pixels (640x640 for the stock exports), which
/// is the space every YOLO26 head reports in. Use Letterbox::destToSource(),
/// or the getBoxInSource() helpers on the result classes, to get back to the
/// frame you handed in.
struct Detection {
	ofRectangle box;
	int label = 0;
	std::string labelName;
	float score = 0.0f;
};

float intersectionOverUnion(const ofRectangle & a, const ofRectangle & b);

/// Greedy per-class non-maximum suppression, matching nmsPerClass() in the
/// reference web app. Returns indices into `dets`, ordered by descending score.
///
/// YOLO26 exports are end2end — NMS already runs inside the graph — so this is
/// not needed to decode them. The reference still applies it to segmentation
/// results, where near-duplicate boxes that survive the in-graph NMS end up
/// fighting over the same mask pixels.
std::vector<int> nmsPerClassIndices(const std::vector<Detection> & dets, float iouThreshold, int topk);

std::vector<Detection> nmsPerClass(const std::vector<Detection> & dets, float iouThreshold, int topk);

/// Stable, reasonably distinct colour for a class id, for overlays and masks.
ofColor classColor(int label);

} // namespace ofxYolo26
