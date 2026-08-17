#pragma once

#include "ofxYolo26Detection.h"
#include "ofxYolo26Model.h"
#include "ofxYolo26ModelThread.h"

namespace ofxYolo26 {

class DetectionResult {
public:
	Letterbox letterbox;
	std::vector<Detection> detections;

	size_t size() const { return detections.size(); }
	bool empty() const { return detections.empty(); }
	void clear();

	/// Detection box mapped back to source-image pixels.
	ofRectangle getBoxInSource(size_t index) const;
};

/// Object detection with a YOLO26 detect export.
///
/// Input:  images,  float32 [1, 3, 640, 640], planar RGB in [0,1]
/// Output: output0, float32 [1, 300, 6]
///         per proposal: x1, y1, x2, y2, score, class
///
/// These exports are end2end — NMS runs inside the graph — so decoding is a
/// score threshold and a sort, exactly as decodeYOLOv26() does in the reference
/// web app. Note the box is corner form here, unlike the OBB head's centre form.
class Detector : public Model {
public:
	bool setup(const std::string & modelPath = "models/yolo26n.onnx",
		const Settings & settings = Settings());

	/// Preprocesses and runs the model. Blocking; use DetectorThread for live
	/// video.
	const DetectionResult & update(const ofPixels & pixels);

	const DetectionResult & getDetectionResult() const { return result; }

	/// Minimum detection score. Default 0.4, matching the reference's DET_SCORE_T.
	void setScoreThreshold(float threshold) { scoreThreshold = threshold; }
	float getScoreThreshold() const { return scoreThreshold; }

	/// Cap on returned detections, highest score first. Default 100 (DET_TOPK).
	void setMaxDetections(int count) { maxDetections = count; }
	int getMaxDetections() const { return maxDetections; }

	/// Keep only these class ids. Empty (the default) keeps everything.
	void setClassFilter(const std::vector<int> & labels);
	void clearClassFilter() { classFilter.clear(); }

protected:
	bool onLoaded() override;

	DetectionResult result;
	float scoreThreshold = 0.4f;
	int maxDetections = 100;
	std::vector<int> classFilter;
	int proposalStride = 0;
};

/// Object detection on a background thread. See ModelThread for the mechanics.
class DetectorThread : public ModelThread<Detector, DetectionResult> {
public:
	const DetectionResult & getDetectionResult() const { return getResult(); }
	const Detector & getDetector() const { return getModel(); }
};

/// Draws boxes and labels, mapping source-image coordinates into `rect` — pass
/// the same rectangle you drew the frame into.
void drawDetections(const DetectionResult & result,
	const ofRectangle & rect,
	bool drawLabels = true);

} // namespace ofxYolo26
