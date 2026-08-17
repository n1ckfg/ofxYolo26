#include "ofxYolo26Detector.h"

namespace ofxYolo26 {

//--------------------------------------------------------------
// DetectionResult
//--------------------------------------------------------------

void DetectionResult::clear() {
	detections.clear();
	letterbox = Letterbox();
}

ofRectangle DetectionResult::getBoxInSource(size_t index) const {
	if (index >= detections.size()) return ofRectangle();

	const ofRectangle & box = detections[index].box;
	const glm::vec2 topLeft = letterbox.destToSource(glm::vec2(box.getMinX(), box.getMinY()));
	const glm::vec2 bottomRight = letterbox.destToSource(glm::vec2(box.getMaxX(), box.getMaxY()));
	return ofRectangle(topLeft.x, topLeft.y, bottomRight.x - topLeft.x, bottomRight.y - topLeft.y);
}

//--------------------------------------------------------------
// Detector
//--------------------------------------------------------------

bool Detector::setup(const std::string & modelPath, const Settings & settings) {
	result.clear();
	return load(modelPath, settings);
}

void Detector::setClassFilter(const std::vector<int> & labels) {
	classFilter = labels;
	std::sort(classFilter.begin(), classFilter.end());
}

bool Detector::onLoaded() {
	if (num_outputs < 1) {
		ofLogError("ofxYolo26::Detector") << "model has no outputs";
		return false;
	}

	const std::vector<int64_t> shape = getOutputShape(0);
	if (shape.size() != 3) {
		ofLogError("ofxYolo26::Detector") << "expected an end2end detect output [1, N, 6]";
		return false;
	}

	proposalStride = int(shape[2]);
	if (proposalStride != 6) {
		// 7 is the OBB head, which needs the extra angle column decoded.
		ofLogError("ofxYolo26::Detector") << "detect stride is " << proposalStride
										  << ", expected 6"
										  << (proposalStride == 7 ? " -- this looks like an OBB model, use ofxYolo26::Obb" : "");
		return false;
	}

	const std::string task = getMetadata("task");
	if (!task.empty() && task != "detect") {
		ofLogWarning("ofxYolo26::Detector") << "model reports task '" << task << "' rather than 'detect'";
	}
	return true;
}

const DetectionResult & Detector::update(const ofPixels & pixels) {
	if (!isLoaded()) return result;

	preprocess(pixels);

	result.detections.clear();
	result.letterbox = letterbox;

	if (!runInference()) return result;

	const Ort::Value & value = output_values[0];
	if (!value || !value.IsTensor()) {
		ofLogError("ofxYolo26::Detector") << "detect output is missing";
		return result;
	}

	const std::vector<int64_t> shape = value.GetTensorTypeAndShapeInfo().GetShape();
	if (shape.size() != 3) return result;
	const int proposals = int(shape[1]);
	const int stride = int(shape[2]);
	if (stride != 6) {
		ofLogError("ofxYolo26::Detector") << "detect stride changed to " << stride << " at runtime";
		return result;
	}

	const float * data = getOutputData(0);
	if (data == nullptr) return result;

	// The export is end2end, so this is a threshold and a sort — no NMS.
	for (int i = 0; i < proposals; i++) {
		const float * p = data + size_t(i) * size_t(stride);
		const float score = p[4];
		if (score < scoreThreshold) continue;

		const int label = int(p[5]);
		if (!classFilter.empty()
			&& !std::binary_search(classFilter.begin(), classFilter.end(), label)) {
			continue;
		}

		Detection det;
		// Corner form: x1, y1, x2, y2.
		det.box = ofRectangle(p[0], p[1], p[2] - p[0], p[3] - p[1]);
		det.score = score;
		det.label = label;
		det.labelName = getClassName(label);
		result.detections.push_back(std::move(det));
	}

	std::stable_sort(result.detections.begin(), result.detections.end(),
		[](const Detection & a, const Detection & b) { return a.score > b.score; });

	if (maxDetections > 0 && int(result.detections.size()) > maxDetections) {
		result.detections.resize(maxDetections);
	}
	return result;
}

//--------------------------------------------------------------
// Drawing
//--------------------------------------------------------------

void drawDetections(const DetectionResult & result, const ofRectangle & rect, bool drawLabels) {
	if (result.letterbox.srcWidth <= 0 || result.letterbox.srcHeight <= 0) return;

	const float sx = rect.getWidth() / float(result.letterbox.srcWidth);
	const float sy = rect.getHeight() / float(result.letterbox.srcHeight);

	ofPushStyle();
	for (size_t i = 0; i < result.size(); i++) {
		const ofRectangle box = result.getBoxInSource(i);
		const ofColor color = classColor(result.detections[i].label);
		const float x = rect.x + box.x * sx;
		const float y = rect.y + box.y * sy;

		ofNoFill();
		ofSetColor(color);
		ofSetLineWidth(2.0f);
		ofDrawRectangle(x, y, box.getWidth() * sx, box.getHeight() * sy);
		ofFill();

		if (drawLabels) {
			ofDrawBitmapStringHighlight(result.detections[i].labelName + " "
					+ ofToString(result.detections[i].score, 2),
				x + 3, y - 4, color, ofColor::black);
		}
	}
	ofSetLineWidth(1.0f);
	ofPopStyle();
}

} // namespace ofxYolo26
