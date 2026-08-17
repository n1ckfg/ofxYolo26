#include "ofxYolo26Obb.h"

namespace ofxYolo26 {

//--------------------------------------------------------------
// OrientedBox
//--------------------------------------------------------------

std::array<glm::vec2, 4> OrientedBox::getCorners() const {
	const float hw = 0.5f * size.x;
	const float hh = 0.5f * size.y;
	const float c = std::cos(angle);
	const float s = std::sin(angle);

	// Same ordering as polygonFromXYWHR() in the reference: top-left,
	// top-right, bottom-right, bottom-left of the unrotated box.
	const glm::vec2 local[4] = {
		{ -hw, -hh }, { hw, -hh }, { hw, hh }, { -hw, hh }
	};

	std::array<glm::vec2, 4> out;
	for (int i = 0; i < 4; i++) {
		out[i] = glm::vec2(
			center.x + local[i].x * c - local[i].y * s,
			center.y + local[i].x * s + local[i].y * c);
	}
	return out;
}

ofRectangle OrientedBox::getBoundingBox() const {
	const std::array<glm::vec2, 4> corners = getCorners();
	glm::vec2 lo = corners[0];
	glm::vec2 hi = corners[0];
	for (int i = 1; i < 4; i++) {
		lo = glm::min(lo, corners[i]);
		hi = glm::max(hi, corners[i]);
	}
	return ofRectangle(lo.x, lo.y, hi.x - lo.x, hi.y - lo.y);
}

//--------------------------------------------------------------
// ObbResult
//--------------------------------------------------------------

void ObbResult::clear() {
	boxes.clear();
	letterbox = Letterbox();
}

std::array<glm::vec2, 4> ObbResult::getCornersInSource(size_t index) const {
	std::array<glm::vec2, 4> out {};
	if (index >= boxes.size()) return out;

	const std::array<glm::vec2, 4> corners = boxes[index].getCorners();
	for (int i = 0; i < 4; i++) out[i] = letterbox.destToSource(corners[i]);
	return out;
}

glm::vec2 ObbResult::getCenterInSource(size_t index) const {
	if (index >= boxes.size()) return glm::vec2(0.0f);
	return letterbox.destToSource(boxes[index].center);
}

float ObbResult::getAngleInSource(size_t index) const {
	if (index >= boxes.size()) return 0.0f;
	// Letterboxing is a uniform scale plus a translation, so angles carry over
	// untouched. Stretch mode would skew them, so guard against that.
	if (letterbox.scaleX != letterbox.scaleY && letterbox.scaleY != 0.0f) {
		const OrientedBox & b = boxes[index];
		const float ratio = letterbox.scaleY / letterbox.scaleX;
		return std::atan2(std::sin(b.angle) * ratio, std::cos(b.angle));
	}
	return boxes[index].angle;
}

//--------------------------------------------------------------
// Obb
//--------------------------------------------------------------

bool Obb::setup(const std::string & modelPath, const Settings & settings) {
	result.clear();
	return load(modelPath, settings);
}

void Obb::setClassFilter(const std::vector<int> & labels) {
	classFilter = labels;
	std::sort(classFilter.begin(), classFilter.end());
}

bool Obb::onLoaded() {
	if (num_outputs < 1) {
		ofLogError("ofxYolo26::Obb") << "model has no outputs";
		return false;
	}

	const std::vector<int64_t> shape = getOutputShape(0);
	if (shape.size() != 3) {
		ofLogError("ofxYolo26::Obb") << "expected an end2end OBB output [1, N, 7]";
		return false;
	}

	proposalStride = int(shape[2]);
	if (proposalStride != 7) {
		ofLogError("ofxYolo26::Obb") << "OBB stride is " << proposalStride << ", expected 7"
									 << (proposalStride == 6 ? " -- this looks like a plain detect model, use ofxYolo26::Detector" : "");
		return false;
	}

	const std::string task = getMetadata("task");
	if (!task.empty() && task != "obb") {
		ofLogWarning("ofxYolo26::Obb") << "model reports task '" << task << "' rather than 'obb'";
	}
	return true;
}

const ObbResult & Obb::update(const ofPixels & pixels) {
	if (!isLoaded()) return result;

	preprocess(pixels);

	result.boxes.clear();
	result.letterbox = letterbox;

	if (!runInference()) return result;

	const Ort::Value & value = output_values[0];
	if (!value || !value.IsTensor()) {
		ofLogError("ofxYolo26::Obb") << "OBB output is missing";
		return result;
	}

	const std::vector<int64_t> shape = value.GetTensorTypeAndShapeInfo().GetShape();
	if (shape.size() != 3) return result;
	const int proposals = int(shape[1]);
	const int stride = int(shape[2]);
	if (stride != 7) {
		ofLogError("ofxYolo26::Obb") << "OBB stride changed to " << stride << " at runtime";
		return result;
	}

	const float * data = getOutputData(0);
	if (data == nullptr) return result;

	// End2end: threshold and sort, no NMS. Column order is
	// cx, cy, w, h, score, class, angle.
	for (int i = 0; i < proposals; i++) {
		const float * p = data + size_t(i) * size_t(stride);
		const float score = p[4];
		if (score < scoreThreshold) continue;

		const int label = int(p[5]);
		if (!classFilter.empty()
			&& !std::binary_search(classFilter.begin(), classFilter.end(), label)) {
			continue;
		}

		OrientedBox box;
		box.center = glm::vec2(p[0], p[1]);
		box.size = glm::vec2(p[2], p[3]);
		box.angle = p[6];
		box.score = score;
		box.label = label;
		box.labelName = getClassName(label);
		result.boxes.push_back(std::move(box));
	}

	std::stable_sort(result.boxes.begin(), result.boxes.end(),
		[](const OrientedBox & a, const OrientedBox & b) { return a.score > b.score; });

	if (maxDetections > 0 && int(result.boxes.size()) > maxDetections) {
		result.boxes.resize(maxDetections);
	}
	return result;
}

//--------------------------------------------------------------
// Drawing
//--------------------------------------------------------------

void drawOrientedBoxes(const ObbResult & result,
	const ofRectangle & rect,
	bool drawLabels,
	bool drawHeading) {

	if (result.letterbox.srcWidth <= 0 || result.letterbox.srcHeight <= 0) return;

	const float sx = rect.getWidth() / float(result.letterbox.srcWidth);
	const float sy = rect.getHeight() / float(result.letterbox.srcHeight);
	const auto toScreen = [&](const glm::vec2 & src) {
		return glm::vec2(rect.x + src.x * sx, rect.y + src.y * sy);
	};

	ofPushStyle();
	for (size_t i = 0; i < result.size(); i++) {
		const OrientedBox & box = result.boxes[i];
		const std::array<glm::vec2, 4> corners = result.getCornersInSource(i);
		const ofColor color = classColor(box.label);

		ofSetColor(color);
		ofSetLineWidth(2.0f);
		for (int c = 0; c < 4; c++) {
			const glm::vec2 a = toScreen(corners[c]);
			const glm::vec2 b = toScreen(corners[(c + 1) % 4]);
			ofDrawLine(a.x, a.y, b.x, b.y);
		}

		if (drawHeading) {
			// A spur from the centre through the middle of the leading edge, so
			// which way the box points is visible at a glance.
			const glm::vec2 center = toScreen(result.getCenterInSource(i));
			const glm::vec2 lead = toScreen((corners[1] + corners[2]) * 0.5f);
			ofDrawLine(center.x, center.y, lead.x, lead.y);
			ofDrawCircle(center.x, center.y, 2.0f);
		}

		if (drawLabels) {
			// Anchor the label to the topmost corner so it stays clear of the box.
			glm::vec2 top = corners[0];
			for (int c = 1; c < 4; c++) {
				if (corners[c].y < top.y) top = corners[c];
			}
			const glm::vec2 anchor = toScreen(top);
			ofDrawBitmapStringHighlight(box.labelName + " " + ofToString(box.score, 2)
					+ "  " + ofToString(ofRadToDeg(result.getAngleInSource(i)), 0) + "deg",
				anchor.x + 3, anchor.y - 6, color, ofColor::black);
		}
	}
	ofSetLineWidth(1.0f);
	ofPopStyle();
}

} // namespace ofxYolo26
