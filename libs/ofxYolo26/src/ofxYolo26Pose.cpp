#include "ofxYolo26Pose.h"

namespace ofxYolo26 {

//--------------------------------------------------------------
// COCO layout
//--------------------------------------------------------------

const std::vector<std::string> & getCocoKeypointNames() {
	static const std::vector<std::string> names = {
		"nose", "left_eye", "right_eye", "left_ear", "right_ear",
		"left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
		"left_wrist", "right_wrist", "left_hip", "right_hip",
		"left_knee", "right_knee", "left_ankle", "right_ankle"
	};
	return names;
}

const std::vector<glm::ivec2> & getCocoSkeleton() {
	static const std::vector<glm::ivec2> bones = {
		// Face
		{ KP_NOSE, KP_LEFT_EYE }, { KP_NOSE, KP_RIGHT_EYE },
		{ KP_LEFT_EYE, KP_LEFT_EAR }, { KP_RIGHT_EYE, KP_RIGHT_EAR },
		// Arms
		{ KP_LEFT_SHOULDER, KP_LEFT_ELBOW }, { KP_LEFT_ELBOW, KP_LEFT_WRIST },
		{ KP_RIGHT_SHOULDER, KP_RIGHT_ELBOW }, { KP_RIGHT_ELBOW, KP_RIGHT_WRIST },
		// Torso
		{ KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER },
		{ KP_LEFT_SHOULDER, KP_LEFT_HIP }, { KP_RIGHT_SHOULDER, KP_RIGHT_HIP },
		{ KP_LEFT_HIP, KP_RIGHT_HIP },
		// Legs
		{ KP_LEFT_HIP, KP_LEFT_KNEE }, { KP_LEFT_KNEE, KP_LEFT_ANKLE },
		{ KP_RIGHT_HIP, KP_RIGHT_KNEE }, { KP_RIGHT_KNEE, KP_RIGHT_ANKLE }
	};
	return bones;
}

//--------------------------------------------------------------
// PoseResult
//--------------------------------------------------------------

void PoseResult::clear() {
	poses.clear();
	letterbox = Letterbox();
	numKeypoints = 0;
}

ofRectangle PoseResult::getBoxInSource(size_t index) const {
	if (index >= poses.size()) return ofRectangle();

	const ofRectangle & box = poses[index].detection.box;
	const glm::vec2 topLeft = letterbox.destToSource(glm::vec2(box.getMinX(), box.getMinY()));
	const glm::vec2 bottomRight = letterbox.destToSource(glm::vec2(box.getMaxX(), box.getMaxY()));
	return ofRectangle(topLeft.x, topLeft.y, bottomRight.x - topLeft.x, bottomRight.y - topLeft.y);
}

glm::vec2 PoseResult::getKeypointInSource(size_t index, int keypoint) const {
	if (index >= poses.size()) return glm::vec2(0.0f);
	const std::vector<Keypoint> & kps = poses[index].keypoints;
	if (keypoint < 0 || keypoint >= int(kps.size())) return glm::vec2(0.0f);
	return letterbox.destToSource(kps[keypoint].position);
}

float PoseResult::getKeypointScore(size_t index, int keypoint) const {
	if (index >= poses.size()) return 0.0f;
	const std::vector<Keypoint> & kps = poses[index].keypoints;
	if (keypoint < 0 || keypoint >= int(kps.size())) return 0.0f;
	return kps[keypoint].score;
}

//--------------------------------------------------------------
// Pose
//--------------------------------------------------------------

bool Pose::setup(const std::string & modelPath, const Settings & settings) {
	result.clear();
	return load(modelPath, settings);
}

bool Pose::onLoaded() {
	if (num_outputs < 1) {
		ofLogError("ofxYolo26::Pose") << "model has no outputs";
		return false;
	}

	const std::vector<int64_t> shape = getOutputShape(0);
	if (shape.size() != 3 || shape[0] != 1) {
		ofLogError("ofxYolo26::Pose") << "expected an end2end pose output [1, N, C]";
		return false;
	}

	numProposals = int(shape[1]);
	proposalStride = int(shape[2]);

	// 4 box + 1 score + 1 class, then 3 floats per keypoint.
	if (proposalStride < 6 || (proposalStride - 6) % 3 != 0) {
		ofLogError("ofxYolo26::Pose") << "unexpected pose stride " << proposalStride
									  << "; expected 6 + 3 * numKeypoints";
		return false;
	}
	numKeypoints = (proposalStride - 6) / 3;
	result.numKeypoints = numKeypoints;

	if (numKeypoints != KP_COUNT) {
		ofLogWarning("ofxYolo26::Pose") << "model has " << numKeypoints
										<< " keypoints; the COCO helpers assume " << int(KP_COUNT);
	}

	const std::string task = getMetadata("task");
	if (!task.empty() && task != "pose") {
		ofLogWarning("ofxYolo26::Pose") << "model reports task '" << task << "' rather than 'pose'";
	}
	return true;
}

const PoseResult & Pose::update(const ofPixels & pixels) {
	if (!isLoaded()) return result;

	preprocess(pixels);

	result.poses.clear();
	result.letterbox = letterbox;
	result.numKeypoints = numKeypoints;

	if (!runInference()) return result;

	const Ort::Value & value = output_values[0];
	if (!value || !value.IsTensor()) {
		ofLogError("ofxYolo26::Pose") << "pose output is missing";
		return result;
	}

	const std::vector<int64_t> shape = value.GetTensorTypeAndShapeInfo().GetShape();
	if (shape.size() != 3) return result;
	const int proposals = int(shape[1]);
	const int stride = int(shape[2]);
	if (stride != proposalStride) {
		ofLogError("ofxYolo26::Pose") << "pose stride changed to " << stride << " at runtime";
		return result;
	}

	const float * data = getOutputData(0);
	if (data == nullptr) return result;

	// The export is end2end, so this is a threshold and a sort — no NMS.
	for (int i = 0; i < proposals; i++) {
		const float * p = data + size_t(i) * size_t(stride);
		const float score = p[4];
		if (score < scoreThreshold) continue;

		PoseInstance pose;
		pose.detection.box = ofRectangle(p[0], p[1], p[2] - p[0], p[3] - p[1]);
		pose.detection.score = score;
		pose.detection.label = int(p[5]);
		pose.detection.labelName = getClassName(pose.detection.label);

		pose.keypoints.resize(numKeypoints);
		for (int k = 0; k < numKeypoints; k++) {
			const float * kp = p + 6 + k * 3;
			pose.keypoints[k].position = glm::vec2(kp[0], kp[1]);
			pose.keypoints[k].score = kp[2];
		}
		result.poses.push_back(std::move(pose));
	}

	std::stable_sort(result.poses.begin(), result.poses.end(),
		[](const PoseInstance & a, const PoseInstance & b) {
			return a.detection.score > b.detection.score;
		});

	if (maxPoses > 0 && int(result.poses.size()) > maxPoses) {
		result.poses.resize(maxPoses);
	}
	return result;
}

//--------------------------------------------------------------
// Drawing
//--------------------------------------------------------------

void drawPoses(const PoseResult & result,
	const ofRectangle & rect,
	float keypointThreshold,
	bool drawBoxes) {

	if (result.letterbox.srcWidth <= 0 || result.letterbox.srcHeight <= 0) return;

	// Source-image pixels -> the rectangle the frame was drawn into.
	const float sx = rect.getWidth() / float(result.letterbox.srcWidth);
	const float sy = rect.getHeight() / float(result.letterbox.srcHeight);
	const auto toScreen = [&](const glm::vec2 & src) {
		return glm::vec2(rect.x + src.x * sx, rect.y + src.y * sy);
	};

	const std::vector<glm::ivec2> & skeleton = getCocoSkeleton();
	ofPushStyle();

	for (size_t i = 0; i < result.poses.size(); i++) {
		const ofColor color = classColor(int(i));

		if (drawBoxes) {
			const ofRectangle box = result.getBoxInSource(i);
			ofNoFill();
			ofSetColor(color, 140);
			ofDrawRectangle(rect.x + box.x * sx, rect.y + box.y * sy,
				box.getWidth() * sx, box.getHeight() * sy);
			ofFill();
		}

		ofSetColor(color);
		ofSetLineWidth(2.0f);
		for (const glm::ivec2 & bone : skeleton) {
			if (bone.x >= result.numKeypoints || bone.y >= result.numKeypoints) continue;
			if (result.getKeypointScore(i, bone.x) < keypointThreshold) continue;
			if (result.getKeypointScore(i, bone.y) < keypointThreshold) continue;

			const glm::vec2 a = toScreen(result.getKeypointInSource(i, bone.x));
			const glm::vec2 b = toScreen(result.getKeypointInSource(i, bone.y));
			ofDrawLine(a.x, a.y, b.x, b.y);
		}

		ofSetColor(255);
		for (int k = 0; k < result.numKeypoints; k++) {
			if (result.getKeypointScore(i, k) < keypointThreshold) continue;
			const glm::vec2 p = toScreen(result.getKeypointInSource(i, k));
			ofDrawCircle(p.x, p.y, 2.5f);
		}
	}

	ofSetLineWidth(1.0f);
	ofPopStyle();
}

} // namespace ofxYolo26
