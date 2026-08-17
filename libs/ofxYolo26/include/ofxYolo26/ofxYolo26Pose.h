#pragma once

#include "ofxYolo26Detection.h"
#include "ofxYolo26Model.h"
#include "ofxYolo26ModelThread.h"

namespace ofxYolo26 {

/// Indices into a COCO 17-keypoint pose, the layout every stock YOLO pose
/// export uses. Confirm against the model's "kpt_shape" metadata if you swap in
/// a custom export.
enum CocoKeypoint {
	KP_NOSE = 0,
	KP_LEFT_EYE,
	KP_RIGHT_EYE,
	KP_LEFT_EAR,
	KP_RIGHT_EAR,
	KP_LEFT_SHOULDER,
	KP_RIGHT_SHOULDER,
	KP_LEFT_ELBOW,
	KP_RIGHT_ELBOW,
	KP_LEFT_WRIST,
	KP_RIGHT_WRIST,
	KP_LEFT_HIP,
	KP_RIGHT_HIP,
	KP_LEFT_KNEE,
	KP_RIGHT_KNEE,
	KP_LEFT_ANKLE,
	KP_RIGHT_ANKLE,
	KP_COUNT
};

const std::vector<std::string> & getCocoKeypointNames();

/// Keypoint index pairs forming the skeleton, for drawing bones.
const std::vector<glm::ivec2> & getCocoSkeleton();

struct Keypoint {
	/// Model input-plane pixels.
	glm::vec2 position { 0.0f, 0.0f };
	float score = 0.0f;
};

struct PoseInstance {
	Detection detection;
	std::vector<Keypoint> keypoints;
};

class PoseResult {
public:
	Letterbox letterbox;
	int numKeypoints = 0;
	std::vector<PoseInstance> poses;

	size_t size() const { return poses.size(); }
	bool empty() const { return poses.empty(); }
	void clear();

	/// Detection box mapped back to source-image pixels.
	ofRectangle getBoxInSource(size_t index) const;

	/// Keypoint mapped back to source-image pixels.
	glm::vec2 getKeypointInSource(size_t index, int keypoint) const;
	float getKeypointScore(size_t index, int keypoint) const;
};

/// Multi-person pose estimation with a YOLO26 pose export.
///
/// Input:  images,  float32 [1, 3, 640, 640], planar RGB in [0,1]
/// Output: output0, float32 [1, 300, 57]
///         per proposal: x1, y1, x2, y2, score, class, then 17 * (x, y, score)
///
/// These exports are end2end — NMS runs inside the graph — so decoding is a
/// score threshold and a sort, exactly as decodeYOLOv26Pose() does in the
/// reference web app.
class Pose : public Model {
public:
	bool setup(const std::string & modelPath = "models/yolo26n-pose.onnx",
		const Settings & settings = Settings());

	/// Preprocesses and runs the model. Blocking; use PoseThread for live video.
	const PoseResult & update(const ofPixels & pixels);

	const PoseResult & getPoseResult() const { return result; }

	/// Minimum detection score. Default 0.35, matching the reference's POSE_SCORE_T.
	void setScoreThreshold(float threshold) { scoreThreshold = threshold; }
	float getScoreThreshold() const { return scoreThreshold; }

	/// Cap on returned poses, highest score first. Default 50 (POSE_TOPK).
	void setMaxPoses(int count) { maxPoses = count; }
	int getMaxPoses() const { return maxPoses; }

	int getNumKeypoints() const { return numKeypoints; }

protected:
	bool onLoaded() override;

	PoseResult result;
	float scoreThreshold = 0.35f;
	int maxPoses = 50;
	int numKeypoints = 0;
	int numProposals = 0;
	int proposalStride = 0;
};

/// Pose estimation on a background thread. See ModelThread for the mechanics.
class PoseThread : public ModelThread<Pose, PoseResult> {
public:
	const PoseResult & getPoseResult() const { return getResult(); }
	const Pose & getPose() const { return getModel(); }
};

/// Draws skeletons and keypoints, mapping source-image coordinates into `rect`
/// — pass the same rectangle you drew the frame into. Bones are drawn only
/// where both endpoints clear `keypointThreshold`.
void drawPoses(const PoseResult & result,
	const ofRectangle & rect,
	float keypointThreshold = 0.3f,
	bool drawBoxes = true);

} // namespace ofxYolo26
