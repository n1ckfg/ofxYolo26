#pragma once

#include "ofxYolo26Detection.h"
#include "ofxYolo26Model.h"
#include "ofxYolo26ModelThread.h"

namespace ofxYolo26 {

/// A rotated rectangle, in model input-plane pixels.
///
/// The OBB head reports centre form (cx, cy, w, h, angle), unlike the detect
/// head's corner form — kept as-is here rather than flattened into an
/// axis-aligned box, since the rotation is the whole point.
///
/// `angle` is radians in image coordinates, where +Y runs *down*, so a positive
/// angle turns clockwise on screen. Convert with `mapAngleToBottomLeft()` if you
/// are feeding something that uses a bottom-left origin.
struct OrientedBox {
	glm::vec2 center { 0.0f, 0.0f };
	glm::vec2 size { 0.0f, 0.0f };
	float angle = 0.0f;

	int label = 0;
	std::string labelName;
	float score = 0.0f;

	/// Corners in model input-plane pixels, clockwise from the top-left of the
	/// unrotated box. Matches polygonFromXYWHR() in the reference web app.
	std::array<glm::vec2, 4> getCorners() const;

	/// Axis-aligned bounds of the rotated rectangle.
	ofRectangle getBoundingBox() const;
};

class ObbResult {
public:
	Letterbox letterbox;
	std::vector<OrientedBox> boxes;

	size_t size() const { return boxes.size(); }
	bool empty() const { return boxes.empty(); }
	void clear();

	/// Corners mapped back to source-image pixels.
	std::array<glm::vec2, 4> getCornersInSource(size_t index) const;

	/// Centre mapped back to source-image pixels.
	glm::vec2 getCenterInSource(size_t index) const;

	/// The angle as seen in the source image. Letterboxing scales x and y
	/// equally, so the angle is unchanged — this is here to make that explicit
	/// rather than leaving callers to wonder.
	float getAngleInSource(size_t index) const;
};

/// Oriented bounding boxes with a YOLO26 OBB export.
///
/// Input:  images,  float32 [batch, 3, height, width] — dynamic on the stock
///         export, pinned to the "imgsz" metadata (640x640) at load
/// Output: output0, float32 [batch, 300, 7]
///         per proposal: cx, cy, w, h, score, class, angle
///
/// End2end, so decoding is a score threshold and a sort. Note the column order:
/// the reference web app's header comment claims `cx, cy, w, h, angle, score,
/// cls`, but its code — and this model's actual output — put score at 4, class
/// at 5 and angle at 6.
///
/// The stock model is trained on DOTA, so its classes are aerial: plane, ship,
/// storage tank, harbour, roundabout and so on. It will find very little in an
/// ordinary photograph.
class Obb : public Model {
public:
	bool setup(const std::string & modelPath = "models/yolo26n-obb.onnx",
		const Settings & settings = Settings());

	/// Preprocesses and runs the model. Blocking; use ObbThread for live video.
	const ObbResult & update(const ofPixels & pixels);

	const ObbResult & getObbResult() const { return result; }

	/// Minimum detection score. Default 0.4, matching the reference's DET_SCORE_T.
	void setScoreThreshold(float threshold) { scoreThreshold = threshold; }
	float getScoreThreshold() const { return scoreThreshold; }

	/// Cap on returned boxes, highest score first. Default 100 (DET_TOPK).
	void setMaxDetections(int count) { maxDetections = count; }
	int getMaxDetections() const { return maxDetections; }

	/// Keep only these class ids. Empty (the default) keeps everything.
	void setClassFilter(const std::vector<int> & labels);
	void clearClassFilter() { classFilter.clear(); }

protected:
	bool onLoaded() override;

	ObbResult result;
	float scoreThreshold = 0.4f;
	int maxDetections = 100;
	std::vector<int> classFilter;
	int proposalStride = 0;
};

/// Oriented detection on a background thread. See ModelThread for the mechanics.
class ObbThread : public ModelThread<Obb, ObbResult> {
public:
	const ObbResult & getObbResult() const { return getResult(); }
	const Obb & getObb() const { return getModel(); }
};

/// Draws rotated boxes and labels, mapping source-image coordinates into `rect`
/// — pass the same rectangle you drew the frame into. A short spur marks the
/// box's local +X so the orientation is readable rather than merely present.
void drawOrientedBoxes(const ObbResult & result,
	const ofRectangle & rect,
	bool drawLabels = true,
	bool drawHeading = true);

/// Image-coordinate angle (+Y down) to a bottom-left origin (+Y up).
/// Matches mapAngleToBottomLeft() in the reference web app.
inline float mapAngleToBottomLeft(float angleRad) { return -angleRad; }

} // namespace ofxYolo26
