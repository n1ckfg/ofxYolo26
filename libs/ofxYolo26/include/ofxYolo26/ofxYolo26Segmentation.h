#pragma once

#include "ofxYolo26Detection.h"
#include "ofxYolo26Model.h"
#include "ofxYolo26ModelThread.h"

namespace ofxYolo26 {

/// One instance's mask, in prototype-plane pixels (160x160 for the stock
/// exports — a quarter of the model input in each axis).
///
/// The mask is stored cropped to the detection's box, because that is the only
/// region where the prototype combination is meaningful: outside the box the
/// same coefficients light up unrelated parts of the image. Sampling outside
/// `rect` returns 0.
class InstanceMask {
public:
	/// Region of the prototype plane covered by `alpha`.
	ofRectangle rect;

	/// rect-sized, single channel, in [0,1].
	ofFloatPixels alpha;

	/// Alpha at a prototype pixel; 0 outside `rect`.
	float getAlpha(int protoX, int protoY) const;

	/// Bilinearly interpolated alpha at a fractional prototype coordinate.
	float getAlphaInterpolated(float protoX, float protoY) const;
};

class SegmentationResult {
public:
	Letterbox letterbox;

	int protoWidth = 0;
	int protoHeight = 0;

	/// Model input-plane pixels per prototype pixel (4 for a 640 input).
	float protoScaleX = 1.0f;
	float protoScaleY = 1.0f;

	/// One entry per instance; `detections` and `masks` stay index-aligned.
	std::vector<Detection> detections;
	std::vector<InstanceMask> masks;

	size_t size() const { return detections.size(); }
	bool empty() const { return detections.empty(); }
	void clear();

	/// Detection box mapped back to source-image pixels.
	ofRectangle getBoxInSource(size_t index) const;

	/// One instance's alpha at a source-image pixel.
	float getAlphaAtSource(size_t index, float srcX, float srcY) const;

	/// Highest alpha across every instance at a source-image pixel. When
	/// `instanceOut` is given it receives the winning index, or -1 for none.
	float getAlphaAtSource(float srcX, float srcY, int * instanceOut = nullptr) const;

	glm::vec2 sourceToProto(const glm::vec2 & src) const;
	glm::vec2 protoToSource(const glm::vec2 & proto) const;

	/// The part of the prototype plane that covers the source frame rather than
	/// letterbox padding, in prototype pixels.
	///
	/// Mask textures span the whole padded input, so drawing one straight over
	/// the video would be offset and squashed. Feed this to
	/// ofTexture::drawSubsection() to line them up.
	ofRectangle getContentRectInProto() const;
};

/// Instance segmentation with a YOLO26 segmentation export.
///
/// Input:  images,  float32 [1, 3, 640, 640], planar RGB in [0,1]
/// Output: output0, float32 [1, 300, 38]  x1, y1, x2, y2, score, class,
///                                        then 32 prototype coefficients
///         output1, float32 [1, 32, 160, 160]  mask prototypes
///
/// A mask is sigmoid(sum of coefficient * prototype) evaluated inside the
/// detection's box, exactly as the reference web app's compute shader does.
///
/// Unlike the reference, instances are kept separate rather than composited
/// into one float map with the class id packed into the integer part and alpha
/// into the fraction. That encoding exists to squeeze masks through a
/// WebSocket into a single-channel TouchDesigner TOP; here you can just iterate
/// the instances. toColorPixels() still produces a composited overlay when a
/// single texture is what you want.
class Segmentation : public Model {
public:
	bool setup(const std::string & modelPath = "models/yolo26n-seg.onnx",
		const Settings & settings = Settings());

	/// Preprocesses and runs the model. Blocking; use SegmentationThread for
	/// live video.
	const SegmentationResult & update(const ofPixels & pixels);

	const SegmentationResult & getSegmentationResult() const { return result; }

	/// Minimum detection score. Default 0.2, matching the reference's SEG_SCORE_T.
	void setScoreThreshold(float threshold) { scoreThreshold = threshold; }
	float getScoreThreshold() const { return scoreThreshold; }

	/// IoU for the duplicate-suppression pass the reference runs over
	/// segmentation results. Default 0.45; set to 1 to disable.
	void setIouThreshold(float threshold) { iouThreshold = threshold; }
	float getIouThreshold() const { return iouThreshold; }

	/// Cap on returned instances, highest score first. Default 100 (SEG_TOPK).
	void setMaxInstances(int count) { maxInstances = count; }
	int getMaxInstances() const { return maxInstances; }

	/// Alpha below this is clamped to zero and the remainder rescaled, which
	/// clears the low-level noise the prototypes leave across the whole box.
	/// Default 0.01, matching the reference. Set to 0 for the raw sigmoid.
	void setMaskNoiseFloor(float floorValue) { maskNoiseFloor = floorValue; }
	float getMaskNoiseFloor() const { return maskNoiseFloor; }

	/// Keep only these class ids. Empty (the default) keeps everything.
	/// `{0}` reproduces the reference's PERSON_SEG_ONLY.
	void setClassFilter(const std::vector<int> & labels);
	void clearClassFilter() { classFilter.clear(); }

	int getNumPrototypes() const { return numPrototypes; }

protected:
	bool onLoaded() override;

	/// Locates the detection and prototype outputs by rank, since their order
	/// is not guaranteed.
	bool findOutputIndices(int & detIndex, int & protoIndex) const;

	SegmentationResult result;

	float scoreThreshold = 0.2f;
	float iouThreshold = 0.45f;
	int maxInstances = 100;
	float maskNoiseFloor = 0.01f;
	std::vector<int> classFilter;

	int numPrototypes = 0;
	int proposalStride = 0;
};

/// Instance segmentation on a background thread. See ModelThread for the mechanics.
class SegmentationThread : public ModelThread<Segmentation, SegmentationResult> {
public:
	const SegmentationResult & getSegmentationResult() const { return getResult(); }
	const Segmentation & getSegmentation() const { return getModel(); }
};

/// Union of every instance's alpha, at prototype resolution.
void toAlphaPixels(const SegmentationResult & result, ofFloatPixels & out);

/// One instance's alpha over the full prototype plane.
void toAlphaPixels(const SegmentationResult & result, ofFloatPixels & out, size_t index);

/// Composited RGBA overlay at prototype resolution: RGB is the class colour (or
/// a per-instance colour), A is that pixel's winning alpha. Drawn over the
/// source frame it tints each instance in place.
///
/// Instances are painted largest first, so small foreground objects stay on top
/// — the painter's ordering the reference uses.
void toColorPixels(const SegmentationResult & result, ofPixels & out, bool colorByInstance = false);

} // namespace ofxYolo26
