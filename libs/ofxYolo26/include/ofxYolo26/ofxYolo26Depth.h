#pragma once

#include "ofxYolo26Model.h"

namespace ofxYolo26 {

/// A dense per-pixel depth result, in the model's own output units.
///
/// The values are passed through untouched, exactly as the reference web app
/// does before shipping them to TouchDesigner: no normalisation and no
/// inversion. Ultralytics describes the yolo26n-depth export as metric depth;
/// the file bundled with the reference project identifies itself as
/// "YOLO26n-depth-log", so treat the absolute scale as the model's business and
/// rely on the relative ordering. Larger values are further away.
///
/// Coordinates are in the model's input plane, so the padding introduced by
/// letterboxing is present in `values`. Use `letterbox` (or the cropToContent
/// option on the conversion helpers) to get back to source-image coordinates.
class DepthMap {
public:
	int width = 0;
	int height = 0;
	std::vector<float> values;
	Letterbox letterbox;

	/// Range over the non-padded content only, so letterbox bars do not skew
	/// auto-scaled visualisations.
	float minValue = 0.0f;
	float maxValue = 0.0f;

	bool isAllocated() const { return width > 0 && height > 0 && values.size() == size_t(width) * size_t(height); }
	void clear();

	/// Raw depth at a model-plane pixel. Coordinates are clamped.
	float getValue(int x, int y) const;

	/// Raw depth at a source-image pixel coordinate, bilinearly interpolated.
	/// Returns `outside` for coordinates that fall outside the source image.
	float getValueAtSource(float srcX, float srcY, float outside = 0.0f) const;

	/// Convenience for normalised source coordinates in [0,1].
	float getValueAtSourceNormalized(float u, float v, float outside = 0.0f) const;

	/// Recomputes minValue / maxValue over the content region.
	void updateRange();
};

/// Monocular depth estimation with a YOLO26 depth export.
///
/// Input:  images, float32 [1, 3, 640, 640], planar RGB in [0,1]
/// Output: output0, float32 [1, 1, 640, 640]
class Depth : public Model {
public:
	/// `modelPath` is resolved with ofToDataPath().
	bool setup(const std::string & modelPath = "models/yolo26n-depth.onnx",
		const Settings & settings = Settings());

	/// Preprocesses and runs the model. Blocking; expect tens of milliseconds on
	/// CPU. Use DepthThread for live video.
	const DepthMap & update(const ofPixels & pixels);

	const DepthMap & getDepthMap() const { return depthMap; }

	int getOutputWidth() const { return outputWidth; }
	int getOutputHeight() const { return outputHeight; }

protected:
	bool onLoaded() override;

	DepthMap depthMap;
	int outputWidth = 0;
	int outputHeight = 0;
};

/// Raw depth values as single-channel float pixels.
/// With cropToContent the letterbox padding is dropped, leaving an image that
/// lines up with the source frame.
void toFloatPixels(const DepthMap & map, ofFloatPixels & out, bool cropToContent = true);

/// Depth rescaled into [0,1]. Pass rangeMax <= rangeMin to auto-scale from the
/// map's own content range. `invert` makes near pixels bright.
void toNormalizedPixels(const DepthMap & map,
	ofFloatPixels & out,
	float rangeMin = 0.0f,
	float rangeMax = 0.0f,
	bool invert = false,
	bool cropToContent = true);

/// Depth through a blue-to-white colour ramp, over the same normalised range.
void toColorPixels(const DepthMap & map,
	ofPixels & out,
	float rangeMin = 0.0f,
	float rangeMax = 0.0f,
	bool invert = false,
	bool cropToContent = true);

/// The colour ramp used by toColorPixels(), for `t` in [0,1].
ofFloatColor depthRamp(float t);

} // namespace ofxYolo26
