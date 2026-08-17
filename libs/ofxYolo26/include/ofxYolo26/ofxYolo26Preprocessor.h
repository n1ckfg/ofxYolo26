#pragma once

#include "ofxYolo26Types.h"

namespace ofxYolo26 {

/// Converts ofPixels into the tensor layout every YOLO26 export expects:
/// planar RGB (NCHW), float32, scaled to [0,1], with no mean/std normalisation.
/// This mirrors `toInputTensorFromImageData()` in the reference web app.
///
/// The resampler picks its filter per axis: a box (area) average when shrinking,
/// bilinear when growing. That approximates what the reference app gets for free
/// from canvas `drawImage()`, and matters a lot when feeding a 1280x720 camera
/// into a 640x640 input, where naive bilinear would alias badly.
class Preprocessor {
public:
	/// Writes 3 * dstHeight * dstWidth floats into `dst` and returns the mapping
	/// used, so results can be traced back to source coordinates.
	/// `dst` must already be allocated by the caller.
	static Letterbox toTensorData(const ofPixels & src,
		float * dst,
		int dstWidth,
		int dstHeight,
		ResizeMode mode = ResizeMode::Letterbox,
		float padValue = 0.0f);

private:
	/// Per-destination-index resampling taps along one axis.
	struct AxisTaps {
		std::vector<int> start; // first source index for each destination index
		std::vector<int> count; // number of source samples
		std::vector<float> weights; // maxCount entries per destination index
		int maxCount = 0;
	};

	static AxisTaps buildTaps(int srcSize, int dstSize);
};

} // namespace ofxYolo26
