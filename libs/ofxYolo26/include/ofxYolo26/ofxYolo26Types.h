#pragma once

#include "ofMain.h"
#include "ofxOnnxRuntime.h"

namespace ofxYolo26 {

/// How a source image of arbitrary size is fitted into the model's fixed input.
///
/// Letterbox preserves aspect ratio and pads the remainder with a flat colour.
/// This matches the webcam path of the yolo-touchdesigner reference app, which
/// draws the video centred into a black 640x640 canvas.
///
/// Stretch fills the whole input, distorting aspect ratio. It matches the
/// TouchDesigner binary path, where the host has already sized the frame.
enum class ResizeMode {
	Letterbox,
	Stretch
};

/// Mapping between a source image and the model's fixed-size input plane.
///
/// Model outputs are in *destination* (input-plane) coordinates, so this is
/// what you need to relate a depth pixel, a bounding box or a keypoint back to
/// a pixel of the image you handed in.
struct Letterbox {
	int srcWidth = 0;
	int srcHeight = 0;
	int dstWidth = 0;
	int dstHeight = 0;

	/// Size of the actual image content inside the destination plane.
	/// Equal to dstWidth/dstHeight in Stretch mode.
	int contentWidth = 0;
	int contentHeight = 0;

	/// Top-left corner of the content inside the destination plane.
	int padX = 0;
	int padY = 0;

	/// contentWidth / srcWidth and contentHeight / srcHeight.
	float scaleX = 1.0f;
	float scaleY = 1.0f;

	static Letterbox make(int srcW, int srcH, int dstW, int dstH, ResizeMode mode);

	/// Source-image pixel coordinate -> destination (model input) pixel coordinate.
	glm::vec2 sourceToDest(const glm::vec2 & src) const;

	/// Destination (model input) pixel coordinate -> source-image pixel coordinate.
	/// Points inside the padding map outside the source image bounds.
	glm::vec2 destToSource(const glm::vec2 & dst) const;

	/// True when a destination pixel falls on real image content rather than padding.
	bool destContains(float dstX, float dstY) const;

	/// The content region as a rectangle in destination coordinates.
	ofRectangle getContentRect() const;
};

/// Session and preprocessing options shared by every ofxYolo26 model.
struct Settings {
	/// Execution provider. The macOS build of onnxruntime shipped with
	/// ofxOnnxRuntime is CPU-only, so INFER_CUDA / INFER_TENSORRT are for
	/// platforms where you have swapped in a matching runtime.
	ofxOnnxRuntime::InferType inferType = ofxOnnxRuntime::INFER_CPU;
	int deviceId = 0;

	/// 0 leaves the onnxruntime default (one thread per physical core).
	/// Lower this when running inference on a background thread so it does not
	/// starve the render thread.
	int intraOpNumThreads = 0;
	int interOpNumThreads = 0;

	/// Matches `graphOptimizationLevel: "all"` in the reference web app.
	GraphOptimizationLevel graphOptimizationLevel = ORT_ENABLE_ALL;

	ResizeMode resizeMode = ResizeMode::Letterbox;

	/// Value written into the letterbox padding, in normalised [0,1] units.
	/// The reference app fills with black.
	float padValue = 0.0f;

	/// Log model IO shapes and metadata on load.
	bool verbose = true;
};

} // namespace ofxYolo26
