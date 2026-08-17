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

/// Execution provider to run inference on.
///
/// onnxruntime has no Metal or MPS provider; on Apple hardware the accelerated
/// path is CoreML, which dispatches each supported subgraph to the Neural
/// Engine, the GPU or the CPU as CoreML sees fit.
enum class Backend {
	CPU,
	CoreML,
	CUDA,
	TensorRT
};

/// Which processors CoreML may dispatch to.
///
/// `All` lets CoreML spread the graph across all of them and is the fastest by
/// some margin — measured on yolo26n-pose, 640x640, Apple Silicon:
///
///     CPU backend           29.2 ms
///     All                    5.6 ms
///     CPUAndGPU              9.6 ms
///     CPUAndNeuralEngine    22.9 ms
///
/// Restricting to the Neural Engine is *slower* than letting CoreML choose:
/// the parts of a YOLO graph the ANE will not take fall back to CPU rather
/// than to the GPU.
///
/// Accelerated paths may compute at reduced precision, but on these models the
/// divergence from the CPU provider is negligible: depth and segmentation agree
/// to within 1e-5, and pose scores differ by at most 4e-5 with keypoints landing
/// on the same pixel. Use CPUOnly (or Backend::CPU) if you need a bit-exact
/// match anyway.
enum class CoreMLComputeUnits {
	All,
	CPUAndNeuralEngine,
	CPUAndGPU,
	CPUOnly
};

/// CoreML where the runtime provides it, CPU otherwise.
Backend getDefaultBackend();

const char * toString(Backend backend);

/// Session and preprocessing options shared by every ofxYolo26 model.
struct Settings {
	/// Defaults to CoreML on Apple hardware, CPU elsewhere.
	Backend backend = getDefaultBackend();
	int deviceId = 0;

	/// Fall back to the CPU provider, with a warning, when the requested
	/// backend is unavailable rather than failing the load.
	bool fallbackToCPU = true;

	/// CoreML: which processors it may dispatch to.
	CoreMLComputeUnits coreMLComputeUnits = CoreMLComputeUnits::All;

	/// CoreML: use the newer MLProgram model format, which has better operator
	/// coverage than the legacy NeuralNetwork format and so leaves less of the
	/// graph falling back to CPU.
	bool coreMLUseMLProgram = true;

	/// CoreML: directory for compiled CoreML models, resolved with
	/// ofToDataPath() and created if missing. CoreML compilation costs a second
	/// or two per model; caching pays that once instead of on every launch.
	/// Empty uses a temp directory that is discarded when the session closes.
	///
	/// The cache is keyed on the model path, and onnxruntime does not check
	/// whether the file changed — clear this directory if you replace a model
	/// without renaming it.
	std::string coreMLCacheDirectory = "coreml-cache";

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
