#pragma once

#include "ofxYolo26Depth.h"

namespace ofxYolo26 {

/// Runs Depth on a background thread so a live video source keeps its framerate.
///
/// Inference costs tens of milliseconds on CPU, which would otherwise stall the
/// render thread. Frames handed in while the worker is busy are dropped rather
/// than queued, so results stay as close to live as the model allows.
///
/// Typical use:
///     setup(); start();
///     update():  thread.setInput(grabber.getPixels());
///                if (thread.isFrameNew()) { ... thread.getDepthMap() ... }
class DepthThread : public ofThread {
public:
	~DepthThread();

	/// Must be called before start(). Loads the model on the calling thread.
	bool setup(const std::string & modelPath = "models/yolo26n-depth.onnx",
		const Settings & settings = Settings());

	void start();
	void stop();

	/// Hands a frame to the worker. Returns false when a frame is already in
	/// flight, in which case this one is dropped.
	bool setInput(const ofPixels & pixels);

	/// Consumes any pending result. True once per completed inference.
	bool isFrameNew();

	const DepthMap & getDepthMap() const { return depthMap; }

	/// True while a frame is being processed.
	bool isBusy() const { return pending.load() > 0; }

	float getLastInferenceTimeMs() const { return lastInferenceMs.load(); }

	/// Completed inferences per second, smoothed.
	float getFps() const { return fps.load(); }

	/// The underlying model, for metadata and IO shapes. Do not call update()
	/// on it while the thread is running.
	const Depth & getDepth() const { return depth; }

protected:
	void threadedFunction() override;

	Depth depth;
	DepthMap depthMap;

	ofThreadChannel<ofPixels> toAnalyze;
	ofThreadChannel<DepthMap> fromAnalysis;

	std::atomic<int> pending { 0 };
	std::atomic<float> lastInferenceMs { 0.0f };
	std::atomic<float> fps { 0.0f };
	uint64_t lastResultMicros = 0;
};

} // namespace ofxYolo26
