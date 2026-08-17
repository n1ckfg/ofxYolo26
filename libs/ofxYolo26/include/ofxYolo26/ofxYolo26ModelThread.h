#pragma once

#include "ofxYolo26Model.h"

namespace ofxYolo26 {

/// Runs any ofxYolo26 model on a background thread so a live video source keeps
/// its framerate.
///
/// Inference costs tens of milliseconds on CPU, which would otherwise stall the
/// render thread. Frames handed in while the worker is busy are dropped rather
/// than queued, so results stay as close to live as the model allows.
///
/// `ModelType` needs `setup(path, settings)`, `update(const ofPixels &)`
/// returning a `const ResultType &`, and `getLastInferenceTimeMs()`. The
/// concrete aliases below (DepthThread, PoseThread, SegmentationThread) add
/// task-named accessors on top.
///
/// Typical use:
///     setup(); start();
///     update():  thread.setInput(grabber.getPixels());
///                if (thread.isFrameNew()) { ... thread.getResult() ... }
template <typename ModelType, typename ResultType>
class ModelThread : public ofThread {
public:
	virtual ~ModelThread() { stop(); }

	/// Must be called before start(). Loads the model on the calling thread.
	bool setup(const std::string & modelPath, const Settings & settings = Settings()) {
		if (isThreadRunning()) {
			ofLogError("ofxYolo26::ModelThread") << "setup() called while the thread is running";
			return false;
		}
		return model.setup(modelPath, settings);
	}

	void start() {
		if (!model.isLoaded()) {
			ofLogError("ofxYolo26::ModelThread") << "start() called before a model was loaded";
			return;
		}
		if (isThreadRunning()) return;
		startThread();
	}

	/// Ends the worker. The channels close with it, so a stopped thread cannot
	/// be restarted.
	void stop() {
		toAnalyze.close();
		fromAnalysis.close();
		waitForThread(false);
	}

	/// Hands a frame to the worker. Returns false when a frame is already in
	/// flight, in which case this one is dropped.
	bool setInput(const ofPixels & pixels) {
		if (!isThreadRunning() || !pixels.isAllocated()) return false;
		if (pending.load() > 0) return false;

		pending++;
		if (!toAnalyze.send(pixels)) {
			pending--;
			return false;
		}
		return true;
	}

	/// Consumes any pending result. True once per completed inference.
	bool isFrameNew() {
		bool gotResult = false;
		ResultType incoming;
		while (fromAnalysis.tryReceive(incoming)) {
			result = std::move(incoming);
			gotResult = true;
		}

		if (gotResult) {
			const uint64_t now = ofGetElapsedTimeMicros();
			if (lastResultMicros > 0 && now > lastResultMicros) {
				const float instant = 1000000.0f / float(now - lastResultMicros);
				fps.store(ofLerp(fps.load(), instant, 0.2f));
			}
			lastResultMicros = now;
		}
		return gotResult;
	}

	const ResultType & getResult() const { return result; }

	/// The underlying model, for metadata, IO shapes and thresholds. Do not call
	/// update() on it while the thread is running.
	const ModelType & getModel() const { return model; }

	/// Non-const access for configuration. Only safe before start().
	ModelType & getModelRef() { return model; }

	bool isBusy() const { return pending.load() > 0; }
	float getLastInferenceTimeMs() const { return lastInferenceMs.load(); }

	/// Completed inferences per second, smoothed.
	float getFps() const { return fps.load(); }

protected:
	void threadedFunction() override {
		ofPixels pixels;
		while (toAnalyze.receive(pixels)) {
			const ResultType & computed = model.update(pixels);
			lastInferenceMs.store(model.getLastInferenceTimeMs());
			fromAnalysis.send(computed);
			pending--;
		}
	}

	ModelType model;
	ResultType result;

	ofThreadChannel<ofPixels> toAnalyze;
	ofThreadChannel<ResultType> fromAnalysis;

	std::atomic<int> pending { 0 };
	std::atomic<float> lastInferenceMs { 0.0f };
	std::atomic<float> fps { 0.0f };
	uint64_t lastResultMicros = 0;
};

} // namespace ofxYolo26
