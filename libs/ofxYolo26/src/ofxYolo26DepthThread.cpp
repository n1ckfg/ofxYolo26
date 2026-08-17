#include "ofxYolo26DepthThread.h"

namespace ofxYolo26 {

DepthThread::~DepthThread() {
	stop();
}

bool DepthThread::setup(const std::string & modelPath, const Settings & settings) {
	if (isThreadRunning()) {
		ofLogError("ofxYolo26::DepthThread") << "setup() called while the thread is running";
		return false;
	}
	return depth.setup(modelPath, settings);
}

void DepthThread::start() {
	if (!depth.isLoaded()) {
		ofLogError("ofxYolo26::DepthThread") << "start() called before a model was loaded";
		return;
	}
	if (isThreadRunning()) return;
	startThread();
}

void DepthThread::stop() {
	toAnalyze.close();
	fromAnalysis.close();
	waitForThread(false);
}

bool DepthThread::setInput(const ofPixels & pixels) {
	if (!isThreadRunning() || !pixels.isAllocated()) return false;
	if (pending.load() > 0) return false;

	pending++;
	if (!toAnalyze.send(pixels)) {
		pending--;
		return false;
	}
	return true;
}

bool DepthThread::isFrameNew() {
	bool gotResult = false;
	DepthMap result;
	while (fromAnalysis.tryReceive(result)) {
		depthMap = std::move(result);
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

void DepthThread::threadedFunction() {
	ofPixels pixels;
	while (toAnalyze.receive(pixels)) {
		const DepthMap & result = depth.update(pixels);
		lastInferenceMs.store(depth.getLastInferenceTimeMs());
		fromAnalysis.send(result);
		pending--;
	}
}

} // namespace ofxYolo26
