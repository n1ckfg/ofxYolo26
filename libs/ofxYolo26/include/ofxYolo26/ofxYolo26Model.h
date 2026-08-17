#pragma once

#include "ofxYolo26Types.h"
#include "ofxYolo26Preprocessor.h"

namespace ofxYolo26 {

/// Shared plumbing for every YOLO26 export: session creation, IO introspection
/// and preprocessing into the session's own input buffer.
///
/// Inherits from ofxOnnxRuntime::BaseHandler, which owns the Ort::Session and
/// the float input buffer. Subclasses read raw results out of `output_values`
/// after calling runInference().
class Model : public ofxOnnxRuntime::BaseHandler {
public:
	virtual ~Model() { }

	/// `modelPath` is resolved with ofToDataPath(), so "models/yolo26n-depth.onnx"
	/// refers to bin/data/models/yolo26n-depth.onnx.
	bool load(const std::string & modelPath, const Settings & settings = Settings());

	bool isLoaded() const { return loaded; }

	int getInputWidth() const { return inputWidth; }
	int getInputHeight() const { return inputHeight; }
	int getInputChannels() const { return inputChannels; }

	int getNumOutputs() const { return num_outputs; }
	std::string getInputName() const;
	std::string getOutputName(int index) const;
	std::vector<int64_t> getOutputShape(int index) const;

	/// Value from the model's custom metadata map, or "" when absent.
	/// YOLO exports carry useful keys here: "task", "imgsz", "names", "stride".
	std::string getMetadata(const std::string & key) const;
	const std::vector<std::string> & getMetadataKeys() const { return metadataKeys; }

	/// Class names parsed from the "names" metadata entry, indexed by class id.
	/// Empty when the export did not record them.
	const std::vector<std::string> & getClassNames() const { return classNames; }
	std::string getClassName(int label) const;

	/// Wall-clock time of the last Session::Run, in milliseconds.
	float getLastInferenceTimeMs() const { return lastInferenceMs; }

	const Settings & getSettings() const { return settings; }
	const std::string & getModelPath() const { return modelPath; }

	/// Mapping produced by the most recent call to preprocess().
	const Letterbox & getLetterbox() const { return letterbox; }

protected:
	/// Resamples `pixels` straight into the session's input buffer.
	void preprocess(const ofPixels & pixels);

	/// Runs the session over whatever is currently in the input buffer.
	/// Results land in the inherited `output_values`.
	bool runInference();

	/// Typed view of one output tensor, or nullptr when unavailable.
	const float * getOutputData(int index) const;

	/// Called after the session exists so subclasses can validate shapes.
	/// Returning false fails the load.
	virtual bool onLoaded() { return true; }

	Settings settings;
	std::string modelPath;
	bool loaded = false;

	int inputWidth = 0;
	int inputHeight = 0;
	int inputChannels = 0;

	Letterbox letterbox;
	float lastInferenceMs = 0.0f;

	std::vector<std::string> metadataKeys;
	std::map<std::string, std::string> metadataMap;
	std::vector<std::string> classNames;

private:
	void readMetadata();
	void parseClassNames();
	void logModelInfo() const;
};

} // namespace ofxYolo26
