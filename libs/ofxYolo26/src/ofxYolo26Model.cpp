#include "ofxYolo26Model.h"

namespace ofxYolo26 {

namespace {

	/// onnxruntime hands back allocator-owned strings that the caller must free.
	std::string takeOrtString(Ort::AllocatorWithDefaultOptions & allocator, char * raw) {
		if (raw == nullptr) return std::string();
		std::string out(raw);
		allocator.Free(raw);
		return out;
	}

	std::string shapeToString(const std::vector<int64_t> & shape) {
		std::string s = "[";
		for (size_t i = 0; i < shape.size(); i++) {
			if (i > 0) s += ", ";
			s += ofToString(shape[i]);
		}
		return s + "]";
	}

} // namespace

bool Model::load(const std::string & path, const Settings & s) {
	loaded = false;
	settings = s;
	modelPath = path;

	const std::string resolved = ofToDataPath(path, true);
	if (!ofFile::doesFileExist(resolved)) {
		ofLogError("ofxYolo26::Model") << "model not found: " << resolved;
		return false;
	}

	Ort::SessionOptions sessionOptions;
	sessionOptions.SetGraphOptimizationLevel(settings.graphOptimizationLevel);
	if (settings.intraOpNumThreads > 0) sessionOptions.SetIntraOpNumThreads(settings.intraOpNumThreads);
	if (settings.interOpNumThreads > 0) sessionOptions.SetInterOpNumThreads(settings.interOpNumThreads);

	if (settings.inferType == ofxOnnxRuntime::INFER_TENSORRT) {
		OrtTensorRTProviderOptions op;
		memset(&op, 0, sizeof(op));
		op.device_id = settings.deviceId;
		op.trt_fp16_enable = 1;
		op.trt_engine_cache_enable = 1;
		std::string cachePath = resolved;
		ofStringReplace(cachePath, ".onnx", "_trt_cache");
		op.trt_engine_cache_path = cachePath.c_str();
		sessionOptions.AppendExecutionProvider_TensorRT(op);
	}
	if (settings.inferType == ofxOnnxRuntime::INFER_CUDA || settings.inferType == ofxOnnxRuntime::INFER_TENSORRT) {
		OrtCUDAProviderOptions op;
		op.device_id = settings.deviceId;
		sessionOptions.AppendExecutionProvider_CUDA(op);
	}

	try {
		// BaseHandler::setup2 creates the session and sizes the input buffer from
		// the model's declared input shape.
		setup2(path, sessionOptions);
	} catch (const Ort::Exception & e) {
		ofLogError("ofxYolo26::Model") << "failed to create session for " << resolved << ": " << e.what();
		return false;
	}

	// YOLO exports are NCHW with a fixed batch of 1.
	if (input_node_dims.size() != 4) {
		ofLogError("ofxYolo26::Model") << "expected a 4D NCHW input, got " << shapeToString(input_node_dims);
		return false;
	}
	inputChannels = int(input_node_dims[1]);
	inputHeight = int(input_node_dims[2]);
	inputWidth = int(input_node_dims[3]);

	if (input_node_dims[0] != 1 || inputChannels <= 0 || inputHeight <= 0 || inputWidth <= 0) {
		ofLogError("ofxYolo26::Model") << "unsupported input shape " << shapeToString(input_node_dims)
									   << "; ofxYolo26 needs a static batch-1 shape";
		return false;
	}
	if (inputChannels != 3) {
		ofLogError("ofxYolo26::Model") << "expected 3 input channels, got " << inputChannels;
		return false;
	}

	readMetadata();
	loaded = true;

	if (!onLoaded()) {
		loaded = false;
		return false;
	}

	if (settings.verbose) logModelInfo();
	return true;
}

void Model::readMetadata() {
	metadataKeys.clear();
	metadataMap.clear();

	try {
		Ort::AllocatorWithDefaultOptions allocator;
		Ort::ModelMetadata meta = ort_session->GetModelMetadata();

		int64_t numKeys = 0;
		char ** keys = meta.GetCustomMetadataMapKeys(allocator, numKeys);
		if (keys == nullptr) return;

		for (int64_t i = 0; i < numKeys; i++) {
			const std::string key = takeOrtString(allocator, keys[i]);
			if (key.empty()) continue;
			metadataKeys.push_back(key);
			metadataMap[key] = takeOrtString(allocator, meta.LookupCustomMetadataMap(key.c_str(), allocator));
		}
		allocator.Free(keys);
	} catch (const Ort::Exception & e) {
		ofLogVerbose("ofxYolo26::Model") << "could not read model metadata: " << e.what();
	}
}

void Model::logModelInfo() const {
	ofLogNotice("ofxYolo26::Model") << ofFilePath::getFileName(modelPath)
									<< " task=" << getMetadata("task")
									<< " input " << getInputName() << " " << shapeToString(input_node_dims);
	for (int i = 0; i < num_outputs; i++) {
		ofLogNotice("ofxYolo26::Model") << "  output " << i << " " << getOutputName(i)
										<< " " << shapeToString(getOutputShape(i));
	}
}

std::string Model::getInputName() const {
	if (input_node_names.empty() || input_node_names[0] == nullptr) return std::string();
	return std::string(input_node_names[0]);
}

std::string Model::getOutputName(int index) const {
	if (index < 0 || index >= int(output_node_names.size()) || output_node_names[index] == nullptr) {
		return std::string();
	}
	return std::string(output_node_names[index]);
}

std::vector<int64_t> Model::getOutputShape(int index) const {
	if (index < 0 || index >= int(output_node_dims.size())) return {};
	return output_node_dims[index];
}

std::string Model::getMetadata(const std::string & key) const {
	auto it = metadataMap.find(key);
	return it == metadataMap.end() ? std::string() : it->second;
}

void Model::preprocess(const ofPixels & pixels) {
	if (!loaded) return;
	letterbox = Preprocessor::toTensorData(pixels,
		getInputTensorData(),
		inputWidth,
		inputHeight,
		settings.resizeMode,
		settings.padValue);
}

bool Model::runInference() {
	if (!loaded) return false;

	const uint64_t start = ofGetElapsedTimeMicros();
	try {
		BaseHandler::run();
	} catch (const Ort::Exception & e) {
		ofLogError("ofxYolo26::Model") << "inference failed: " << e.what();
		return false;
	}
	lastInferenceMs = (ofGetElapsedTimeMicros() - start) / 1000.0f;
	return true;
}

const float * Model::getOutputData(int index) const {
	if (index < 0 || index >= int(output_values.size())) return nullptr;
	const Ort::Value & value = output_values[index];
	if (!value) return nullptr;
	return value.GetTensorData<float>();
}

} // namespace ofxYolo26
