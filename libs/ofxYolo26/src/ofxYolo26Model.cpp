#include "ofxYolo26Model.h"
#include "coreml_provider_factory.h"

namespace ofxYolo26 {

namespace {

	std::string shapeToString(const std::vector<int64_t> & shape) {
		std::string s = "[";
		for (size_t i = 0; i < shape.size(); i++) {
			if (i > 0) s += ", ";
			s += ofToString(shape[i]);
		}
		return s + "]";
	}

} // namespace

void Model::configureSessionOptions(Ort::SessionOptions & sessionOptions,
	Backend backend,
	const std::string & resolvedPath) {

	sessionOptions.SetGraphOptimizationLevel(settings.graphOptimizationLevel);
	if (settings.intraOpNumThreads > 0) sessionOptions.SetIntraOpNumThreads(settings.intraOpNumThreads);
	if (settings.interOpNumThreads > 0) sessionOptions.SetInterOpNumThreads(settings.interOpNumThreads);

	switch (backend) {
	case Backend::CPU:
		break;

	case Backend::CoreML: {
		std::unordered_map<std::string, std::string> options;
		options[kCoremlProviderOption_MLComputeUnits] = coreMLComputeUnitsString(settings.coreMLComputeUnits);
		options[kCoremlProviderOption_ModelFormat] = settings.coreMLUseMLProgram ? "MLProgram" : "NeuralNetwork";

		// Every ofxYolo26 export has a static input shape, and telling CoreML so
		// lets it take subgraphs it would otherwise refuse.
		options[kCoremlProviderOption_RequireStaticInputShapes] = "1";

		if (!settings.coreMLCacheDirectory.empty()) {
			const std::string cacheDir = ofToDataPath(settings.coreMLCacheDirectory, true);
			ofDirectory::createDirectory(cacheDir, false, true);
			options[kCoremlProviderOption_ModelCacheDirectory] = cacheDir;
		}
		sessionOptions.AppendExecutionProvider("CoreML", options);
		break;
	}

	case Backend::TensorRT: {
		OrtTensorRTProviderOptions op;
		memset(&op, 0, sizeof(op));
		op.device_id = settings.deviceId;
		op.trt_fp16_enable = 1;
		op.trt_engine_cache_enable = 1;
		std::string cachePath = resolvedPath;
		ofStringReplace(cachePath, ".onnx", "_trt_cache");
		op.trt_engine_cache_path = cachePath.c_str();
		sessionOptions.AppendExecutionProvider_TensorRT(op);
	}
		// fall through: TensorRT is layered on top of CUDA
		[[fallthrough]];

	case Backend::CUDA: {
		OrtCUDAProviderOptions op;
		op.device_id = settings.deviceId;
		sessionOptions.AppendExecutionProvider_CUDA(op);
		break;
	}
	}
}

const char * Model::coreMLComputeUnitsString(CoreMLComputeUnits units) {
	// Bare tokens, case-sensitive. The coreml_provider_factory.h shipped with
	// onnxruntime 1.29 documents "MLComputeUnitsAll" and friends, but the
	// runtime rejects those; only these four are accepted.
	switch (units) {
	case CoreMLComputeUnits::All: return "ALL";
	case CoreMLComputeUnits::CPUAndNeuralEngine: return "CPUAndNeuralEngine";
	case CoreMLComputeUnits::CPUAndGPU: return "CPUAndGPU";
	case CoreMLComputeUnits::CPUOnly: return "CPUOnly";
	}
	return "ALL";
}

bool Model::load(const std::string & path, const Settings & s) {
	loaded = false;
	settings = s;
	modelPath = path;

	const std::string resolved = ofToDataPath(path, true);
	if (!ofFile::doesFileExist(resolved)) {
		ofLogError("ofxYolo26::Model") << "model not found: " << resolved;
		return false;
	}

	// Requesting a provider the runtime was not built with throws rather than
	// degrading, so provider setup and session creation share one guard and one
	// retry on the CPU provider.
	activeBackend = settings.backend;
	try {
		Ort::SessionOptions sessionOptions;
		configureSessionOptions(sessionOptions, settings.backend, resolved);
		// BaseHandler::setup2 creates the session and sizes the input buffer from
		// the model's declared input shape.
		setup2(path, sessionOptions);
	} catch (const Ort::Exception & e) {
		if (settings.backend == Backend::CPU || !settings.fallbackToCPU) {
			ofLogError("ofxYolo26::Model") << "failed to create session for " << resolved
										   << " on " << toString(settings.backend) << ": " << e.what();
			return false;
		}

		ofLogWarning("ofxYolo26::Model")
			<< toString(settings.backend) << " unavailable for " << ofFilePath::getFileName(path)
			<< " (" << e.what() << "); falling back to CPU";

		activeBackend = Backend::CPU;
		try {
			Ort::SessionOptions cpuOptions;
			configureSessionOptions(cpuOptions, Backend::CPU, resolved);
			setup2(path, cpuOptions);
		} catch (const Ort::Exception & e2) {
			ofLogError("ofxYolo26::Model") << "failed to create session for " << resolved
										   << " on CPU: " << e2.what();
			return false;
		}
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

		for (const auto & keyPtr : meta.GetCustomMetadataMapKeysAllocated(allocator)) {
			if (keyPtr == nullptr) continue;
			const std::string key(keyPtr.get());
			if (key.empty()) continue;

			metadataKeys.push_back(key);
			const auto valuePtr = meta.LookupCustomMetadataMapAllocated(key.c_str(), allocator);
			metadataMap[key] = valuePtr ? std::string(valuePtr.get()) : std::string();
		}
	} catch (const Ort::Exception & e) {
		ofLogVerbose("ofxYolo26::Model") << "could not read model metadata: " << e.what();
	}

	parseClassNames();
}

void Model::parseClassNames() {
	classNames.clear();

	// Ultralytics writes a Python dict literal: {0: 'person', 1: 'bicycle', ...}.
	// Keys are contiguous from zero in every export seen so far, but read the
	// index rather than assuming it.
	const std::string names = getMetadata("names");
	if (names.empty()) return;

	std::map<int, std::string> byIndex;
	size_t pos = 0;
	while (pos < names.size()) {
		const size_t colon = names.find(':', pos);
		if (colon == std::string::npos) break;

		// Index is the run of digits immediately before the colon.
		size_t digitEnd = colon;
		while (digitEnd > pos && std::isspace((unsigned char)names[digitEnd - 1]))
			digitEnd--;
		size_t digitStart = digitEnd;
		while (digitStart > pos && std::isdigit((unsigned char)names[digitStart - 1]))
			digitStart--;
		if (digitStart == digitEnd) {
			pos = colon + 1;
			continue;
		}

		const char quote = names.find('\'', colon) < names.find('"', colon) ? '\'' : '"';
		const size_t open = names.find(quote, colon);
		if (open == std::string::npos) break;
		const size_t close = names.find(quote, open + 1);
		if (close == std::string::npos) break;

		byIndex[ofToInt(names.substr(digitStart, digitEnd - digitStart))]
			= names.substr(open + 1, close - open - 1);
		pos = close + 1;
	}

	if (byIndex.empty()) return;
	classNames.resize(byIndex.rbegin()->first + 1);
	for (const auto & entry : byIndex) {
		classNames[entry.first] = entry.second;
	}
}

std::string Model::getClassName(int label) const {
	if (label < 0 || label >= int(classNames.size()) || classNames[label].empty()) {
		return ofToString(label);
	}
	return classNames[label];
}

void Model::logModelInfo() const {
	ofLogNotice("ofxYolo26::Model") << ofFilePath::getFileName(modelPath)
									<< " task=" << getMetadata("task")
									<< " backend=" << toString(activeBackend)
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
