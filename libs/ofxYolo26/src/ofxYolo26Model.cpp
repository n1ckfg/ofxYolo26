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
	const std::string & resolvedPath,
	int pinWidth,
	int pinHeight) {

	sessionOptions.SetGraphOptimizationLevel(settings.graphOptimizationLevel);
	if (settings.intraOpNumThreads > 0) sessionOptions.SetIntraOpNumThreads(settings.intraOpNumThreads);
	if (settings.interOpNumThreads > 0) sessionOptions.SetInterOpNumThreads(settings.interOpNumThreads);

	// Pin the free axes of a dynamic export so the graph looks static to the
	// execution providers. This is what makes CoreML worth using on
	// yolo26n-obb: left free, it runs at CPU speed (23.8 ms vs 23.2 ms); pinned,
	// it drops to 8.5 ms. Ultralytics names its free axes batch/height/width,
	// and an override naming an axis the model does not have — or one that is
	// already a fixed size — is simply ignored.
	sessionOptions.AddFreeDimensionOverrideByName("batch", 1);
	sessionOptions.AddFreeDimensionOverrideByName("height", pinHeight);
	sessionOptions.AddFreeDimensionOverrideByName("width", pinWidth);

	switch (backend) {
	case Backend::CPU:
		break;

	case Backend::CoreML: {
		std::unordered_map<std::string, std::string> options;
		options[kCoremlProviderOption_MLComputeUnits] = coreMLComputeUnitsString(settings.coreMLComputeUnits);
		options[kCoremlProviderOption_ModelFormat] = settings.coreMLUseMLProgram ? "MLProgram" : "NeuralNetwork";

		// With the free axes pinned above, every shape is known, which lets
		// CoreML take subgraphs it would otherwise refuse. Measured on
		// yolo26n-obb this is worth 24 ms -> 8 ms; turning it off costs more
		// than leaving it on ever saves.
		options[kCoremlProviderOption_RequireStaticInputShapes] = "1";

		if (!settings.coreMLCacheDirectory.empty()) {
			// onnxruntime keys this cache on the model path alone. Anything that
			// changes how the graph gets partitioned — input size, compute units,
			// model format — then silently reuses a compiled model that no longer
			// matches, and inference fails with "Feature ... is required but not
			// specified". Replacing the .onnx in place has the same effect. So
			// give every configuration its own subdirectory.
			const std::string variant = ofFilePath::getBaseName(modelPath)
				+ "-" + ofToString(pinWidth) + "x" + ofToString(pinHeight)
				+ "-" + coreMLComputeUnitsString(settings.coreMLComputeUnits)
				+ "-" + (settings.coreMLUseMLProgram ? "mlprogram" : "neuralnetwork")
				+ "-" + modelFingerprint(resolvedPath);

			const std::string cacheDir = ofToDataPath(
				ofFilePath::join(settings.coreMLCacheDirectory, variant), true);
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

std::string Model::modelFingerprint(const std::string & resolvedPath) {
	// Size plus modification time is enough to notice a model swapped in place,
	// and costs nothing next to hashing 20 MB on every load.
	try {
		const std::filesystem::path p(resolvedPath);
		const auto size = std::filesystem::file_size(p);
		const auto written = std::filesystem::last_write_time(p).time_since_epoch().count();
		return ofToString(uint64_t(size)) + "-" + ofToString(uint64_t(written));
	} catch (const std::exception &) {
		return "nofingerprint";
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

bool Model::createSession(const std::string & path,
	const std::string & resolved,
	int pinWidth,
	int pinHeight) {

	// Requesting a provider the runtime was not built with throws rather than
	// degrading, so provider setup and session creation share one guard and one
	// retry on the CPU provider.
	activeBackend = settings.backend;
	try {
		Ort::SessionOptions sessionOptions;
		configureSessionOptions(sessionOptions, settings.backend, resolved, pinWidth, pinHeight);
		// BaseHandler::setup2 creates the session and sizes the input buffer from
		// the model's declared input shape.
		setup2(path, sessionOptions);
		return true;
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
			configureSessionOptions(cpuOptions, Backend::CPU, resolved, pinWidth, pinHeight);
			setup2(path, cpuOptions);
			return true;
		} catch (const Ort::Exception & e2) {
			ofLogError("ofxYolo26::Model") << "failed to create session for " << resolved
										   << " on CPU: " << e2.what();
			return false;
		}
	}
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

	// Free axes have to be pinned before the session exists, but the export's
	// "imgsz" metadata can only be read once it does. Start from the caller's
	// setting or the 640x640 every stock YOLO export uses, then rebuild below if
	// the model turns out to want something else.
	const bool callerPinned = settings.dynamicInputWidth > 0 || settings.dynamicInputHeight > 0;
	int pinWidth = settings.dynamicInputWidth > 0 ? settings.dynamicInputWidth : 640;
	int pinHeight = settings.dynamicInputHeight > 0 ? settings.dynamicInputHeight : 640;

	if (!createSession(path, resolved, pinWidth, pinHeight)) return false;
	readMetadata();

	if (!callerPinned) {
		int metaHeight = 0;
		int metaWidth = 0;
		parseImgsz(getMetadata("imgsz"), metaHeight, metaWidth);

		// Only meaningful if the pin actually took — on a genuinely static model
		// the override is ignored and the declared shape stands.
		const bool pinTook = input_node_dims.size() == 4
			&& input_node_dims[2] == pinHeight && input_node_dims[3] == pinWidth;

		if (pinTook && metaWidth > 0 && metaHeight > 0
			&& (metaWidth != pinWidth || metaHeight != pinHeight)) {

			ofLogNotice("ofxYolo26::Model")
				<< ofFilePath::getFileName(path) << " declares imgsz " << metaWidth << "x" << metaHeight
				<< "; rebuilding the session at that size";

			pinWidth = metaWidth;
			pinHeight = metaHeight;
			if (!createSession(path, resolved, pinWidth, pinHeight)) return false;
			readMetadata();
		}
	}

	// YOLO exports are NCHW with a batch of 1.
	if (input_node_dims.size() != 4) {
		ofLogError("ofxYolo26::Model") << "expected a 4D NCHW input, got " << shapeToString(input_node_dims);
		return false;
	}

	readMetadata();

	if (!resolveInputShape()) return false;

	inputChannels = int(input_node_dims[1]);
	inputHeight = int(input_node_dims[2]);
	inputWidth = int(input_node_dims[3]);

	if (inputChannels != 3) {
		ofLogError("ofxYolo26::Model") << "expected 3 input channels, got " << inputChannels;
		return false;
	}

	loaded = true;

	if (!onLoaded()) {
		loaded = false;
		return false;
	}

	if (settings.verbose) logModelInfo();
	return true;
}

bool Model::resolveInputShape() {
	// Dynamic axes come back as -1 from onnxruntime; BaseHandler::setup2 has
	// already clamped its copy to 1, so read the declared shape again to see
	// which axes were actually free.
	std::vector<int64_t> declared;
	try {
		declared = ort_session->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
	} catch (const Ort::Exception & e) {
		ofLogError("ofxYolo26::Model") << "could not read input shape: " << e.what();
		return false;
	}
	if (declared.size() != 4) return false;

	if (declared[0] > 1) {
		ofLogError("ofxYolo26::Model") << "expected a batch-1 input, got batch " << declared[0];
		return false;
	}
	if (declared[1] > 0 && declared[1] != 3) {
		ofLogError("ofxYolo26::Model") << "expected 3 input channels, got " << declared[1];
		return false;
	}

	const bool dynamicHeight = declared[2] < 1;
	const bool dynamicWidth = declared[3] < 1;

	int height = dynamicHeight ? 0 : int(declared[2]);
	int width = dynamicWidth ? 0 : int(declared[3]);

	if (dynamicHeight || dynamicWidth) {
		// Preference order: explicit setting, then the export's "imgsz"
		// metadata (Ultralytics writes it as [height, width]), then 640.
		int metaHeight = 0;
		int metaWidth = 0;
		parseImgsz(getMetadata("imgsz"), metaHeight, metaWidth);

		if (dynamicHeight) {
			height = settings.dynamicInputHeight > 0 ? settings.dynamicInputHeight
				: (metaHeight > 0 ? metaHeight : 640);
		}
		if (dynamicWidth) {
			width = settings.dynamicInputWidth > 0 ? settings.dynamicInputWidth
				: (metaWidth > 0 ? metaWidth : 640);
		}

		ofLogVerbose("ofxYolo26::Model")
			<< ofFilePath::getFileName(modelPath) << " has a dynamic input shape; running at "
			<< width << "x" << height;
	}

	if (height <= 0 || width <= 0) {
		ofLogError("ofxYolo26::Model") << "could not resolve a usable input size";
		return false;
	}

	// Feed the concrete shape back into BaseHandler so run() builds the right
	// tensor and the preprocessor has a buffer of the right size.
	input_node_dims = { 1, 3, int64_t(height), int64_t(width) };
	input_tensor_size = size_t(3) * size_t(height) * size_t(width);
	input_values_handler.resize(input_tensor_size);
	return true;
}

void Model::parseImgsz(const std::string & imgsz, int & height, int & width) {
	// Ultralytics writes "[640, 640]" as [height, width]; a bare "640" appears
	// on some exports and means square.
	height = 0;
	width = 0;

	std::vector<int> values;
	std::string digits;
	for (char c : imgsz) {
		if (std::isdigit((unsigned char)c)) {
			digits += c;
		} else if (!digits.empty()) {
			values.push_back(ofToInt(digits));
			digits.clear();
		}
	}
	if (!digits.empty()) values.push_back(ofToInt(digits));

	if (values.size() >= 2) {
		height = values[0];
		width = values[1];
	} else if (values.size() == 1) {
		height = width = values[0];
	}
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
