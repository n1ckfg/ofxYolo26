#include "ofxYolo26Depth.h"

namespace ofxYolo26 {

//--------------------------------------------------------------
// DepthMap
//--------------------------------------------------------------

void DepthMap::clear() {
	width = 0;
	height = 0;
	values.clear();
	letterbox = Letterbox();
	minValue = 0.0f;
	maxValue = 0.0f;
}

float DepthMap::getValue(int x, int y) const {
	if (!isAllocated()) return 0.0f;
	x = ofClamp(x, 0, width - 1);
	y = ofClamp(y, 0, height - 1);
	return values[size_t(y) * size_t(width) + size_t(x)];
}

float DepthMap::getValueAtSource(float srcX, float srcY, float outside) const {
	if (!isAllocated()) return outside;
	if (srcX < 0.0f || srcY < 0.0f || srcX >= letterbox.srcWidth || srcY >= letterbox.srcHeight) {
		return outside;
	}

	const glm::vec2 d = letterbox.sourceToDest(glm::vec2(srcX, srcY));

	// Stay inside the content region so interpolation never pulls in padding.
	const float x0f = ofClamp(d.x, float(letterbox.padX), float(letterbox.padX + letterbox.contentWidth) - 1.0f);
	const float y0f = ofClamp(d.y, float(letterbox.padY), float(letterbox.padY + letterbox.contentHeight) - 1.0f);

	const int x0 = int(std::floor(x0f));
	const int y0 = int(std::floor(y0f));
	const int x1 = std::min(x0 + 1, letterbox.padX + letterbox.contentWidth - 1);
	const int y1 = std::min(y0 + 1, letterbox.padY + letterbox.contentHeight - 1);
	const float tx = x0f - x0;
	const float ty = y0f - y0;

	const float v00 = getValue(x0, y0);
	const float v10 = getValue(x1, y0);
	const float v01 = getValue(x0, y1);
	const float v11 = getValue(x1, y1);

	return glm::mix(glm::mix(v00, v10, tx), glm::mix(v01, v11, tx), ty);
}

float DepthMap::getValueAtSourceNormalized(float u, float v, float outside) const {
	return getValueAtSource(u * letterbox.srcWidth, v * letterbox.srcHeight, outside);
}

void DepthMap::updateRange() {
	minValue = 0.0f;
	maxValue = 0.0f;
	if (!isAllocated()) return;

	// Content only: the letterbox bars are hallucinated depth for black input.
	const int x0 = ofClamp(letterbox.padX, 0, width - 1);
	const int y0 = ofClamp(letterbox.padY, 0, height - 1);
	const int x1 = ofClamp(letterbox.padX + std::max(letterbox.contentWidth, 1), x0 + 1, width);
	const int y1 = ofClamp(letterbox.padY + std::max(letterbox.contentHeight, 1), y0 + 1, height);

	float lo = std::numeric_limits<float>::max();
	float hi = std::numeric_limits<float>::lowest();
	for (int y = y0; y < y1; y++) {
		const float * row = values.data() + size_t(y) * size_t(width);
		for (int x = x0; x < x1; x++) {
			const float v = row[x];
			if (!std::isfinite(v)) continue;
			lo = std::min(lo, v);
			hi = std::max(hi, v);
		}
	}

	if (lo <= hi) {
		minValue = lo;
		maxValue = hi;
	}
}

//--------------------------------------------------------------
// Depth
//--------------------------------------------------------------

bool Depth::setup(const std::string & modelPath, const Settings & settings) {
	depthMap.clear();
	return load(modelPath, settings);
}

bool Depth::onLoaded() {
	if (num_outputs < 1) {
		ofLogError("ofxYolo26::Depth") << "model has no outputs";
		return false;
	}

	const std::vector<int64_t> shape = getOutputShape(0);

	// Exported YOLO26 depth models return [1, 1, H, W]. Accept [1, H, W] too,
	// matching the defensive fallback in the reference web app.
	if (shape.size() == 4 && shape[0] == 1 && shape[1] == 1) {
		outputHeight = int(shape[2]);
		outputWidth = int(shape[3]);
	} else if (shape.size() == 3 && shape[0] == 1) {
		outputHeight = int(shape[1]);
		outputWidth = int(shape[2]);
	} else {
		ofLogError("ofxYolo26::Depth") << "unsupported depth output rank " << shape.size()
									   << "; expected [1, 1, H, W]";
		return false;
	}

	const std::string task = getMetadata("task");
	if (!task.empty() && task != "depth") {
		ofLogWarning("ofxYolo26::Depth") << "model reports task '" << task << "' rather than 'depth'";
	}
	return true;
}

const DepthMap & Depth::update(const ofPixels & pixels) {
	if (!isLoaded()) return depthMap;

	preprocess(pixels);
	if (!runInference()) return depthMap;

	// The shape reported after Run is authoritative; declared dims may be dynamic.
	const Ort::Value & value = output_values[0];
	if (!value || !value.IsTensor()) {
		ofLogError("ofxYolo26::Depth") << "depth output is missing";
		return depthMap;
	}

	const std::vector<int64_t> shape = value.GetTensorTypeAndShapeInfo().GetShape();
	int w = 0;
	int h = 0;
	if (shape.size() == 4) {
		h = int(shape[2]);
		w = int(shape[3]);
	} else if (shape.size() == 3) {
		h = int(shape[1]);
		w = int(shape[2]);
	}

	const size_t count = value.GetTensorTypeAndShapeInfo().GetElementCount();
	if (w <= 0 || h <= 0 || count != size_t(w) * size_t(h)) {
		ofLogError("ofxYolo26::Depth") << "depth output length " << count
									   << " does not match " << w << "x" << h;
		return depthMap;
	}

	const float * data = getOutputData(0);
	if (data == nullptr) return depthMap;

	outputWidth = w;
	outputHeight = h;

	depthMap.width = w;
	depthMap.height = h;
	depthMap.values.assign(data, data + count);
	depthMap.letterbox = letterbox;

	// Depth is produced at the model's input resolution; if an export ever
	// returns a smaller map, rescale the mapping to match.
	if (w != inputWidth || h != inputHeight) {
		const float sx = float(w) / float(inputWidth);
		const float sy = float(h) / float(inputHeight);
		depthMap.letterbox.dstWidth = w;
		depthMap.letterbox.dstHeight = h;
		depthMap.letterbox.padX = int(std::round(letterbox.padX * sx));
		depthMap.letterbox.padY = int(std::round(letterbox.padY * sy));
		depthMap.letterbox.contentWidth = std::max(1, int(std::round(letterbox.contentWidth * sx)));
		depthMap.letterbox.contentHeight = std::max(1, int(std::round(letterbox.contentHeight * sy)));
		depthMap.letterbox.scaleX = float(depthMap.letterbox.contentWidth) / float(std::max(1, letterbox.srcWidth));
		depthMap.letterbox.scaleY = float(depthMap.letterbox.contentHeight) / float(std::max(1, letterbox.srcHeight));
	}

	depthMap.updateRange();
	return depthMap;
}

//--------------------------------------------------------------
// Display helpers
//--------------------------------------------------------------

namespace {

	/// Region of the map to export, and the pixel size that region produces.
	struct CropRegion {
		int x = 0;
		int y = 0;
		int w = 0;
		int h = 0;
	};

	CropRegion regionFor(const DepthMap & map, bool cropToContent) {
		CropRegion r;
		if (!cropToContent || map.letterbox.contentWidth <= 0 || map.letterbox.contentHeight <= 0) {
			r.w = map.width;
			r.h = map.height;
			return r;
		}
		r.x = ofClamp(map.letterbox.padX, 0, map.width - 1);
		r.y = ofClamp(map.letterbox.padY, 0, map.height - 1);
		r.w = ofClamp(map.letterbox.contentWidth, 1, map.width - r.x);
		r.h = ofClamp(map.letterbox.contentHeight, 1, map.height - r.y);
		return r;
	}

	/// Resolves an explicit range, falling back to the map's own content range.
	void resolveRange(const DepthMap & map, float & lo, float & hi) {
		if (hi <= lo) {
			lo = map.minValue;
			hi = map.maxValue;
		}
		if (hi <= lo) hi = lo + 1.0f;
	}

} // namespace

void toFloatPixels(const DepthMap & map, ofFloatPixels & out, bool cropToContent) {
	if (!map.isAllocated()) {
		out.clear();
		return;
	}

	const CropRegion r = regionFor(map, cropToContent);
	out.allocate(r.w, r.h, OF_PIXELS_GRAY);

	for (int y = 0; y < r.h; y++) {
		const float * src = map.values.data() + size_t(y + r.y) * size_t(map.width) + size_t(r.x);
		std::copy(src, src + r.w, out.getData() + size_t(y) * size_t(r.w));
	}
}

void toNormalizedPixels(const DepthMap & map,
	ofFloatPixels & out,
	float rangeMin,
	float rangeMax,
	bool invert,
	bool cropToContent) {

	if (!map.isAllocated()) {
		out.clear();
		return;
	}

	float lo = rangeMin;
	float hi = rangeMax;
	resolveRange(map, lo, hi);
	const float invSpan = 1.0f / (hi - lo);

	const CropRegion r = regionFor(map, cropToContent);
	out.allocate(r.w, r.h, OF_PIXELS_GRAY);
	float * dst = out.getData();

	for (int y = 0; y < r.h; y++) {
		const float * src = map.values.data() + size_t(y + r.y) * size_t(map.width) + size_t(r.x);
		for (int x = 0; x < r.w; x++) {
			float t = ofClamp((src[x] - lo) * invSpan, 0.0f, 1.0f);
			if (invert) t = 1.0f - t;
			dst[size_t(y) * size_t(r.w) + size_t(x)] = t;
		}
	}
}

ofFloatColor depthRamp(float t) {
	// Dark blue -> teal -> green -> amber -> white. Enough separation to read
	// depth banding by eye without a legend.
	static const ofFloatColor stops[] = {
		ofFloatColor(0.05f, 0.03f, 0.20f),
		ofFloatColor(0.10f, 0.42f, 0.62f),
		ofFloatColor(0.16f, 0.72f, 0.55f),
		ofFloatColor(0.85f, 0.72f, 0.20f),
		ofFloatColor(1.00f, 1.00f, 0.96f)
	};
	static const int numStops = sizeof(stops) / sizeof(stops[0]);

	t = ofClamp(t, 0.0f, 1.0f) * (numStops - 1);
	const int i = std::min(int(t), numStops - 2);
	return stops[i].getLerped(stops[i + 1], t - i);
}

void toColorPixels(const DepthMap & map,
	ofPixels & out,
	float rangeMin,
	float rangeMax,
	bool invert,
	bool cropToContent) {

	if (!map.isAllocated()) {
		out.clear();
		return;
	}

	float lo = rangeMin;
	float hi = rangeMax;
	resolveRange(map, lo, hi);
	const float invSpan = 1.0f / (hi - lo);

	const CropRegion r = regionFor(map, cropToContent);
	out.allocate(r.w, r.h, OF_PIXELS_RGB);
	unsigned char * dst = out.getData();

	for (int y = 0; y < r.h; y++) {
		const float * src = map.values.data() + size_t(y + r.y) * size_t(map.width) + size_t(r.x);
		for (int x = 0; x < r.w; x++) {
			float t = ofClamp((src[x] - lo) * invSpan, 0.0f, 1.0f);
			if (invert) t = 1.0f - t;
			const ofFloatColor c = depthRamp(t);
			const size_t o = (size_t(y) * size_t(r.w) + size_t(x)) * 3;
			dst[o + 0] = (unsigned char)(c.r * 255.0f);
			dst[o + 1] = (unsigned char)(c.g * 255.0f);
			dst[o + 2] = (unsigned char)(c.b * 255.0f);
		}
	}
}

} // namespace ofxYolo26
