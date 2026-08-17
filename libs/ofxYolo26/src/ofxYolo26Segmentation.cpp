#include "ofxYolo26Segmentation.h"

namespace ofxYolo26 {

//--------------------------------------------------------------
// InstanceMask
//--------------------------------------------------------------

float InstanceMask::getAlpha(int protoX, int protoY) const {
	if (!alpha.isAllocated()) return 0.0f;
	const int x = protoX - int(rect.x);
	const int y = protoY - int(rect.y);
	if (x < 0 || y < 0 || x >= int(alpha.getWidth()) || y >= int(alpha.getHeight())) return 0.0f;
	return alpha[size_t(y) * size_t(alpha.getWidth()) + size_t(x)];
}

float InstanceMask::getAlphaInterpolated(float protoX, float protoY) const {
	if (!alpha.isAllocated()) return 0.0f;

	// Sample at pixel centres; getAlpha() returns 0 past the edges, so the mask
	// fades out rather than clamping its border value outwards.
	const float fx = protoX - 0.5f;
	const float fy = protoY - 0.5f;
	const int x0 = int(std::floor(fx));
	const int y0 = int(std::floor(fy));
	const float tx = fx - x0;
	const float ty = fy - y0;

	const float a = ofLerp(getAlpha(x0, y0), getAlpha(x0 + 1, y0), tx);
	const float b = ofLerp(getAlpha(x0, y0 + 1), getAlpha(x0 + 1, y0 + 1), tx);
	return ofLerp(a, b, ty);
}

//--------------------------------------------------------------
// SegmentationResult
//--------------------------------------------------------------

void SegmentationResult::clear() {
	detections.clear();
	masks.clear();
	letterbox = Letterbox();
	protoWidth = 0;
	protoHeight = 0;
	protoScaleX = 1.0f;
	protoScaleY = 1.0f;
}

ofRectangle SegmentationResult::getBoxInSource(size_t index) const {
	if (index >= detections.size()) return ofRectangle();

	const ofRectangle & box = detections[index].box;
	const glm::vec2 topLeft = letterbox.destToSource(glm::vec2(box.getMinX(), box.getMinY()));
	const glm::vec2 bottomRight = letterbox.destToSource(glm::vec2(box.getMaxX(), box.getMaxY()));
	return ofRectangle(topLeft.x, topLeft.y, bottomRight.x - topLeft.x, bottomRight.y - topLeft.y);
}

glm::vec2 SegmentationResult::sourceToProto(const glm::vec2 & src) const {
	const glm::vec2 dest = letterbox.sourceToDest(src);
	return glm::vec2(dest.x / protoScaleX, dest.y / protoScaleY);
}

glm::vec2 SegmentationResult::protoToSource(const glm::vec2 & proto) const {
	return letterbox.destToSource(glm::vec2(proto.x * protoScaleX, proto.y * protoScaleY));
}

ofRectangle SegmentationResult::getContentRectInProto() const {
	if (protoScaleX == 0.0f || protoScaleY == 0.0f) return ofRectangle();
	return ofRectangle(letterbox.padX / protoScaleX,
		letterbox.padY / protoScaleY,
		letterbox.contentWidth / protoScaleX,
		letterbox.contentHeight / protoScaleY);
}

float SegmentationResult::getAlphaAtSource(size_t index, float srcX, float srcY) const {
	if (index >= masks.size()) return 0.0f;
	const glm::vec2 proto = sourceToProto(glm::vec2(srcX, srcY));
	return masks[index].getAlphaInterpolated(proto.x, proto.y);
}

float SegmentationResult::getAlphaAtSource(float srcX, float srcY, int * instanceOut) const {
	const glm::vec2 proto = sourceToProto(glm::vec2(srcX, srcY));

	float best = 0.0f;
	int bestIndex = -1;
	for (size_t i = 0; i < masks.size(); i++) {
		const float a = masks[i].getAlphaInterpolated(proto.x, proto.y);
		if (a > best) {
			best = a;
			bestIndex = int(i);
		}
	}

	if (instanceOut != nullptr) *instanceOut = bestIndex;
	return best;
}

//--------------------------------------------------------------
// Segmentation
//--------------------------------------------------------------

bool Segmentation::setup(const std::string & modelPath, const Settings & settings) {
	result.clear();
	return load(modelPath, settings);
}

void Segmentation::setClassFilter(const std::vector<int> & labels) {
	classFilter = labels;
	std::sort(classFilter.begin(), classFilter.end());
}

bool Segmentation::findOutputIndices(int & detIndex, int & protoIndex) const {
	detIndex = -1;
	protoIndex = -1;
	for (int i = 0; i < num_outputs; i++) {
		const std::vector<int64_t> shape = getOutputShape(i);
		if (shape.size() == 4 && protoIndex < 0) {
			protoIndex = i;
		} else if (shape.size() == 3 && detIndex < 0) {
			detIndex = i;
		}
	}
	return detIndex >= 0 && protoIndex >= 0;
}

bool Segmentation::onLoaded() {
	int detIndex = -1;
	int protoIndex = -1;
	if (!findOutputIndices(detIndex, protoIndex)) {
		ofLogError("ofxYolo26::Segmentation")
			<< "expected a 3D detection output and a 4D prototype output";
		return false;
	}

	const std::vector<int64_t> detShape = getOutputShape(detIndex);
	const std::vector<int64_t> protoShape = getOutputShape(protoIndex);

	proposalStride = int(detShape[2]);
	numPrototypes = int(protoShape[1]);

	// 4 box + 1 score + 1 class, then one coefficient per prototype.
	if (proposalStride != 6 + numPrototypes) {
		ofLogError("ofxYolo26::Segmentation")
			<< "detection stride " << proposalStride << " does not match 6 + "
			<< numPrototypes << " prototypes";
		return false;
	}

	const std::string task = getMetadata("task");
	if (!task.empty() && task != "segment") {
		ofLogWarning("ofxYolo26::Segmentation")
			<< "model reports task '" << task << "' rather than 'segment'";
	}
	return true;
}

const SegmentationResult & Segmentation::update(const ofPixels & pixels) {
	if (!isLoaded()) return result;

	preprocess(pixels);

	result.detections.clear();
	result.masks.clear();
	result.letterbox = letterbox;

	if (!runInference()) return result;

	int detIndex = -1;
	int protoIndex = -1;
	if (!findOutputIndices(detIndex, protoIndex)) return result;

	const Ort::Value & detValue = output_values[detIndex];
	const Ort::Value & protoValue = output_values[protoIndex];
	if (!detValue || !protoValue || !detValue.IsTensor() || !protoValue.IsTensor()) {
		ofLogError("ofxYolo26::Segmentation") << "segmentation outputs are missing";
		return result;
	}

	const std::vector<int64_t> detShape = detValue.GetTensorTypeAndShapeInfo().GetShape();
	const std::vector<int64_t> protoShape = protoValue.GetTensorTypeAndShapeInfo().GetShape();
	if (detShape.size() != 3 || protoShape.size() != 4) return result;

	const int proposals = int(detShape[1]);
	const int stride = int(detShape[2]);
	const int maskC = int(protoShape[1]);
	const int protoH = int(protoShape[2]);
	const int protoW = int(protoShape[3]);
	if (stride != 6 + maskC || protoW <= 0 || protoH <= 0) {
		ofLogError("ofxYolo26::Segmentation") << "unexpected output shapes at runtime";
		return result;
	}

	const float * detData = getOutputData(detIndex);
	const float * protoData = getOutputData(protoIndex);
	if (detData == nullptr || protoData == nullptr) return result;

	result.protoWidth = protoW;
	result.protoHeight = protoH;
	result.protoScaleX = float(inputWidth) / float(protoW);
	result.protoScaleY = float(inputHeight) / float(protoH);

	// 1. Threshold, and remember where each survivor's coefficients live.
	std::vector<Detection> candidates;
	std::vector<int> candidateProposal;
	for (int i = 0; i < proposals; i++) {
		const float * p = detData + size_t(i) * size_t(stride);
		const float score = p[4];
		if (score < scoreThreshold) continue;

		const int label = int(p[5]);
		if (!classFilter.empty()
			&& !std::binary_search(classFilter.begin(), classFilter.end(), label)) {
			continue;
		}

		Detection det;
		det.box = ofRectangle(p[0], p[1], p[2] - p[0], p[3] - p[1]);
		det.score = score;
		det.label = label;
		det.labelName = getClassName(label);
		candidates.push_back(det);
		candidateProposal.push_back(i);
	}
	if (candidates.empty()) return result;

	// 2. Drop near-duplicates that would otherwise fight over the same mask
	//    pixels, as the reference does before compositing.
	std::vector<int> keep = (iouThreshold >= 1.0f)
		? [&] {
			  std::vector<int> all(candidates.size());
			  std::iota(all.begin(), all.end(), 0);
			  std::stable_sort(all.begin(), all.end(), [&](int a, int b) {
				  return candidates[a].score > candidates[b].score;
			  });
			  return all;
		  }()
		: nmsPerClassIndices(candidates, iouThreshold, maxInstances);

	if (maxInstances > 0 && int(keep.size()) > maxInstances) keep.resize(maxInstances);

	// 3. Build each surviving instance's mask inside its box.
	const size_t protoPlane = size_t(protoW) * size_t(protoH);
	const float noiseScale = (maskNoiseFloor < 1.0f) ? 1.0f / (1.0f - maskNoiseFloor) : 0.0f;

	result.detections.reserve(keep.size());
	result.masks.reserve(keep.size());

	for (int index : keep) {
		const Detection & det = candidates[index];
		const float * coeffs = detData + size_t(candidateProposal[index]) * size_t(stride) + 6;

		const int mx1 = ofClamp(int(std::floor(det.box.getMinX() / result.protoScaleX)), 0, protoW);
		const int my1 = ofClamp(int(std::floor(det.box.getMinY() / result.protoScaleY)), 0, protoH);
		const int mx2 = ofClamp(int(std::ceil(det.box.getMaxX() / result.protoScaleX)), mx1, protoW);
		const int my2 = ofClamp(int(std::ceil(det.box.getMaxY() / result.protoScaleY)), my1, protoH);

		InstanceMask mask;
		mask.rect = ofRectangle(float(mx1), float(my1), float(mx2 - mx1), float(my2 - my1));
		if (mx2 <= mx1 || my2 <= my1) {
			result.detections.push_back(det);
			result.masks.push_back(std::move(mask));
			continue;
		}

		const int mw = mx2 - mx1;
		mask.alpha.allocate(mw, my2 - my1, OF_PIXELS_GRAY);
		float * dst = mask.alpha.getData();

		for (int y = my1; y < my2; y++) {
			for (int x = mx1; x < mx2; x++) {
				const size_t i = size_t(y) * size_t(protoW) + size_t(x);

				float sum = 0.0f;
				for (int c = 0; c < maskC; c++) {
					sum += coeffs[c] * protoData[size_t(c) * protoPlane + i];
				}

				float a = 1.0f / (1.0f + std::exp(-sum));
				if (maskNoiseFloor > 0.0f) {
					a = std::max(0.0f, (a - maskNoiseFloor) * noiseScale);
				}
				dst[size_t(y - my1) * size_t(mw) + size_t(x - mx1)] = a;
			}
		}

		result.detections.push_back(det);
		result.masks.push_back(std::move(mask));
	}

	return result;
}

//--------------------------------------------------------------
// Display helpers
//--------------------------------------------------------------

void toAlphaPixels(const SegmentationResult & result, ofFloatPixels & out) {
	if (result.protoWidth <= 0 || result.protoHeight <= 0) {
		out.clear();
		return;
	}

	out.allocate(result.protoWidth, result.protoHeight, OF_PIXELS_GRAY);
	out.set(0);
	float * dst = out.getData();

	for (const InstanceMask & mask : result.masks) {
		if (!mask.alpha.isAllocated()) continue;
		const int x0 = int(mask.rect.x);
		const int y0 = int(mask.rect.y);
		const int w = int(mask.alpha.getWidth());
		const int h = int(mask.alpha.getHeight());
		const float * src = mask.alpha.getData();

		for (int y = 0; y < h; y++) {
			float * dstRow = dst + size_t(y0 + y) * size_t(result.protoWidth) + size_t(x0);
			const float * srcRow = src + size_t(y) * size_t(w);
			for (int x = 0; x < w; x++) {
				dstRow[x] = std::max(dstRow[x], srcRow[x]);
			}
		}
	}
}

void toAlphaPixels(const SegmentationResult & result, ofFloatPixels & out, size_t index) {
	if (result.protoWidth <= 0 || result.protoHeight <= 0 || index >= result.masks.size()) {
		out.clear();
		return;
	}

	out.allocate(result.protoWidth, result.protoHeight, OF_PIXELS_GRAY);
	out.set(0);

	const InstanceMask & mask = result.masks[index];
	if (!mask.alpha.isAllocated()) return;

	const int x0 = int(mask.rect.x);
	const int y0 = int(mask.rect.y);
	const int w = int(mask.alpha.getWidth());
	const int h = int(mask.alpha.getHeight());
	const float * src = mask.alpha.getData();

	for (int y = 0; y < h; y++) {
		std::copy(src + size_t(y) * size_t(w),
			src + size_t(y + 1) * size_t(w),
			out.getData() + size_t(y0 + y) * size_t(result.protoWidth) + size_t(x0));
	}
}

void toColorPixels(const SegmentationResult & result, ofPixels & out, bool colorByInstance) {
	if (result.protoWidth <= 0 || result.protoHeight <= 0) {
		out.clear();
		return;
	}

	out.allocate(result.protoWidth, result.protoHeight, OF_PIXELS_RGBA);
	out.set(0);
	unsigned char * dst = out.getData();

	// Painter's ordering: biggest boxes first so small foreground objects, which
	// are drawn last, stay on top where masks overlap.
	std::vector<int> order(result.masks.size());
	std::iota(order.begin(), order.end(), 0);
	std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
		const ofRectangle & ra = result.detections[a].box;
		const ofRectangle & rb = result.detections[b].box;
		return ra.getWidth() * ra.getHeight() > rb.getWidth() * rb.getHeight();
	});

	for (int index : order) {
		const InstanceMask & mask = result.masks[index];
		if (!mask.alpha.isAllocated()) continue;

		const ofColor color = classColor(colorByInstance ? index : result.detections[index].label);
		const int x0 = int(mask.rect.x);
		const int y0 = int(mask.rect.y);
		const int w = int(mask.alpha.getWidth());
		const int h = int(mask.alpha.getHeight());
		const float * src = mask.alpha.getData();

		for (int y = 0; y < h; y++) {
			const float * srcRow = src + size_t(y) * size_t(w);
			for (int x = 0; x < w; x++) {
				const float a = srcRow[x];
				const size_t o = (size_t(y0 + y) * size_t(result.protoWidth) + size_t(x0 + x)) * 4;

				// Highest alpha wins the pixel, matching the reference's rule.
				if (a * 255.0f <= float(dst[o + 3])) continue;
				dst[o + 0] = color.r;
				dst[o + 1] = color.g;
				dst[o + 2] = color.b;
				dst[o + 3] = (unsigned char)ofClamp(a * 255.0f, 0.0f, 255.0f);
			}
		}
	}
}

} // namespace ofxYolo26
