#include "ofxYolo26Detection.h"

namespace ofxYolo26 {

float intersectionOverUnion(const ofRectangle & a, const ofRectangle & b) {
	const float x1 = std::max(a.getMinX(), b.getMinX());
	const float y1 = std::max(a.getMinY(), b.getMinY());
	const float x2 = std::min(a.getMaxX(), b.getMaxX());
	const float y2 = std::min(a.getMaxY(), b.getMaxY());

	const float w = x2 - x1;
	const float h = y2 - y1;
	if (w <= 0.0f || h <= 0.0f) return 0.0f;

	const float inter = w * h;
	const float uni = a.getWidth() * a.getHeight() + b.getWidth() * b.getHeight() - inter;
	return uni > 0.0f ? inter / uni : 0.0f;
}

std::vector<int> nmsPerClassIndices(const std::vector<Detection> & dets, float iouThreshold, int topk) {
	// Bucket by class, keep the highest-scoring box in each overlapping group,
	// then merge the per-class survivors back into one score-ordered list.
	std::map<int, std::vector<int>> byClass;
	for (int i = 0; i < int(dets.size()); i++) {
		byClass[dets[i].label].push_back(i);
	}

	std::vector<int> keepAll;
	for (auto & entry : byClass) {
		std::vector<int> & candidates = entry.second;
		std::stable_sort(candidates.begin(), candidates.end(), [&](int a, int b) {
			return dets[a].score > dets[b].score;
		});

		std::vector<int> keep;
		for (int candidate : candidates) {
			bool ok = true;
			for (int kept : keep) {
				if (intersectionOverUnion(dets[candidate].box, dets[kept].box) > iouThreshold) {
					ok = false;
					break;
				}
			}
			if (!ok) continue;
			keep.push_back(candidate);
			if (topk > 0 && int(keep.size()) >= topk) break;
		}
		keepAll.insert(keepAll.end(), keep.begin(), keep.end());
	}

	std::stable_sort(keepAll.begin(), keepAll.end(), [&](int a, int b) {
		return dets[a].score > dets[b].score;
	});
	if (topk > 0 && int(keepAll.size()) > topk) keepAll.resize(topk);
	return keepAll;
}

std::vector<Detection> nmsPerClass(const std::vector<Detection> & dets, float iouThreshold, int topk) {
	const std::vector<int> keep = nmsPerClassIndices(dets, iouThreshold, topk);
	std::vector<Detection> out;
	out.reserve(keep.size());
	for (int i : keep) out.push_back(dets[i]);
	return out;
}

ofColor classColor(int label) {
	// Van der Corput (bit-reversed) hue placement. Each new id lands in the
	// largest remaining gap, so any prefix of the sequence is spread as widely
	// as it can be — 0 and 5 come out red and blue rather than two pinks, which
	// is what golden-angle stepping gives you at these indices.
	uint32_t bits = uint32_t(std::max(label, 0));
	bits = (bits << 16) | (bits >> 16);
	bits = ((bits & 0x55555555u) << 1) | ((bits & 0xAAAAAAAAu) >> 1);
	bits = ((bits & 0x33333333u) << 2) | ((bits & 0xCCCCCCCCu) >> 2);
	bits = ((bits & 0x0F0F0F0Fu) << 4) | ((bits & 0xF0F0F0F0u) >> 4);
	bits = ((bits & 0x00FF00FFu) << 8) | ((bits & 0xFF00FF00u) >> 8);
	const float hue = float(bits) * 2.3283064365386963e-10f * 255.0f;

	// Alternate brightness so ids that do eventually share a hue still differ.
	const float brightness = (label / 64) % 2 ? 190.0f : 255.0f;
	return ofColor::fromHsb(hue, 210.0f, brightness);
}

} // namespace ofxYolo26
