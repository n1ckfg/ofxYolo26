#include "ofxYolo26Preprocessor.h"

namespace ofxYolo26 {

namespace {

	/// True when the pixel format stores blue before red, so the two need swapping
	/// on the way into an RGB tensor.
	bool isBlueFirst(ofPixelFormat format) {
		return format == OF_PIXELS_BGR || format == OF_PIXELS_BGRA;
	}

} // namespace

Preprocessor::AxisTaps Preprocessor::buildTaps(int srcSize, int dstSize) {
	AxisTaps taps;
	if (srcSize <= 0 || dstSize <= 0) return taps;

	const double ratio = double(srcSize) / double(dstSize);
	const bool shrinking = dstSize < srcSize;

	taps.maxCount = shrinking ? std::max(1, int(std::ceil(ratio)) + 1) : 2;
	taps.start.resize(dstSize);
	taps.count.resize(dstSize);
	taps.weights.assign(size_t(dstSize) * taps.maxCount, 0.0f);

	for (int d = 0; d < dstSize; d++) {
		float * w = &taps.weights[size_t(d) * taps.maxCount];

		if (shrinking) {
			// Box filter: average every source sample the destination pixel covers.
			int s0 = int(std::floor(d * ratio));
			int s1 = int(std::ceil((d + 1) * ratio));
			s0 = ofClamp(s0, 0, srcSize - 1);
			s1 = ofClamp(s1, s0 + 1, srcSize);

			const int n = std::min(s1 - s0, taps.maxCount);
			taps.start[d] = s0;
			taps.count[d] = n;
			const float weight = 1.0f / float(n);
			for (int i = 0; i < n; i++) w[i] = weight;
		} else {
			// Bilinear, sampling at destination pixel centres.
			double f = (d + 0.5) * ratio - 0.5;
			f = ofClamp(f, 0.0, double(srcSize - 1));
			const int s0 = std::min(int(std::floor(f)), srcSize - 1);
			const int s1 = std::min(s0 + 1, srcSize - 1);
			const float t = float(f - s0);

			taps.start[d] = s0;
			taps.count[d] = (s1 == s0) ? 1 : 2;
			w[0] = 1.0f - t;
			if (taps.count[d] == 2) w[1] = t;
		}
	}

	return taps;
}

Letterbox Preprocessor::toTensorData(const ofPixels & src,
	float * dst,
	int dstWidth,
	int dstHeight,
	ResizeMode mode,
	float padValue) {

	Letterbox lb = Letterbox::make(src.getWidth(), src.getHeight(), dstWidth, dstHeight, mode);

	if (dst == nullptr || dstWidth <= 0 || dstHeight <= 0) return lb;

	const size_t plane = size_t(dstWidth) * size_t(dstHeight);
	std::fill(dst, dst + plane * 3, padValue);

	if (!src.isAllocated() || lb.contentWidth <= 0 || lb.contentHeight <= 0) {
		ofLogWarning("ofxYolo26::Preprocessor") << "source pixels are empty; tensor left as padding";
		return lb;
	}

	const int srcW = src.getWidth();
	const int srcH = src.getHeight();
	const int nch = src.getNumChannels();
	const unsigned char * srcData = src.getData();
	const size_t srcStride = size_t(srcW) * size_t(nch);
	const bool swapRB = isBlueFirst(src.getPixelFormat());
	const bool gray = nch < 3;

	const AxisTaps tx = buildTaps(srcW, lb.contentWidth);
	const AxisTaps ty = buildTaps(srcH, lb.contentHeight);

	const float inv255 = 1.0f / 255.0f;
	float * dstR = dst;
	float * dstG = dst + plane;
	float * dstB = dst + plane * 2;

	for (int dy = 0; dy < lb.contentHeight; dy++) {
		const int sy0 = ty.start[dy];
		const int nY = ty.count[dy];
		const float * wY = &ty.weights[size_t(dy) * ty.maxCount];
		const size_t rowOut = size_t(dy + lb.padY) * size_t(dstWidth) + size_t(lb.padX);

		for (int dx = 0; dx < lb.contentWidth; dx++) {
			const int sx0 = tx.start[dx];
			const int nX = tx.count[dx];
			const float * wX = &tx.weights[size_t(dx) * tx.maxCount];

			float r = 0.0f, g = 0.0f, b = 0.0f;

			for (int j = 0; j < nY; j++) {
				const int sy = std::min(sy0 + j, srcH - 1);
				const float wy = wY[j];
				const unsigned char * row = srcData + size_t(sy) * srcStride;

				for (int i = 0; i < nX; i++) {
					const int sx = std::min(sx0 + i, srcW - 1);
					const float w = wy * wX[i];
					const unsigned char * p = row + size_t(sx) * size_t(nch);

					if (gray) {
						const float v = float(p[0]) * w;
						r += v;
						g += v;
						b += v;
					} else {
						r += float(p[0]) * w;
						g += float(p[1]) * w;
						b += float(p[2]) * w;
					}
				}
			}

			if (swapRB) std::swap(r, b);

			const size_t o = rowOut + size_t(dx);
			dstR[o] = r * inv255;
			dstG[o] = g * inv255;
			dstB[o] = b * inv255;
		}
	}

	return lb;
}

} // namespace ofxYolo26
