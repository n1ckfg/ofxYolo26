#include "ofxYolo26Types.h"

namespace ofxYolo26 {

Letterbox Letterbox::make(int srcW, int srcH, int dstW, int dstH, ResizeMode mode) {
	Letterbox lb;
	lb.srcWidth = srcW;
	lb.srcHeight = srcH;
	lb.dstWidth = dstW;
	lb.dstHeight = dstH;

	if (srcW <= 0 || srcH <= 0 || dstW <= 0 || dstH <= 0) {
		return lb;
	}

	if (mode == ResizeMode::Stretch) {
		lb.contentWidth = dstW;
		lb.contentHeight = dstH;
		lb.padX = 0;
		lb.padY = 0;
	} else {
		const float scale = std::min(float(dstW) / float(srcW), float(dstH) / float(srcH));
		lb.contentWidth = std::max(1, int(std::floor(srcW * scale)));
		lb.contentHeight = std::max(1, int(std::floor(srcH * scale)));
		lb.padX = (dstW - lb.contentWidth) / 2;
		lb.padY = (dstH - lb.contentHeight) / 2;
	}

	lb.scaleX = float(lb.contentWidth) / float(srcW);
	lb.scaleY = float(lb.contentHeight) / float(srcH);
	return lb;
}

glm::vec2 Letterbox::sourceToDest(const glm::vec2 & src) const {
	return glm::vec2(src.x * scaleX + padX, src.y * scaleY + padY);
}

glm::vec2 Letterbox::destToSource(const glm::vec2 & dst) const {
	if (scaleX == 0.0f || scaleY == 0.0f) return glm::vec2(0.0f);
	return glm::vec2((dst.x - padX) / scaleX, (dst.y - padY) / scaleY);
}

bool Letterbox::destContains(float dstX, float dstY) const {
	return dstX >= padX && dstX < padX + contentWidth
		&& dstY >= padY && dstY < padY + contentHeight;
}

ofRectangle Letterbox::getContentRect() const {
	return ofRectangle(float(padX), float(padY), float(contentWidth), float(contentHeight));
}

} // namespace ofxYolo26
