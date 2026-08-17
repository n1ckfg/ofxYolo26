#pragma once

#include "ofxYolo26Depth.h"
#include "ofxYolo26ModelThread.h"

namespace ofxYolo26 {

/// Depth estimation on a background thread. See ModelThread for the mechanics.
class DepthThread : public ModelThread<Depth, DepthMap> {
public:
	const DepthMap & getDepthMap() const { return getResult(); }
	const Depth & getDepth() const { return getModel(); }
};

} // namespace ofxYolo26
