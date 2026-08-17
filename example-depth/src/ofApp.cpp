#include "ofApp.h"

namespace {
const std::string MODEL_PATH = "models/yolo26n-depth.onnx";
const std::string STILL_PATH = "test.jpg";
const int PANE_GAP = 12;
const int FOOTER_HEIGHT = 96;
}

//--------------------------------------------------------------
void ofApp::setup() {
	ofSetWindowTitle("ofxYolo26 :: depth");
	ofSetVerticalSync(true);
	ofBackground(24);

	ofxYolo26::Settings settings;
	// Leave a core or two for the render thread while inference runs alongside it.
	settings.intraOpNumThreads = std::max(1, int(std::thread::hardware_concurrency()) - 2);

	if (!depth.setup(MODEL_PATH, settings)) {
		ofLogError("ofApp") << "could not load " << MODEL_PATH
							<< " -- copy it into bin/data/models/";
		return;
	}
	depth.start();

	still.load(STILL_PATH);
	if (!still.isAllocated()) {
		ofLogError("ofApp") << "could not load " << STILL_PATH;
	}

	setSource(SOURCE_IMAGE);
}

//--------------------------------------------------------------
void ofApp::exit() {
	depth.stop();
}

//--------------------------------------------------------------
void ofApp::setSource(Source newSource) {
	source = newSource;

	if (source == SOURCE_WEBCAM) {
		if (!grabber.isInitialized()) {
			grabber.setDesiredFrameRate(30);
			grabber.setup(1280, 720);
		}
	} else {
		imageNeedsInference = true;
	}
}

//--------------------------------------------------------------
const ofPixels & ofApp::getSourcePixels() const {
	return source == SOURCE_WEBCAM ? grabber.getPixels() : still.getPixels();
}

//--------------------------------------------------------------
void ofApp::update() {
	if (source == SOURCE_WEBCAM) {
		grabber.update();
		if (grabber.isFrameNew()) {
			depth.setInput(grabber.getPixels());
		}
	} else if (imageNeedsInference && still.isAllocated()) {
		// A still only needs one pass; setInput drops the frame while the worker
		// is busy, so keep asking until it is accepted.
		if (depth.setInput(still.getPixels())) {
			imageNeedsInference = false;
		}
	}

	if (!depth.isFrameNew()) return;

	const ofxYolo26::DepthMap & map = depth.getDepthMap();
	if (!map.isAllocated()) return;

	if (useColorMap) {
		ofxYolo26::toColorPixels(map, depthColorPixels, 0.0f, 0.0f, invert, cropToContent);
		if (!depthTexture.isAllocated() || depthTexture.getWidth() != depthColorPixels.getWidth()) {
			depthTexture.allocate(depthColorPixels);
		}
		depthTexture.loadData(depthColorPixels);
	} else {
		ofxYolo26::toNormalizedPixels(map, depthGrayPixels, 0.0f, 0.0f, invert, cropToContent);
		if (!depthTexture.isAllocated() || depthTexture.getWidth() != depthGrayPixels.getWidth()) {
			depthTexture.allocate(depthGrayPixels);
		}
		depthTexture.loadData(depthGrayPixels);
	}
}

//--------------------------------------------------------------
ofRectangle ofApp::getSourceRect() const {
	const ofPixels & px = getSourcePixels();
	const float paneW = (ofGetWidth() - PANE_GAP * 3) * 0.5f;
	const float paneH = ofGetHeight() - FOOTER_HEIGHT - PANE_GAP * 2;
	ofRectangle r(0, 0, px.isAllocated() ? px.getWidth() : 1, px.isAllocated() ? px.getHeight() : 1);
	r.scaleTo(ofRectangle(PANE_GAP, PANE_GAP, paneW, paneH));
	return r;
}

//--------------------------------------------------------------
ofRectangle ofApp::getDepthRect() const {
	// The depth pane mirrors the source pane so the two line up pixel for pixel.
	ofRectangle r = getSourceRect();
	r.x += r.width + PANE_GAP;
	return r;
}

//--------------------------------------------------------------
void ofApp::draw() {
	const ofRectangle srcRect = getSourceRect();

	if (source == SOURCE_WEBCAM) {
		if (grabber.isInitialized()) grabber.draw(srcRect);
	} else if (still.isAllocated()) {
		still.draw(srcRect);
	}

	const ofRectangle depthRect = getDepthRect();
	if (depthTexture.isAllocated()) {
		depthTexture.draw(depthRect);
	} else {
		ofSetColor(48);
		ofDrawRectangle(depthRect);
		ofSetColor(255);
		ofDrawBitmapString("waiting for inference...", depthRect.x + 12, depthRect.y + 24);
	}

	ofNoFill();
	ofSetColor(90);
	ofDrawRectangle(srcRect);
	ofDrawRectangle(depthRect);
	ofFill();
	ofSetColor(255);

	// Probe: read depth under the cursor, in source-image coordinates.
	const ofxYolo26::DepthMap & map = depth.getDepthMap();
	std::string probe = "probe: move the mouse over either pane";
	const bool overSource = srcRect.inside(mouseX, mouseY);
	const bool overDepth = depthRect.inside(mouseX, mouseY);
	const bool hovering = overSource || overDepth;
	const ofRectangle & hoveredPane = overSource ? srcRect : depthRect;

	if (hovering && map.isAllocated()) {
		const float u = (mouseX - hoveredPane.x) / hoveredPane.width;
		const float v = (mouseY - hoveredPane.y) / hoveredPane.height;
		const float value = map.getValueAtSourceNormalized(u, v, std::numeric_limits<float>::quiet_NaN());

		// Echo the cursor onto the other pane, since the two are aligned.
		const ofRectangle & otherPane = overSource ? depthRect : srcRect;
		const float mirrorX = otherPane.x + u * otherPane.width;
		ofSetColor(255, 220, 60);
		ofDrawLine(mouseX - 8, mouseY, mouseX + 8, mouseY);
		ofDrawLine(mouseX, mouseY - 8, mouseX, mouseY + 8);
		ofDrawLine(mirrorX - 8, mouseY, mirrorX + 8, mouseY);
		ofDrawLine(mirrorX, mouseY - 8, mirrorX, mouseY + 8);
		ofSetColor(255);

		probe = std::isnan(value)
			? "probe: outside the frame"
			: "probe: " + ofToString(value, 4) + " raw  (u " + ofToString(u, 3) + ", v " + ofToString(v, 3) + ")";
	}

	const float y = ofGetHeight() - FOOTER_HEIGHT + 8;
	std::string info;
	info += "source: " + std::string(source == SOURCE_WEBCAM ? "webcam" : STILL_PATH);
	info += "   inference: " + ofToString(depth.getLastInferenceTimeMs(), 1) + " ms";
	info += "   inference fps: " + ofToString(depth.getFps(), 1);
	info += "   app fps: " + ofToString(ofGetFrameRate(), 1);
	info += "   backend: " + std::string(ofxYolo26::toString(depth.getDepth().getActiveBackend()));
	info += "\ndepth range: " + ofToString(map.minValue, 4) + " .. " + ofToString(map.maxValue, 4);
	info += "   map: " + ofToString(map.width) + "x" + ofToString(map.height);
	info += "   content: " + ofToString(map.letterbox.contentWidth) + "x" + ofToString(map.letterbox.contentHeight);
	info += "   pad: " + ofToString(map.letterbox.padX) + "," + ofToString(map.letterbox.padY);
	info += "\n" + probe;

	if (showHelp) {
		info += "\n[space] source   [c] colour map: " + std::string(useColorMap ? "on" : "off")
			+ "   [i] invert: " + std::string(invert ? "on" : "off")
			+ "   [x] crop padding: " + std::string(cropToContent ? "on" : "off")
			+ "   [d] dump stats   [h] help";
	}

	ofDrawBitmapStringHighlight(info, 12, y + 12, ofColor(0, 160), ofColor(255));
}

//--------------------------------------------------------------
void ofApp::dumpDepthStats() const {
	const ofxYolo26::DepthMap & map = depth.getDepthMap();
	if (!map.isAllocated()) {
		ofLogNotice("ofApp") << "no depth map yet";
		return;
	}

	ofLogNotice("ofApp") << "depth " << map.width << "x" << map.height
						 << "  range " << map.minValue << " .. " << map.maxValue;
	ofLogNotice("ofApp") << "  corners TL " << map.getValue(0, 0)
						 << "  TR " << map.getValue(map.width - 1, 0)
						 << "  BL " << map.getValue(0, map.height - 1)
						 << "  BR " << map.getValue(map.width - 1, map.height - 1);
	ofLogNotice("ofApp") << "  centre " << map.getValue(map.width / 2, map.height / 2);

	std::string row;
	const int y = map.height * 5 / 8;
	for (int x = 0; x < map.width; x += map.width / 8) {
		row += ofToString(map.getValue(x, y), 3) + " ";
	}
	ofLogNotice("ofApp") << "  row " << y << ": " << row;
}

//--------------------------------------------------------------
void ofApp::keyPressed(int key) {
	switch (key) {
	case ' ':
		setSource(source == SOURCE_IMAGE ? SOURCE_WEBCAM : SOURCE_IMAGE);
		break;
	case 'c':
		useColorMap = !useColorMap;
		depthTexture.clear();
		imageNeedsInference = true;
		break;
	case 'i':
		invert = !invert;
		imageNeedsInference = true;
		break;
	case 'x':
		cropToContent = !cropToContent;
		depthTexture.clear();
		imageNeedsInference = true;
		break;
	case 'd':
		dumpDepthStats();
		break;
	case 'h':
		showHelp = !showHelp;
		break;
	}
}

//--------------------------------------------------------------
void ofApp::keyReleased(int key) { }
void ofApp::mouseMoved(int x, int y) { }
void ofApp::mouseDragged(int x, int y, int button) { }
void ofApp::mousePressed(int x, int y, int button) { }
void ofApp::mouseReleased(int x, int y, int button) { }
void ofApp::mouseEntered(int x, int y) { }
void ofApp::mouseExited(int x, int y) { }
void ofApp::windowResized(int w, int h) { }
void ofApp::gotMessage(ofMessage msg) { }
void ofApp::dragEvent(ofDragInfo dragInfo) { }
