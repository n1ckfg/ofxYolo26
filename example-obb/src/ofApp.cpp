#include "ofApp.h"

namespace {
const std::string MODEL_PATH = "models/yolo26n-obb.onnx";
const std::string STILL_PATH = "test.jpg";
const int MARGIN = 12;
const int FOOTER_HEIGHT = 110;
}

//--------------------------------------------------------------
void ofApp::setup() {
	ofSetWindowTitle("ofxYolo26 :: oriented bounding boxes");
	ofSetVerticalSync(true);
	ofBackground(24);

	ofxYolo26::Settings settings;
	// Leave a core or two for the render thread while inference runs alongside it.
	settings.intraOpNumThreads = std::max(1, int(std::thread::hardware_concurrency()) - 2);

	if (!obb.setup(MODEL_PATH, settings)) {
		ofLogError("ofApp") << "could not load " << MODEL_PATH
							<< " -- copy it into bin/data/models/";
		return;
	}
	obb.getModelRef().setScoreThreshold(scoreThreshold);
	obb.start();

	ofLogNotice("ofApp") << "yolo26n-obb is trained on DOTA (aerial imagery). Its classes are "
							"plane, ship, storage tank, harbour, roundabout and so on, so an "
							"ordinary photo scores near zero. Drop an aerial or satellite image "
							"on the window to see it work.";

	still.load(STILL_PATH);
	stillName = STILL_PATH;
	if (!still.isAllocated()) {
		ofLogError("ofApp") << "could not load " << STILL_PATH;
	}

	setSource(SOURCE_IMAGE);
}

//--------------------------------------------------------------
void ofApp::exit() {
	obb.stop();
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
			obb.setInput(grabber.getPixels());
		}
	} else if (imageNeedsInference && still.isAllocated()) {
		// A still only needs one pass; setInput drops the frame while the worker
		// is busy, so keep asking until it is accepted.
		if (obb.setInput(still.getPixels())) {
			imageNeedsInference = false;
		}
	}

	obb.isFrameNew();
}

//--------------------------------------------------------------
ofRectangle ofApp::getFrameRect() const {
	const ofPixels & px = getSourcePixels();
	ofRectangle r(0, 0, px.isAllocated() ? px.getWidth() : 1, px.isAllocated() ? px.getHeight() : 1);
	r.scaleTo(ofRectangle(MARGIN, MARGIN,
		ofGetWidth() - MARGIN * 2,
		ofGetHeight() - FOOTER_HEIGHT - MARGIN * 2));
	return r;
}

//--------------------------------------------------------------
void ofApp::draw() {
	const ofRectangle frameRect = getFrameRect();

	ofSetColor(255);
	if (source == SOURCE_WEBCAM) {
		if (grabber.isInitialized()) grabber.draw(frameRect);
	} else if (still.isAllocated()) {
		still.draw(frameRect);
	}

	const ofxYolo26::ObbResult & result = obb.getObbResult();
	ofxYolo26::drawOrientedBoxes(result, frameRect, drawLabels, drawHeading);

	ofNoFill();
	ofSetColor(90);
	ofDrawRectangle(frameRect);
	ofFill();
	ofSetColor(255);

	const float y = ofGetHeight() - FOOTER_HEIGHT + 8;
	std::string info;
	info += "source: " + std::string(source == SOURCE_WEBCAM ? "webcam" : stillName);
	info += "   boxes: " + ofToString(result.size());
	info += "   inference: " + ofToString(obb.getLastInferenceTimeMs(), 1) + " ms";
	info += "   inference fps: " + ofToString(obb.getFps(), 1);
	info += "   app fps: " + ofToString(ofGetFrameRate(), 1);
	info += "   backend: " + std::string(ofxYolo26::toString(obb.getObb().getActiveBackend()));
	info += "\nscore >= " + ofToString(scoreThreshold, 2)
		+ "   model input: " + ofToString(obb.getObb().getInputWidth()) + "x"
		+ ofToString(obb.getObb().getInputHeight()) + " (declared dynamic, pinned from imgsz)";

	// The bundled photo is not aerial, so say so rather than letting an empty
	// screen read as a broken model.
	if (result.empty()) {
		info += "\nno boxes -- this model knows DOTA aerial classes (plane, ship, harbour...);"
				" drop an aerial image on the window";
	} else {
		info += "\ntop: " + result.boxes[0].labelName + " " + ofToString(result.boxes[0].score, 2)
			+ " at " + ofToString(ofRadToDeg(result.getAngleInSource(0)), 1) + " deg";
	}

	if (showHelp) {
		info += "\n[space] source   [-/=] score   [l] labels   [o] heading: "
			+ std::string(drawHeading ? "on" : "off")
			+ "   [d] dump   [h] help   -- or drop an image on the window";
	}

	ofDrawBitmapStringHighlight(info, MARGIN, y + 12, ofColor(0, 160), ofColor(255));
}

//--------------------------------------------------------------
void ofApp::dumpBoxes() const {
	const ofxYolo26::ObbResult & result = obb.getObbResult();
	if (result.empty()) {
		ofLogNotice("ofApp") << "no oriented boxes";
		return;
	}

	for (size_t i = 0; i < result.size(); i++) {
		const ofxYolo26::OrientedBox & box = result.boxes[i];
		const glm::vec2 center = result.getCenterInSource(i);
		const std::array<glm::vec2, 4> corners = result.getCornersInSource(i);

		ofLogNotice("ofApp") << "box " << i << " " << box.labelName << " (" << box.label << ")"
							 << " score " << box.score
							 << " centre (src px) " << center.x << "," << center.y
							 << " size " << box.size.x << "x" << box.size.y
							 << " angle " << ofRadToDeg(result.getAngleInSource(i)) << " deg";

		std::string cornerText;
		for (const glm::vec2 & c : corners) {
			cornerText += "(" + ofToString(c.x, 1) + "," + ofToString(c.y, 1) + ") ";
		}
		ofLogNotice("ofApp") << "    corners " << cornerText;
	}
}

//--------------------------------------------------------------
void ofApp::keyPressed(int key) {
	switch (key) {
	case ' ':
		setSource(source == SOURCE_IMAGE ? SOURCE_WEBCAM : SOURCE_IMAGE);
		break;
	case '-':
	case '=':
	case '+':
		scoreThreshold = ofClamp(scoreThreshold + (key == '-' ? -0.05f : 0.05f), 0.01f, 0.95f);
		obb.getModelRef().setScoreThreshold(scoreThreshold);
		imageNeedsInference = true;
		break;
	case 'l':
		drawLabels = !drawLabels;
		break;
	case 'o':
		drawHeading = !drawHeading;
		break;
	case 'd':
		dumpBoxes();
		break;
	case 'h':
		showHelp = !showHelp;
		break;
	}
}

//--------------------------------------------------------------
void ofApp::dragEvent(ofDragInfo dragInfo) {
	if (dragInfo.files.empty()) return;

	ofImage dropped;
	if (!dropped.load(dragInfo.files[0])) {
		ofLogError("ofApp") << "could not load " << dragInfo.files[0];
		return;
	}
	still = dropped;
	stillName = ofFilePath::getFileName(dragInfo.files[0]);
	setSource(SOURCE_IMAGE);
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
