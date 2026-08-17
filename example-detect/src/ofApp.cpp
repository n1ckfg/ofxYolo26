#include "ofApp.h"

namespace {
const std::string MODEL_PATH = "models/yolo26n.onnx";
const std::string STILL_PATH = "test.jpg";
const int MARGIN = 12;
const int FOOTER_HEIGHT = 92;

// COCO ids, for the filter presets.
const std::vector<int> PEOPLE = { 0 };
const std::vector<int> VEHICLES = { 1, 2, 3, 5, 6, 7 }; // bicycle, car, motorcycle, bus, train, truck
}

//--------------------------------------------------------------
void ofApp::setup() {
	ofSetWindowTitle("ofxYolo26 :: detect");
	ofSetVerticalSync(true);
	ofBackground(24);

	ofxYolo26::Settings settings;
	// Leave a core or two for the render thread while inference runs alongside it.
	settings.intraOpNumThreads = std::max(1, int(std::thread::hardware_concurrency()) - 2);

	if (!detector.setup(MODEL_PATH, settings)) {
		ofLogError("ofApp") << "could not load " << MODEL_PATH
							<< " -- copy it into bin/data/models/";
		return;
	}
	detector.getModelRef().setScoreThreshold(scoreThreshold);
	detector.start();

	still.load(STILL_PATH);
	stillName = STILL_PATH;
	if (!still.isAllocated()) {
		ofLogError("ofApp") << "could not load " << STILL_PATH;
	}

	setSource(SOURCE_IMAGE);
}

//--------------------------------------------------------------
void ofApp::exit() {
	detector.stop();
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
void ofApp::applyClassFilter() {
	// The model lives on the worker thread; it is only safe to reconfigure it
	// between frames, which setInput's drop-while-busy behaviour gives us.
	switch (filterMode) {
	case 1: detector.getModelRef().setClassFilter(PEOPLE); break;
	case 2: detector.getModelRef().setClassFilter(VEHICLES); break;
	default: detector.getModelRef().clearClassFilter(); break;
	}
	imageNeedsInference = true;
}

//--------------------------------------------------------------
void ofApp::update() {
	if (source == SOURCE_WEBCAM) {
		grabber.update();
		if (grabber.isFrameNew()) {
			detector.setInput(grabber.getPixels());
		}
	} else if (imageNeedsInference && still.isAllocated()) {
		// A still only needs one pass; setInput drops the frame while the worker
		// is busy, so keep asking until it is accepted.
		if (detector.setInput(still.getPixels())) {
			imageNeedsInference = false;
		}
	}

	detector.isFrameNew();
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

	const ofxYolo26::DetectionResult & result = detector.getDetectionResult();
	ofxYolo26::drawDetections(result, frameRect, drawLabels);

	ofNoFill();
	ofSetColor(90);
	ofDrawRectangle(frameRect);
	ofFill();
	ofSetColor(255);

	const char * filterName = filterMode == 1 ? "people" : filterMode == 2 ? "vehicles" : "all classes";

	const float y = ofGetHeight() - FOOTER_HEIGHT + 8;
	std::string info;
	info += "source: " + std::string(source == SOURCE_WEBCAM ? "webcam" : stillName);
	info += "   objects: " + ofToString(result.size());
	info += "   inference: " + ofToString(detector.getLastInferenceTimeMs(), 1) + " ms";
	info += "   inference fps: " + ofToString(detector.getFps(), 1);
	info += "   app fps: " + ofToString(ofGetFrameRate(), 1);
	info += "   backend: " + std::string(ofxYolo26::toString(detector.getDetector().getActiveBackend()));
	info += "\nscore >= " + ofToString(scoreThreshold, 2)
		+ "   filter: " + std::string(filterName)
		+ "   classes in model: " + ofToString(detector.getDetector().getClassNames().size());

	if (showHelp) {
		info += "\n[space] source   [-/=] score   [f] filter   [l] labels: "
			+ std::string(drawLabels ? "on" : "off")
			+ "   [d] dump   [h] help   -- or drop an image on the window";
	}

	ofDrawBitmapStringHighlight(info, MARGIN, y + 12, ofColor(0, 160), ofColor(255));
}

//--------------------------------------------------------------
void ofApp::dumpDetections() const {
	const ofxYolo26::DetectionResult & result = detector.getDetectionResult();
	if (result.empty()) {
		ofLogNotice("ofApp") << "no detections";
		return;
	}

	for (size_t i = 0; i < result.size(); i++) {
		const ofRectangle box = result.getBoxInSource(i);
		ofLogNotice("ofApp") << "detection " << i << " " << result.detections[i].labelName
							 << " (" << result.detections[i].label << ")"
							 << " score " << result.detections[i].score
							 << " box (src px) " << box.x << "," << box.y
							 << " " << box.getWidth() << "x" << box.getHeight();
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
		scoreThreshold = ofClamp(scoreThreshold + (key == '-' ? -0.05f : 0.05f), 0.05f, 0.95f);
		detector.getModelRef().setScoreThreshold(scoreThreshold);
		imageNeedsInference = true;
		break;
	case 'f':
		filterMode = (filterMode + 1) % 3;
		applyClassFilter();
		break;
	case 'l':
		drawLabels = !drawLabels;
		break;
	case 'd':
		dumpDetections();
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
