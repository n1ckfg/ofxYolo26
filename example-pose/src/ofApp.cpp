#include "ofApp.h"

namespace {
const std::string MODEL_PATH = "models/yolo26n-pose.onnx";
const std::string STILL_PATH = "people.jpg";
const int MARGIN = 12;
const int FOOTER_HEIGHT = 92;
}

//--------------------------------------------------------------
void ofApp::setup() {
	ofSetWindowTitle("ofxYolo26 :: pose");
	ofSetVerticalSync(true);
	ofBackground(24);

	ofxYolo26::Settings settings;
	// Leave a core or two for the render thread while inference runs alongside it.
	settings.intraOpNumThreads = std::max(1, int(std::thread::hardware_concurrency()) - 2);

	if (!pose.setup(MODEL_PATH, settings)) {
		ofLogError("ofApp") << "could not load " << MODEL_PATH
							<< " -- copy it into bin/data/models/";
		return;
	}
	pose.getModelRef().setScoreThreshold(scoreThreshold);
	pose.start();

	still.load(STILL_PATH);
	if (!still.isAllocated()) {
		ofLogError("ofApp") << "could not load " << STILL_PATH;
	}

	setSource(SOURCE_IMAGE);
}

//--------------------------------------------------------------
void ofApp::exit() {
	pose.stop();
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
			pose.setInput(grabber.getPixels());
		}
	} else if (imageNeedsInference && still.isAllocated()) {
		// A still only needs one pass; setInput drops the frame while the worker
		// is busy, so keep asking until it is accepted.
		if (pose.setInput(still.getPixels())) {
			imageNeedsInference = false;
		}
	}

	pose.isFrameNew();
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

	const ofxYolo26::PoseResult & result = pose.getPoseResult();
	ofxYolo26::drawPoses(result, frameRect, keypointThreshold, drawBoxes);

	if (drawLabels) {
		for (size_t i = 0; i < result.size(); i++) {
			const ofRectangle box = result.getBoxInSource(i);
			const float sx = frameRect.getWidth() / float(result.letterbox.srcWidth);
			const float sy = frameRect.getHeight() / float(result.letterbox.srcHeight);
			const std::string label = ofToString(i) + " " + ofToString(result.poses[i].detection.score, 2);
			ofDrawBitmapStringHighlight(label,
				frameRect.x + box.x * sx + 3,
				frameRect.y + box.y * sy - 4,
				ofxYolo26::classColor(int(i)),
				ofColor::black);
		}
	}

	ofNoFill();
	ofSetColor(90);
	ofDrawRectangle(frameRect);
	ofFill();
	ofSetColor(255);

	const float y = ofGetHeight() - FOOTER_HEIGHT + 8;
	std::string info;
	info += "source: " + std::string(source == SOURCE_WEBCAM ? "webcam" : STILL_PATH);
	info += "   poses: " + ofToString(result.size());
	info += "   inference: " + ofToString(pose.getLastInferenceTimeMs(), 1) + " ms";
	info += "   inference fps: " + ofToString(pose.getFps(), 1);
	info += "   app fps: " + ofToString(ofGetFrameRate(), 1);
	info += "\nscore >= " + ofToString(scoreThreshold, 2)
		+ "   keypoint >= " + ofToString(keypointThreshold, 2)
		+ "   keypoints per pose: " + ofToString(result.numKeypoints);

	if (showHelp) {
		info += "\n[space] source   [-/=] score threshold   [[/]] keypoint threshold"
				"   [b] boxes: " + std::string(drawBoxes ? "on" : "off")
			+ "   [l] labels: " + std::string(drawLabels ? "on" : "off")
			+ "   [d] dump   [h] help";
	}

	ofDrawBitmapStringHighlight(info, MARGIN, y + 12, ofColor(0, 160), ofColor(255));
}

//--------------------------------------------------------------
void ofApp::dumpPoses() const {
	const ofxYolo26::PoseResult & result = pose.getPoseResult();
	if (result.empty()) {
		ofLogNotice("ofApp") << "no poses";
		return;
	}

	const std::vector<std::string> & names = ofxYolo26::getCocoKeypointNames();
	for (size_t i = 0; i < result.size(); i++) {
		const ofRectangle box = result.getBoxInSource(i);
		ofLogNotice("ofApp") << "pose " << i
							 << " score " << result.poses[i].detection.score
							 << " box (src px) " << box.x << "," << box.y
							 << " " << box.getWidth() << "x" << box.getHeight();

		// A couple of landmarks are enough to sanity-check the mapping.
		for (int k : { ofxYolo26::KP_NOSE, ofxYolo26::KP_LEFT_SHOULDER, ofxYolo26::KP_RIGHT_ANKLE }) {
			if (k >= result.numKeypoints) continue;
			const glm::vec2 p = result.getKeypointInSource(i, k);
			ofLogNotice("ofApp") << "    " << names[k] << " (" << p.x << ", " << p.y
								 << ") score " << result.getKeypointScore(i, k);
		}
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
		// The model lives on the worker thread, so re-run the still afterwards
		// to see the new threshold applied.
		scoreThreshold = ofClamp(scoreThreshold + (key == '-' ? -0.05f : 0.05f), 0.0f, 0.95f);
		pose.getModelRef().setScoreThreshold(scoreThreshold);
		imageNeedsInference = true;
		break;
	case '[':
		keypointThreshold = ofClamp(keypointThreshold - 0.05f, 0.0f, 0.95f);
		break;
	case ']':
		keypointThreshold = ofClamp(keypointThreshold + 0.05f, 0.0f, 0.95f);
		break;
	case 'b':
		drawBoxes = !drawBoxes;
		break;
	case 'l':
		drawLabels = !drawLabels;
		break;
	case 'd':
		dumpPoses();
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
