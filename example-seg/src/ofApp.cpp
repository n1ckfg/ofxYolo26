#include "ofApp.h"

namespace {
const std::string MODEL_PATH = "models/yolo26n-seg.onnx";
const std::string STILL_PATH = "test.jpg";
const int MARGIN = 12;
const int FOOTER_HEIGHT = 92;
}

//--------------------------------------------------------------
void ofApp::setup() {
	ofSetWindowTitle("ofxYolo26 :: segmentation");
	ofSetVerticalSync(true);
	ofBackground(24);

	ofxYolo26::Settings settings;
	// Leave a core or two for the render thread while inference runs alongside it.
	settings.intraOpNumThreads = std::max(1, int(std::thread::hardware_concurrency()) - 2);

	if (!segmentation.setup(MODEL_PATH, settings)) {
		ofLogError("ofApp") << "could not load " << MODEL_PATH
							<< " -- copy it into bin/data/models/";
		return;
	}
	segmentation.getModelRef().setScoreThreshold(scoreThreshold);
	segmentation.start();

	still.load(STILL_PATH);
	if (!still.isAllocated()) {
		ofLogError("ofApp") << "could not load " << STILL_PATH;
	}

	setSource(SOURCE_IMAGE);
}

//--------------------------------------------------------------
void ofApp::exit() {
	segmentation.stop();
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
			segmentation.setInput(grabber.getPixels());
		}
	} else if (imageNeedsInference && still.isAllocated()) {
		// A still only needs one pass; setInput drops the frame while the worker
		// is busy, so keep asking until it is accepted.
		if (segmentation.setInput(still.getPixels())) {
			imageNeedsInference = false;
		}
	}

	if (segmentation.isFrameNew()) rebuildTextures();
}

//--------------------------------------------------------------
void ofApp::rebuildTextures() {
	const ofxYolo26::SegmentationResult & result = segmentation.getSegmentationResult();
	if (result.protoWidth <= 0) return;

	ofxYolo26::toColorPixels(result, overlayPixels, colorByInstance);
	if (!overlayTexture.isAllocated() || overlayTexture.getWidth() != overlayPixels.getWidth()) {
		overlayTexture.allocate(overlayPixels);
	}
	overlayTexture.loadData(overlayPixels);

	ofxYolo26::toAlphaPixels(result, alphaPixels);
	if (!alphaTexture.isAllocated() || alphaTexture.getWidth() != alphaPixels.getWidth()) {
		alphaTexture.allocate(alphaPixels);
	}
	alphaTexture.loadData(alphaPixels);
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
	const ofxYolo26::SegmentationResult & result = segmentation.getSegmentationResult();

	// Masks live on the padded model input, so only the content sub-rectangle
	// lines up with the frame.
	const ofRectangle content = result.getContentRectInProto();

	ofSetColor(255);
	const bool drawFrame = (viewMode != VIEW_MASK);
	if (drawFrame) {
		if (source == SOURCE_WEBCAM) {
			if (grabber.isInitialized()) grabber.draw(frameRect);
		} else if (still.isAllocated()) {
			still.draw(frameRect);
		}
	}

	if (content.getWidth() > 0) {
		if (viewMode == VIEW_OVERLAY && overlayTexture.isAllocated()) {
			ofEnableAlphaBlending();
			overlayTexture.drawSubsection(frameRect.x, frameRect.y,
				frameRect.getWidth(), frameRect.getHeight(),
				content.x, content.y, content.getWidth(), content.getHeight());
		} else if (viewMode == VIEW_MASK && alphaTexture.isAllocated()) {
			alphaTexture.drawSubsection(frameRect.x, frameRect.y,
				frameRect.getWidth(), frameRect.getHeight(),
				content.x, content.y, content.getWidth(), content.getHeight());
		} else if (viewMode == VIEW_CUTOUT && alphaTexture.isAllocated()) {
			// Multiplying the frame by the mask keys it out against black,
			// no shader required.
			ofEnableBlendMode(OF_BLENDMODE_MULTIPLY);
			alphaTexture.drawSubsection(frameRect.x, frameRect.y,
				frameRect.getWidth(), frameRect.getHeight(),
				content.x, content.y, content.getWidth(), content.getHeight());
			ofEnableAlphaBlending();
		}
	}

	if (drawBoxes && result.letterbox.srcWidth > 0) {
		const float sx = frameRect.getWidth() / float(result.letterbox.srcWidth);
		const float sy = frameRect.getHeight() / float(result.letterbox.srcHeight);

		for (size_t i = 0; i < result.size(); i++) {
			const ofRectangle box = result.getBoxInSource(i);
			const ofColor color = ofxYolo26::classColor(
				colorByInstance ? int(i) : result.detections[i].label);

			ofNoFill();
			ofSetColor(color, 180);
			ofDrawRectangle(frameRect.x + box.x * sx, frameRect.y + box.y * sy,
				box.getWidth() * sx, box.getHeight() * sy);
			ofFill();

			const std::string label = result.detections[i].labelName
				+ " " + ofToString(result.detections[i].score, 2);
			ofDrawBitmapStringHighlight(label,
				frameRect.x + box.x * sx + 3,
				frameRect.y + box.y * sy - 4,
				color, ofColor::black);
		}
	}

	ofNoFill();
	ofSetColor(90);
	ofDrawRectangle(frameRect);
	ofFill();
	ofSetColor(255);

	// Probe: which instance owns the pixel under the cursor.
	std::string probe = "probe: move the mouse over the frame";
	if (frameRect.inside(mouseX, mouseY) && result.letterbox.srcWidth > 0) {
		const float srcX = (mouseX - frameRect.x) / frameRect.getWidth() * result.letterbox.srcWidth;
		const float srcY = (mouseY - frameRect.y) / frameRect.getHeight() * result.letterbox.srcHeight;
		int instance = -1;
		const float alpha = result.getAlphaAtSource(srcX, srcY, &instance);
		probe = instance >= 0
			? "probe: " + result.detections[instance].labelName + " #" + ofToString(instance)
				+ "  alpha " + ofToString(alpha, 3)
			: "probe: background";
	}

	const char * viewName = viewMode == VIEW_OVERLAY ? "overlay"
		: viewMode == VIEW_MASK						 ? "mask"
													 : "cutout";

	const float y = ofGetHeight() - FOOTER_HEIGHT + 8;
	std::string info;
	info += "source: " + std::string(source == SOURCE_WEBCAM ? "webcam" : STILL_PATH);
	info += "   instances: " + ofToString(result.size());
	info += "   inference: " + ofToString(segmentation.getLastInferenceTimeMs(), 1) + " ms";
	info += "   inference fps: " + ofToString(segmentation.getFps(), 1);
	info += "   app fps: " + ofToString(ofGetFrameRate(), 1);
	info += "\nview: " + std::string(viewName)
		+ "   score >= " + ofToString(scoreThreshold, 2)
		+ "   masks: " + ofToString(result.protoWidth) + "x" + ofToString(result.protoHeight)
		+ "   " + probe;

	if (showHelp) {
		info += "\n[space] source   [v] view   [-/=] score   [c] colour by "
			+ std::string(colorByInstance ? "instance" : "class")
			+ "   [p] person only: " + std::string(personOnly ? "on" : "off")
			+ "   [b] boxes   [d] dump   [h] help";
	}

	ofDrawBitmapStringHighlight(info, MARGIN, y + 12, ofColor(0, 160), ofColor(255));
}

//--------------------------------------------------------------
void ofApp::dumpInstances() const {
	const ofxYolo26::SegmentationResult & result = segmentation.getSegmentationResult();
	if (result.empty()) {
		ofLogNotice("ofApp") << "no instances";
		return;
	}

	for (size_t i = 0; i < result.size(); i++) {
		const ofRectangle box = result.getBoxInSource(i);
		const ofxYolo26::InstanceMask & mask = result.masks[i];

		float sum = 0.0f;
		int covered = 0;
		if (mask.alpha.isAllocated()) {
			for (size_t p = 0; p < mask.alpha.size(); p++) {
				sum += mask.alpha[p];
				if (mask.alpha[p] > 0.5f) covered++;
			}
		}

		ofLogNotice("ofApp") << "instance " << i << " " << result.detections[i].labelName
							 << " (" << result.detections[i].label << ")"
							 << " score " << result.detections[i].score
							 << " box (src px) " << box.x << "," << box.y
							 << " " << box.getWidth() << "x" << box.getHeight()
							 << " | mask " << mask.alpha.getWidth() << "x" << mask.alpha.getHeight()
							 << " mean alpha " << (mask.alpha.size() ? sum / mask.alpha.size() : 0.0f)
							 << " px>0.5 " << covered;
	}
}

//--------------------------------------------------------------
void ofApp::keyPressed(int key) {
	switch (key) {
	case ' ':
		setSource(source == SOURCE_IMAGE ? SOURCE_WEBCAM : SOURCE_IMAGE);
		break;
	case 'v':
		viewMode = ViewMode((viewMode + 1) % 3);
		break;
	case '-':
	case '=':
	case '+':
		scoreThreshold = ofClamp(scoreThreshold + (key == '-' ? -0.05f : 0.05f), 0.05f, 0.95f);
		segmentation.getModelRef().setScoreThreshold(scoreThreshold);
		imageNeedsInference = true;
		break;
	case 'c':
		colorByInstance = !colorByInstance;
		rebuildTextures();
		break;
	case 'p':
		personOnly = !personOnly;
		if (personOnly) {
			segmentation.getModelRef().setClassFilter({ 0 });
		} else {
			segmentation.getModelRef().clearClassFilter();
		}
		imageNeedsInference = true;
		break;
	case 'b':
		drawBoxes = !drawBoxes;
		break;
	case 'd':
		dumpInstances();
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
