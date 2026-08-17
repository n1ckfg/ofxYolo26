#pragma once

#include "ofMain.h"
#include "ofxYolo26.h"

class ofApp : public ofBaseApp {

public:
	void setup();
	void update();
	void draw();

	void keyPressed(int key);
	void keyReleased(int key);
	void mouseMoved(int x, int y);
	void mouseDragged(int x, int y, int button);
	void mousePressed(int x, int y, int button);
	void mouseReleased(int x, int y, int button);
	void mouseEntered(int x, int y);
	void mouseExited(int x, int y);
	void windowResized(int w, int h);
	void dragEvent(ofDragInfo dragInfo);
	void gotMessage(ofMessage msg);
	void exit();

private:
	enum Source {
		SOURCE_IMAGE,
		SOURCE_WEBCAM
	};

	enum ViewMode {
		VIEW_OVERLAY, // masks tinted over the frame
		VIEW_MASK, // the union alpha on its own
		VIEW_CUTOUT // the frame keyed by the union alpha
	};

	void setSource(Source source);
	const ofPixels & getSourcePixels() const;
	ofRectangle getFrameRect() const;
	void rebuildTextures();
	void dumpInstances() const;

	ofxYolo26::SegmentationThread segmentation;

	ofImage still;
	ofVideoGrabber grabber;
	Source source = SOURCE_IMAGE;
	bool imageNeedsInference = true;

	ofTexture overlayTexture;
	ofTexture alphaTexture;
	ofPixels overlayPixels;
	ofFloatPixels alphaPixels;

	ViewMode viewMode = VIEW_OVERLAY;
	float scoreThreshold = 0.2f;
	bool colorByInstance = false;
	bool personOnly = false;
	bool drawBoxes = true;
	bool showHelp = true;
};
