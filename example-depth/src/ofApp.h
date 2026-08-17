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

	void setSource(Source source);
	const ofPixels & getSourcePixels() const;
	ofRectangle getSourceRect() const;
	ofRectangle getDepthRect() const;
	void dumpDepthStats() const;

	ofxYolo26::DepthThread depth;

	ofImage still;
	ofVideoGrabber grabber;
	Source source = SOURCE_IMAGE;
	bool imageNeedsInference = true;

	ofTexture depthTexture;
	ofPixels depthColorPixels;
	ofFloatPixels depthGrayPixels;

	bool useColorMap = true;
	bool invert = true;
	bool cropToContent = true;
	bool showHelp = true;
};
