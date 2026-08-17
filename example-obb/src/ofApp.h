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
	ofRectangle getFrameRect() const;
	void dumpBoxes() const;

	ofxYolo26::ObbThread obb;

	ofImage still;
	std::string stillName;
	ofVideoGrabber grabber;
	Source source = SOURCE_IMAGE;
	bool imageNeedsInference = true;

	// The DOTA classes this model knows are aerial, so the bundled street photo
	// scores very low. Start permissive so there is something on screen, and
	// tell the user to drop in a real aerial image.
	float scoreThreshold = 0.25f;
	bool drawLabels = true;
	bool drawHeading = true;
	bool showHelp = true;
};
