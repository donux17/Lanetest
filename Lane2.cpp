#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <string>
#include <vector>
#include "opencv2/opencv.hpp"
#include "RoadLaneDetector.h"

int main()
{
	RoadLaneDetector roadLaneDetector;
	Mat img_frame, img_filter, img_edges, img_mask, img_lines, img_result;
	vector<Vec4i> lines;
	vector<vector<Vec4i> > separated_lines;
	vector<Point> lane;
	string dir, cha;

	VideoCapture video(0);
	//VideoCapture video("input.mp4");
	if (!video.isOpened()) { cout << "동영상을 열 수 없음\n"; return -1; }


	video.read(img_frame);
	if (img_frame.empty()) return -1;

	video.read(img_frame);
	int cnt = 0;

	while (1) {

		if (!video.read(img_frame)) break;


		img_filter = roadLaneDetector.filter_colors(img_frame);


		cvtColor(img_filter, img_filter, COLOR_BGR2GRAY);


		Canny(img_filter, img_edges, 70, 100); //img_edges는 캐니에지 후에 생긴 이미지


		img_mask = roadLaneDetector.limit_region(img_edges);//관심영역 지정한거(파란색)과 캐니에지 비트와이즈 앤드 한거.


		lines = roadLaneDetector.houghLines(img_mask);//허프변환

		if (lines.size() > 0) {

			separated_lines = roadLaneDetector.separateLine(img_mask, lines);
			lane = roadLaneDetector.regression(separated_lines, img_frame);


			dir = roadLaneDetector.predictDir();


			img_result = roadLaneDetector.drawLine(img_frame, lane, dir);

		}

		imshow("result", img_result);


		if (waitKey(1) == 27)
			break;
	}
	return 0;
}