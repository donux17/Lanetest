#include "opencv2/opencv.hpp"
#include <opencv2/highgui/highgui.hpp>
#include "RoadLaneDetector.h"
#include <iostream>
#include <string>
#include <vector>


Mat RoadLaneDetector::filter_colors(Mat img_frame) {

	Mat output;
	UMat img_hsv;
	UMat white_mask, white_image;
	//UMat yellow_mask, yellow_image;
	img_frame.copyTo(output);


	Scalar lower_white = Scalar(252, 252, 252);
	Scalar upper_white = Scalar(255, 255, 255);
	//Scalar lower_yellow = Scalar(10, 100, 100);
	//Scalar upper_yellow = Scalar(40, 255, 255);


	inRange(output, lower_white, upper_white, white_mask);
	bitwise_and(output, output, output, white_mask);

	//cvtColor(output, img_hsv, COLOR_BGR2HSV);


	//inRange(img_hsv, lower_yellow, upper_yellow, yellow_mask);
	//bitwise_and(output, output, yellow_image, yellow_mask);


	//addWeighted(white_image, 1.0, yellow_image, 1.0, 0.0, output);//차선 흰색 + 차선 노란색이 output이다. 그거를 리턴 = img_filter
	return output;
}


Mat RoadLaneDetector::limit_region(Mat img_edges) {

	int width = img_edges.cols;
	int height = img_edges.rows;

	Mat output;
	Mat mask = Mat::zeros(height, width, CV_8UC1);


	Point points[4]{
		Point(200, height),
		Point(230, 400),
		Point(410, 400),
		Point(440, height)
	};

	/*
	Point points[4]{
		Point((width * (1 - poly_bottom_width)) / 2, height),
		Point((width * (1 - poly_top_width)) / 2, height - height * poly_height),
		Point(width - (width * (1 - poly_top_width)) / 2, height - height * poly_height),
		Point(width - (width * (1 - poly_bottom_width)) / 2, height)
	};
	*/
	fillConvexPoly(mask, points, 4, Scalar(255, 0, 0));


	bitwise_and(img_edges, mask, output);
	return output; //zeros 이미지에 관심영역 지정했던곳 파란색으로 채움.그거를 mask라고 하고, 그 후 캐니에지했던 이미지랑 mask를 비트와이즈 앤드 한다. 그거를 output이라하고 그걸 리턴 그게 img_mask임.
}

vector<Vec4i> RoadLaneDetector::houghLines(Mat img_mask) {

	vector<Vec4i> line;

	HoughLinesP(img_mask, line, 1, CV_PI / 180, 20, 10, 20);
	return line;
}

vector<vector<Vec4i>> RoadLaneDetector::separateLine(Mat img_edges, vector<Vec4i> lines) {

	vector<vector<Vec4i>> output(2);
	Point p1, p2;
	vector<double> slopes;
	vector<Vec4i> final_lines, left_lines, right_lines;
	double slope_thresh = 0.3;

	for (int i = 0; i < lines.size(); i++) {
		Vec4i line = lines[i];
		p1 = Point(line[0], line[1]);
		p2 = Point(line[2], line[3]);

		double slope;
		if (p2.x - p1.x == 0)
			slope = 999.0;
		else
			slope = (p2.y - p1.y) / (double)(p2.x - p1.x);


		if (abs(slope) > slope_thresh) {
			slopes.push_back(slope);
			final_lines.push_back(line);
		}
	}

	img_center = (double)((img_edges.cols / 2));

	for (int i = 0; i < final_lines.size(); i++) {
		p1 = Point(final_lines[i][0], final_lines[i][1]);
		p2 = Point(final_lines[i][2], final_lines[i][3]);

		if (slopes[i] > 0 && p1.x > img_center && p2.x > img_center) {
			right_detect = true;
			right_lines.push_back(final_lines[i]);
		}
		else if (slopes[i] < 0 && p1.x < img_center && p2.x < img_center) {
			left_detect = true;
			left_lines.push_back(final_lines[i]);
		}
	}

	output[0] = right_lines;
	output[1] = left_lines;
	return output;
}//우선, 허프변환으로 선분을 구했고, Vec4i line = lines[i]; 이거를 이용해서 선분 끝점의 (x1,y1),(x2,y2)를 구했다.
// 그 후, 기울기 절댓값이 slope_thresh를 넘으면 선분으로 인정.
//그 직선이 양수이고, x1,x2가 둘다 0보다 크면 오른쪽 선분,output[0]
//그 직선이 음수이고, x1,x2가 둘다 0보다 작으면 왼쪽 선분,output[1]으로 보겠다. 그 후 output리턴.


vector<Point> RoadLaneDetector::regression(vector<vector<Vec4i>> separatedLines, Mat img_input) {
	//허프변환 후, 양끝 점의 좌표값이 나왔으니 그 점들을 이용하여 왼쪽 라인 ,오른쪽 라인으로 나누겠다는 내용
	vector<Point> output(4);
	Point p1, p2, p3, p4;
	Vec4d left_line, right_line;
	vector<Point> left_points, right_points;

	if (right_detect) {
		for (auto i : separatedLines[0]) {
			p1 = Point(i[0], i[1]);
			p2 = Point(i[2], i[3]);

			right_points.push_back(p1);
			right_points.push_back(p2);
		}

		if (right_points.size() > 0) {

			fitLine(right_points, right_line, DIST_L2, 0, 0.01, 0.01);

			right_m = right_line[1] / right_line[0];
			right_b = Point(right_line[2], right_line[3]);
		}
	}

	if (left_detect) {
		for (auto j : separatedLines[1]) {
			p3 = Point(j[0], j[1]);
			p4 = Point(j[2], j[3]);

			left_points.push_back(p3);
			left_points.push_back(p4);
		}

		if (left_points.size() > 0) {

			fitLine(left_points, left_line, DIST_L2, 0, 0.01, 0.01);

			left_m = left_line[1] / left_line[0];
			left_b = Point(left_line[2], left_line[3]);
		}//왼쪽라인, 오른쪽라인으로 나누고 나서 fitLine함수 사용해서 직선을 그리기 위한 기울기값이랑 좌표값을 알 수 있다.
	}//separatedLines[0]은 오른쪽 선분을 의미하고, 선분 양 끝점의 좌표정보를 가지고 있는데, 그 점들이 여러개 모여있는 곳에서 fitLine 함수를 사용하여 최적의 선분을 골라내 준다.
	//separatedLines[1]은 왼쪽 선분을 의미하고, 선분 양 끝점의 좌표정보를 가지고 있는데, 그 점들이 여러개 모여있는 곳에서 fitLine 함수를 사용하여 최적의 선분을 골라내 준다.
//그 점들 중에 선택된 시작점과 끝점에 따라서 직선이 중구난방으로 생기게 될 수도 있다.
	int y1 = img_input.rows;
	int y2 = 400;

	double right_x1 = ((y1 - right_b.y) / right_m) + right_b.x;
	double right_x2 = ((y2 - right_b.y) / right_m) + right_b.x;

	double left_x1 = ((y1 - left_b.y) / left_m) + left_b.x;
	double left_x2 = ((y2 - left_b.y) / left_m) + left_b.x;

	output[0] = Point(right_x1, y1);//라인그릴때 사용되는 좌표들
	output[1] = Point(right_x2, y2);
	output[2] = Point(left_x1, y1);
	output[3] = Point(left_x2, y2);
	//이미지 제일 아래에 있는 점의 좌표와 내가 원하는 부분(y값을 이용하여)의 점의 좌표를 구함
	return output;//그리고 그걸 리턴. 그게 lane
}

string RoadLaneDetector::predictDir() {

	string output;
	double x, threshold = 100, thresholdL = 110, threshold2 = 150, threshold3 = 50;
	int y1 = 450;
	int y2 = 450;

	x = (double)(((right_m * right_b.x) - (left_m * left_b.x) - right_b.y + left_b.y) / (right_m - left_m));//소실점 구하기
	double right_x2 = ((y2 - right_b.y) / right_m) + right_b.x;
	double left_x2 = ((y2 - left_b.y) / left_m) + left_b.x;
	double right_x1 = ((y1 - right_b.y) / right_m) + right_b.x;
	double left_x1 = ((y1 - left_b.y) / left_m) + left_b.x;

	if ((left_x1 >= img_center - threshold3) && (right_x2 >= img_center + threshold2))
		output = "line change";
	else if ((right_x1 <= img_center + threshold3) && (left_x2 <= img_center - threshold2))
		output = "line change";
	else if (x >= (img_center - threshold) && x <= (img_center + threshold))
		output = "Straight";
	else if (x > img_center + threshold)
		output = "Right Turn";
	else if (x < img_center - thresholdL)
		output = "Left Turn";
	return output;
}


Mat RoadLaneDetector::drawLine(Mat img_input, vector<Point> lane, string dir) {

	vector<Point> poly_points;
	Mat output;
	img_input.copyTo(output);

	poly_points.push_back(lane[2]);//(left_x1, y1)
	poly_points.push_back(lane[0]);//(right_x1, y1)
	poly_points.push_back(lane[1]);//(right_x2, y2)
	poly_points.push_back(lane[3]);//(left_x2, y2)

	fillConvexPoly(output, poly_points, Scalar(255, 0, 30), LINE_AA, 0);
	addWeighted(output, 0.3, img_input, 0.7, 0, img_input);

	putText(img_input, dir, Point(100, 100), FONT_HERSHEY_COMPLEX, 2, Scalar(255, 255, 255), 3, LINE_AA);
	//Point(100, 100)여기서 Point는 텍스트의 왼쪽 하단 모서리의 위치를 말함.
	line(img_input, lane[0], lane[1], Scalar(255, 0, 255), 5, LINE_AA);
	line(img_input, lane[2], lane[3], Scalar(255, 0, 255), 5, LINE_AA);
	circle(img_input, Point(410, 400), 3, Scalar(255, 255, 255), FILLED);
	circle(img_input, Point(220, 400), 3, Scalar(255, 255, 0), FILLED);
	circle(img_input, Point(630, 400), 3, Scalar(255, 255, 0), FILLED);
	circle(img_input, Point(440, 470), 3, Scalar(255, 255, 0), FILLED);
	circle(img_input, Point(190, 470), 3, Scalar(255, 255, 0), FILLED);
	//circle(img_input, Point(5, 5), 3, Scalar(255,255, 255), FILLED);
	circle(img_input, Point(25, 5), 3, Scalar(255, 255, 255), FILLED);
	circle(img_input, Point(img_input.cols / 2, 5), 3, Scalar(0, 0, 255), FILLED);
	circle(img_input, Point(img_input.cols / 2, img_input.rows / 2), 3, Scalar(0, 255, 0), FILLED);
	return img_input;
}
