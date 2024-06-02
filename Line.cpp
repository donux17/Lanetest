#include "opencv2/opencv.hpp"
#include <opencv2/highgui/highgui.hpp>
#include "RoadLaneDetector.h"
#include <iostream>
#include <string>
#include <vector>


Mat RoadLaneDetector::filter_colors(Mat img_frame) {
	/*
		흰색/노란색 색상의 범위를 정해 해당되는 차선을 필터링한다.
	*/
	Mat output;
	UMat img_hsv, img_combine;
	UMat white_mask, white_image;
	UMat yellow_mask, yellow_image;
	img_frame.copyTo(output);

	//차선 색깔 범위 
	Scalar lower_white = Scalar(200, 200, 200); //흰색 차선 (RGB)
	Scalar upper_white = Scalar(255, 255, 255);
	Scalar lower_yellow = Scalar(10, 100, 100); //노란색 차선 (HSV)
	Scalar upper_yellow = Scalar(40, 255, 255);

	//흰색 필터링
	inRange(output, lower_white, upper_white, white_mask);
	bitwise_and(output, output, white_image, white_mask);

	cvtColor(output, img_hsv, COLOR_BGR2HSV);

	//노란색 필터링
	inRange(img_hsv, lower_yellow, upper_yellow, yellow_mask);
	bitwise_and(output, output, yellow_image, yellow_mask);

	//두 영상을 합친다.
	addWeighted(white_image, 1.0, yellow_image, 1.0, 0.0, output);
	return output;
}


using namespace std;
using namespace cv;

vector<Point> RoadLaneDetector::calculateROI(Mat img_edges) {
	int width = img_edges.cols;
	int height = img_edges.rows;

	// 기울기를 기준으로 ROI 설정
	double roi_height_factor = 0.6;  // 관심 영역 높이를 설정하는 비율
	double roi_top_width_factor = 0.1; // 관심 영역 상단 너비 비율
	double roi_bottom_width_factor = 0.8; // 관심 영역 하단 너비 비율

	Point bottom_left(width * (1 - roi_bottom_width_factor) / 2, height);
	Point bottom_right(width * (1 + roi_bottom_width_factor) / 2, height);
	Point top_left(width * (1 - roi_top_width_factor) / 2, height * (1 - roi_height_factor));
	Point top_right(width * (1 + roi_top_width_factor) / 2, height * (1 - roi_height_factor));

	vector<Point> roi_points = { bottom_left, top_left, top_right, bottom_right };
	return roi_points;
}

Mat RoadLaneDetector::limit_region(Mat img_edges) {
	/*
		관심 영역의 가장자리만 감지되도록 마스킹한다.
		관심 영역의 가장자리만 표시되는 이진 영상을 반환한다.
	*/
	int width = img_edges.cols;
	int height = img_edges.rows;

	Mat output;
	Mat mask = Mat::zeros(height, width, CV_8UC1);

	// 관심 영역 정점 계산
	vector<Point> points = calculateROI(img_edges);

	// 정점으로 정의된 다각형 내부의 색상을 채워 그린다.
	fillConvexPoly(mask, points, Scalar(255, 0, 0));

	// 결과를 얻기 위해 edges 이미지와 mask를 곱한다.
	bitwise_and(img_edges, mask, output);
	return output;
}


vector<Vec4i> RoadLaneDetector::houghLines(Mat img_mask) {
	/*
		관심영역으로 마스킹 된 이미지에서 모든 선을 추출하여 반환
	*/
	vector<Vec4i> line;

	//확률적용 허프변환 직선 검출 함수 
	HoughLinesP(img_mask, line, 1, CV_PI / 180, 20, 10, 20);
	return line;
}

vector<vector<Vec4i>> RoadLaneDetector::separateLine(Mat img_edges, vector<Vec4i> lines) {
	/*
		검출된 모든 허프변환 직선들을 기울기 별로 정렬한다.
		선을 기울기와 대략적인 위치에 따라 좌우로 분류한다.
	*/

	vector<vector<Vec4i>> output(2);
	Point ini, fini;
	vector<double> slopes;
	vector<Vec4i> selected_lines, left_lines, right_lines;
	double slope_thresh = 0.3;

	//검출된 직선들의 기울기를 계산
	for (int i = 0; i < lines.size(); i++) {
		Vec4i line = lines[i];
		ini = Point(line[0], line[1]);
		fini = Point(line[2], line[3]);

		double slope = (static_cast<double>(fini.y) - static_cast<double>(ini.y))
			/ (static_cast<double>(fini.x) - static_cast<double>(ini.x) + 0.00001);

		//기울기가 너무 수평인 선은 제외
		if (abs(slope) > slope_thresh) {
			slopes.push_back(slope);
			selected_lines.push_back(line);
		}
	}

	//선들을 좌우 선으로 분류
	img_center = static_cast<double>((img_edges.cols / 2));
	for (int i = 0; i < selected_lines.size(); i++) {
		ini = Point(selected_lines[i][0], selected_lines[i][1]);
		fini = Point(selected_lines[i][2], selected_lines[i][3]);

		if (slopes[i] > 0 && fini.x > img_center && ini.x > img_center) {
			right_lines.push_back(selected_lines[i]);
			right_detect = true;
		}
		else if (slopes[i] < 0 && fini.x < img_center && ini.x < img_center) {
			left_lines.push_back(selected_lines[i]);
			left_detect = true;
		}
	}

	output[0] = right_lines;
	output[1] = left_lines;
	return output;
}

vector<Point> RoadLaneDetector::regression(vector<vector<Vec4i>> separatedLines, Mat img_input) {
	/*
		선형 회귀를 통해 좌우 차선 각각의 가장 적합한 선을 찾는다.
	*/
	vector<Point> output(4);
	Point ini, fini;
	Point ini2, fini2;
	Vec4d left_line, right_line;
	vector<Point> left_pts, right_pts;

	if (right_detect) {
		for (auto i : separatedLines[0]) {
			ini = Point(i[0], i[1]);
			fini = Point(i[2], i[3]);

			right_pts.push_back(ini);
			right_pts.push_back(fini);
		}

		if (right_pts.size() > 0) {
			//주어진 contour에 최적화된 직선 추출
			fitLine(right_pts, right_line, DIST_L2, 0, 0.01, 0.01);

			right_m = right_line[1] / right_line[0];  //기울기
			right_b = Point(right_line[2], right_line[3]);
		}
	}

	if (left_detect) {
		for (auto j : separatedLines[1]) {
			ini2 = Point(j[0], j[1]);
			fini2 = Point(j[2], j[3]);

			left_pts.push_back(ini2);
			left_pts.push_back(fini2);
		}

		if (left_pts.size() > 0) {
			//주어진 contour에 최적화된 직선 추출
			fitLine(left_pts, left_line, DIST_L2, 0, 0.01, 0.01);

			left_m = left_line[1] / left_line[0];  //기울기
			left_b = Point(left_line[2], left_line[3]);
		}
	}

	// 좌우 선 각각의 두 점을 계산한다.
	// y = m*x + b  --> x = (y-b) / m
	int ini_y = img_input.rows;
	int fin_y = 400; // 새로운 y값

	double right_ini_x = ((ini_y - right_b.y) / right_m) + right_b.x;
	double right_fin_x = ((fin_y - right_b.y) / right_m) + right_b.x;

	double left_ini_x = ((ini_y - left_b.y) / left_m) + left_b.x;
	double left_fin_x = ((fin_y - left_b.y) / left_m) + left_b.x;

	output[0] = Point(right_ini_x, ini_y);
	output[1] = Point(right_fin_x, fin_y);
	output[2] = Point(left_ini_x, ini_y);
	output[3] = Point(left_fin_x, fin_y);

	return output;
}



Mat RoadLaneDetector::drawLine(Mat img_input, vector<Point> lane, string dir) {
	/*
		좌우 차선을 경계로 하는 내부 다각형을 투명하게 색을 채운다.
		좌우 차선을 영상에 선으로 그린다.
		예측 진행 방향 텍스트를 영상에 출력한다.
	*/

	vector<Point> poly_points;
	Mat output;

	img_input.copyTo(output);
	poly_points.push_back(lane[2]);
	poly_points.push_back(lane[0]);
	poly_points.push_back(lane[1]);
	poly_points.push_back(lane[3]);

	fillConvexPoly(output, poly_points, Scalar(0, 230, 30), LINE_AA, 0);  //다각형 색 채우기
	addWeighted(output, 0.3, img_input, 0.7, 0, img_input);  //영상 합하기

	//좌우 차선 선 그리기
	line(img_input, lane[0], lane[1], Scalar(0, 255, 255), 5, LINE_AA);
	line(img_input, lane[2], lane[3], Scalar(0, 255, 255), 5, LINE_AA);

	//예측 진행 방향 텍스트를 영상에 출력
	putText(img_input, dir, Point(520, 100), FONT_HERSHEY_PLAIN, 3, Scalar(255, 255, 255), 3, LINE_AA);

	return img_input;
}

bool RoadLaneDetector::isLaneDeparture(vector<Point> lane, int frameWidth) {
	// 차선의 중앙을 계산합니다.
	double lane_center = (lane[0].x + lane[2].x) / 2.0;

	// 차량의 중앙을 계산합니다.
	double car_center = frameWidth / 2.0;

	// 차선 이탈 여부를 결정합니다.
	double distance_from_center = abs(lane_center - car_center);
	double lane_width = abs(lane[0].x - lane[2].x);

	return (distance_from_center > (lane_width / 2.0));
}