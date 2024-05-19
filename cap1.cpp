#include "opencv2/opencv.hpp"
#include <opencv2/highgui/highgui.hpp>
#include "RoadLaneDetector.h"
#include <iostream>
#include <string>
#include <vector>

using namespace cv;
using namespace std;

Mat RoadLaneDetector::filter_colors(Mat img_frame) {
    Mat output;
    UMat img_hsv;
    UMat white_mask, yellow_mask, combined_mask;

    img_frame.copyTo(output); // 입력 이미지를 출력 이미지에 복사

    // 입력 이미지를 HSV(Hue, Saturation, Value) 색상 공간으로 변환합니다.
    cvtColor(output, img_hsv, COLOR_BGR2HSV);

    // 흰색 색상 범위를 정의하고 마스크를 생성합니다.
    Scalar lower_white = Scalar(0, 0, 200);
    Scalar upper_white = Scalar(255, 55, 255);
    inRange(img_hsv, lower_white, upper_white, white_mask);

    // 노란색 색상 범위를 정의하고 마스크를 생성합니다.
    Scalar lower_yellow = Scalar(15, 100, 100);
    Scalar upper_yellow = Scalar(35, 255, 255);
    inRange(img_hsv, lower_yellow, upper_yellow, yellow_mask);

    // 흰색과 노란색 마스크를 결합합니다.
    bitwise_or(white_mask, yellow_mask, combined_mask);

    // 결합된 마스크를 원본 이미지에 적용합니다.
    bitwise_and(output, output, output, combined_mask);

    return output; // 색상 필터링된 이미지를 반환합니다.
}

Mat RoadLaneDetector::limit_region(Mat img_edges) {
    int width = img_edges.cols;
    int height = img_edges.rows;

    Mat output;
    Mat mask = Mat::zeros(height, width, CV_8UC1); // 관심 영역을 생성합니다.

    // ROI(Region of Interest)를 설정합니다: 이미지의 하단 반쪽을 선택합니다.
    Point points[4]{
        Point(width * 0.1, height),
        Point(width * 0.4, height * 0.6),
        Point(width * 0.6, height * 0.6),
        Point(width * 0.9, height)
    };

    // 다각형 내부를 채워 마스크를 만듭니다.
    fillConvexPoly(mask, points, 4, Scalar(255));

    // 엣지 감지된 이미지와 마스크를 비트와이즈 AND 연산하여 관심 영역을 제한
    bitwise_and(img_edges, mask, output);

    return output; // 관심 영역으로 제한된 이미지를 반환
}

vector<Vec4i> RoadLaneDetector::houghLines(Mat img_mask) {
    vector<Vec4i> lines;
    HoughLinesP(img_mask, lines, 1, CV_PI / 180, 20, 20, 30); // 허프 변환을 사용하여 직선을 감지
    return lines; // 검출된 직선을 반환
}

vector<vector<Vec4i>> RoadLaneDetector::separateLine(Mat img_edges, vector<Vec4i> lines) {
    vector<vector<Vec4i>> output(2);
    Point p1, p2;
    vector<double> slopes;
    vector<Vec4i> final_lines, left_lines, right_lines;
    double slope_thresh = 0.3;

    // 검출된 직선들 중에서 기울기가 일정 임계값 이상인 직선들을 선택합니다.
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

    // 이미지 중앙을 기준으로 오른쪽과 왼쪽으로 직선들을 분리합니다.
    img_center = (double)((img_edges.cols / 2));
    for (int i = 0; i < final_lines.size(); i++) {
        p1 = Point(final_lines[i][0], final_lines[i][1]);
        p2 = Point(final_lines[i][2], final_lines[i][3]);

        if (slopes[i] > 0 && p1.x > img_center && p2.x > img_center) {
            right_detect = true;
            right_lines.push_back(final_lines[i]); // 오른쪽 차선으로 감지된 직선을 추가합니다.
        }
        else if (slopes[i] < 0 && p1.x < img_center && p2.x < img_center) {
            left_detect = true;
            left_lines.push_back(final_lines[i]); // 왼쪽 차선으로 감지된 직선을 추가합니다.
        }
    }

    output[0] = right_lines; // 오른쪽 차선으로 감지된 직선들을 저장합니다.
    output[1] = left_lines; // 왼쪽 차선으로 감지된 직선들을 저장합니다.
    return output; // 오른쪽과 왼쪽 차선으로 분리된 직선들을 반환합니다.
}

vector<Point> RoadLaneDetector::regression(vector<vector<Vec4i>> separatedLines, Mat img_input) {
    vector<Point> output(4);
    Point p1, p2, p3, p4;
    Vec4d left_line, right_line;
    vector<Point> left_points, right_points;

    // 오른쪽 차선에 대한 선형 회귀를 수행합니다.
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

    // 왼쪽 차선에 대한 선형 회귀를 수행합니다.
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
        }
    }

    // 차선 예측을 위해 오른쪽과 왼쪽 차선의 교차점을 계산합니다.
    int y1 = img_input.rows;
    int y2 = 400;

    double right_x1 = ((y1 - right_b.y) / right_m) + right_b.x;
    double right_x2 = ((y2 - right_b.y) / right_m) + right_b.x;

    double left_x1 = ((y1 - left_b.y) / left_m) + left_b.x;
    double left_x2 = ((y2 - left_b.y) / left_m) + left_b.x;

    // 오른쪽과 왼쪽 차선의 두 점을 반환합니다.
    output[0] = Point(right_x1, y1);
    output[1] = Point(right_x2, y2);
    output[2] = Point(left_x1, y1);
    output[3] = Point(left_x2, y2);

    return output; // 오른쪽과 왼쪽 차선의 두 점을 반환합니다.
}

// 차선 이탈을 감지하는 함수
bool detectLaneDeparture(vector<Point> lane, int imageWidth) {
    // 차선의 중앙점을 계산합니다.
    int laneCenter = (lane[0].x + lane[1].x + lane[2].x + lane[3].x) / 4;

    // 이미지의 중앙과 차선 중앙 간의 거리를 계산합니다.
    int distanceFromCenter = abs(imageWidth / 2 - laneCenter);

    // 이미지 너비의 절반에 해당하는 거리를 기준으로 이탈 여부를 판단합니다.
    if (distanceFromCenter > imageWidth / 2 * 0.5) {
        // 차선 이탈이 감지되었습니다.
        return true;
    }
    else {
        // 차선 이탈이 감지되지 않았습니다.
        return false;
    }
}

string RoadLaneDetector::predictDir() {
    string output;
    double x, threshold = 100, thresholdL = 110, threshold2 = 150, threshold3 = 50;
    int y1 = img_input.rows;
    int y2 = 450;

    // 차선 예측을 위해 이미지 중심점과 오른쪽, 왼쪽 차선의 교차점을 계산합니다.
    x = (double)(((right_m * right_b.x) - (left_m * left_b.x) - right_b.y + left_b.y) / (right_m - left_m));
    double right_x2 = ((y2 - right_b.y) / right_m) + right_b.x;
    double left_x2 = ((y2 - left_b.y) / left_m) + left_b.x;
    double right_x1 = ((y1 - right_b.y) / right_m) + right_b.x;
    double left_x1 = ((y1 - left_b.y) / left_m) + left_b.x;

    // 차선 예측 결과 반환
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

    // 차선 이탈 여부 확인
    bool leftLaneDeparture = (left_x1 < 0 || left_x2 < 0);
    bool rightLaneDeparture = (right_x1 > img_input.cols || right_x2 > img_input.cols);

    // 이탈한 경우, 결과값을 "Lane Departure"로 변경
    if (leftLaneDeparture || rightLaneDeparture)
        output = "Lane Departure";

    return output;
}

Mat RoadLaneDetector::drawLine(Mat img_input, vector<Point> lane, string dir) {
    vector<Point> poly_points;
    Mat output;
    img_input.copyTo(output); // 입력 이미지를 출력 이미지에 복사

    // 차선을 다각형으로
    poly_points.push_back(lane[2]);
    poly_points.push_back(lane[0]);
    poly_points.push_back(lane[1]);
    poly_points.push_back(lane[3]);
    fillConvexPoly(output, poly_points, Scalar(255, 0, 30), LINE_AA, 0);

    // 이미지에 방향표시
    putText(img_input, dir, Point(100, 100), FONT_HERSHEY_COMPLEX, 2, Scalar(255, 255, 255), 3, LINE_AA);

    // 오른쪽과 왼쪽 차선을 선으로
    line(img_input, lane[0], lane[1], Scalar(255, 0, 255), 5, LINE_AA);
    line(img_input, lane[2], lane[3], Scalar(255, 0, 255), 5, LINE_AA);

    // 차선 및 관심 지점을 표시
    circle(img_input, Point(410, 400), 3, Scalar(255, 255, 255), FILLED);
    circle(img_input, Point(220, 400), 3, Scalar(255, 255, 0), FILLED);
    circle(img_input, Point(630, 400), 3, Scalar(255, 255, 0), FILLED);
    circle(img_input, Point(440, 470), 3, Scalar(255, 255, 0), FILLED);
    circle(img_input, Point(190, 470), 3, Scalar(255, 255, 0), FILLED);
    circle(img_input, Point(25, 5), 3, Scalar(255, 255, 255), FILLED);
    circle(img_input, Point(img_input.cols / 2, 5), 3, Scalar(0, 0, 255), FILLED);
    circle(img_input, Point(img_input.cols / 2, img_input.rows / 2), 3, Scalar(0, 255, 0), FILLED);

    return img_input; // 완성된 이미지를 반환
}
