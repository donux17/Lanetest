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

    img_frame.copyTo(output); // �Է� �̹����� ��� �̹����� ����

    // �Է� �̹����� HSV(Hue, Saturation, Value) ���� �������� ��ȯ�մϴ�.
    cvtColor(output, img_hsv, COLOR_BGR2HSV);

    // ��� ���� ������ �����ϰ� ����ũ�� �����մϴ�.
    Scalar lower_white = Scalar(0, 0, 200);
    Scalar upper_white = Scalar(255, 55, 255);
    inRange(img_hsv, lower_white, upper_white, white_mask);

    // ����� ���� ������ �����ϰ� ����ũ�� �����մϴ�.
    Scalar lower_yellow = Scalar(15, 100, 100);
    Scalar upper_yellow = Scalar(35, 255, 255);
    inRange(img_hsv, lower_yellow, upper_yellow, yellow_mask);

    // ����� ����� ����ũ�� �����մϴ�.
    bitwise_or(white_mask, yellow_mask, combined_mask);

    // ���յ� ����ũ�� ���� �̹����� �����մϴ�.
    bitwise_and(output, output, output, combined_mask);

    return output; // ���� ���͸��� �̹����� ��ȯ�մϴ�.
}

Mat RoadLaneDetector::limit_region(Mat img_edges) {
    int width = img_edges.cols;
    int height = img_edges.rows;

    Mat output;
    Mat mask = Mat::zeros(height, width, CV_8UC1); // ���� ������ �����մϴ�.

    // ROI(Region of Interest)�� �����մϴ�: �̹����� �ϴ� ������ �����մϴ�.
    Point points[4]{
        Point(width * 0.1, height),
        Point(width * 0.4, height * 0.6),
        Point(width * 0.6, height * 0.6),
        Point(width * 0.9, height)
    };

    // �ٰ��� ���θ� ä�� ����ũ�� ����ϴ�.
    fillConvexPoly(mask, points, 4, Scalar(255));

    // ���� ������ �̹����� ����ũ�� ��Ʈ������ AND �����Ͽ� ���� ������ ����
    bitwise_and(img_edges, mask, output);

    return output; // ���� �������� ���ѵ� �̹����� ��ȯ
}

vector<Vec4i> RoadLaneDetector::houghLines(Mat img_mask) {
    vector<Vec4i> lines;
    HoughLinesP(img_mask, lines, 1, CV_PI / 180, 20, 20, 30); // ���� ��ȯ�� ����Ͽ� ������ ����
    return lines; // ����� ������ ��ȯ
}

vector<vector<Vec4i>> RoadLaneDetector::separateLine(Mat img_edges, vector<Vec4i> lines) {
    vector<vector<Vec4i>> output(2);
    Point p1, p2;
    vector<double> slopes;
    vector<Vec4i> final_lines, left_lines, right_lines;
    double slope_thresh = 0.3;

    // ����� ������ �߿��� ���Ⱑ ���� �Ӱ谪 �̻��� �������� �����մϴ�.
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

    // �̹��� �߾��� �������� �����ʰ� �������� �������� �и��մϴ�.
    img_center = (double)((img_edges.cols / 2));
    for (int i = 0; i < final_lines.size(); i++) {
        p1 = Point(final_lines[i][0], final_lines[i][1]);
        p2 = Point(final_lines[i][2], final_lines[i][3]);

        if (slopes[i] > 0 && p1.x > img_center && p2.x > img_center) {
            right_detect = true;
            right_lines.push_back(final_lines[i]); // ������ �������� ������ ������ �߰��մϴ�.
        }
        else if (slopes[i] < 0 && p1.x < img_center && p2.x < img_center) {
            left_detect = true;
            left_lines.push_back(final_lines[i]); // ���� �������� ������ ������ �߰��մϴ�.
        }
    }

    output[0] = right_lines; // ������ �������� ������ �������� �����մϴ�.
    output[1] = left_lines; // ���� �������� ������ �������� �����մϴ�.
    return output; // �����ʰ� ���� �������� �и��� �������� ��ȯ�մϴ�.
}

vector<Point> RoadLaneDetector::regression(vector<vector<Vec4i>> separatedLines, Mat img_input) {
    vector<Point> output(4);
    Point p1, p2, p3, p4;
    Vec4d left_line, right_line;
    vector<Point> left_points, right_points;

    // ������ ������ ���� ���� ȸ�͸� �����մϴ�.
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

    // ���� ������ ���� ���� ȸ�͸� �����մϴ�.
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

    // ���� ������ ���� �����ʰ� ���� ������ �������� ����մϴ�.
    int y1 = img_input.rows;
    int y2 = 400;

    double right_x1 = ((y1 - right_b.y) / right_m) + right_b.x;
    double right_x2 = ((y2 - right_b.y) / right_m) + right_b.x;

    double left_x1 = ((y1 - left_b.y) / left_m) + left_b.x;
    double left_x2 = ((y2 - left_b.y) / left_m) + left_b.x;

    // �����ʰ� ���� ������ �� ���� ��ȯ�մϴ�.
    output[0] = Point(right_x1, y1);
    output[1] = Point(right_x2, y2);
    output[2] = Point(left_x1, y1);
    output[3] = Point(left_x2, y2);

    return output; // �����ʰ� ���� ������ �� ���� ��ȯ�մϴ�.
}

// ���� ��Ż�� �����ϴ� �Լ�
bool detectLaneDeparture(vector<Point> lane, int imageWidth) {
    // ������ �߾����� ����մϴ�.
    int laneCenter = (lane[0].x + lane[1].x + lane[2].x + lane[3].x) / 4;

    // �̹����� �߾Ӱ� ���� �߾� ���� �Ÿ��� ����մϴ�.
    int distanceFromCenter = abs(imageWidth / 2 - laneCenter);

    // �̹��� �ʺ��� ���ݿ� �ش��ϴ� �Ÿ��� �������� ��Ż ���θ� �Ǵ��մϴ�.
    if (distanceFromCenter > imageWidth / 2 * 0.5) {
        // ���� ��Ż�� �����Ǿ����ϴ�.
        return true;
    }
    else {
        // ���� ��Ż�� �������� �ʾҽ��ϴ�.
        return false;
    }
}

string RoadLaneDetector::predictDir() {
    string output;
    double x, threshold = 100, thresholdL = 110, threshold2 = 150, threshold3 = 50;
    int y1 = img_input.rows;
    int y2 = 450;

    // ���� ������ ���� �̹��� �߽����� ������, ���� ������ �������� ����մϴ�.
    x = (double)(((right_m * right_b.x) - (left_m * left_b.x) - right_b.y + left_b.y) / (right_m - left_m));
    double right_x2 = ((y2 - right_b.y) / right_m) + right_b.x;
    double left_x2 = ((y2 - left_b.y) / left_m) + left_b.x;
    double right_x1 = ((y1 - right_b.y) / right_m) + right_b.x;
    double left_x1 = ((y1 - left_b.y) / left_m) + left_b.x;

    // ���� ���� ��� ��ȯ
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

    // ���� ��Ż ���� Ȯ��
    bool leftLaneDeparture = (left_x1 < 0 || left_x2 < 0);
    bool rightLaneDeparture = (right_x1 > img_input.cols || right_x2 > img_input.cols);

    // ��Ż�� ���, ������� "Lane Departure"�� ����
    if (leftLaneDeparture || rightLaneDeparture)
        output = "Lane Departure";

    return output;
}

Mat RoadLaneDetector::drawLine(Mat img_input, vector<Point> lane, string dir) {
    vector<Point> poly_points;
    Mat output;
    img_input.copyTo(output); // �Է� �̹����� ��� �̹����� ����

    // ������ �ٰ�������
    poly_points.push_back(lane[2]);
    poly_points.push_back(lane[0]);
    poly_points.push_back(lane[1]);
    poly_points.push_back(lane[3]);
    fillConvexPoly(output, poly_points, Scalar(255, 0, 30), LINE_AA, 0);

    // �̹����� ����ǥ��
    putText(img_input, dir, Point(100, 100), FONT_HERSHEY_COMPLEX, 2, Scalar(255, 255, 255), 3, LINE_AA);

    // �����ʰ� ���� ������ ������
    line(img_input, lane[0], lane[1], Scalar(255, 0, 255), 5, LINE_AA);
    line(img_input, lane[2], lane[3], Scalar(255, 0, 255), 5, LINE_AA);

    // ���� �� ���� ������ ǥ��
    circle(img_input, Point(410, 400), 3, Scalar(255, 255, 255), FILLED);
    circle(img_input, Point(220, 400), 3, Scalar(255, 255, 0), FILLED);
    circle(img_input, Point(630, 400), 3, Scalar(255, 255, 0), FILLED);
    circle(img_input, Point(440, 470), 3, Scalar(255, 255, 0), FILLED);
    circle(img_input, Point(190, 470), 3, Scalar(255, 255, 0), FILLED);
    circle(img_input, Point(25, 5), 3, Scalar(255, 255, 255), FILLED);
    circle(img_input, Point(img_input.cols / 2, 5), 3, Scalar(0, 0, 255), FILLED);
    circle(img_input, Point(img_input.cols / 2, img_input.rows / 2), 3, Scalar(0, 255, 0), FILLED);

    return img_input; // �ϼ��� �̹����� ��ȯ
}
