#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <string>
#include <vector>
#include "opencv2/opencv.hpp"
#include "RoadLaneDetector.h"

int main() {
    RoadLaneDetector roadLaneDetector;
    Mat img_frame, img_filter, img_edges, img_mask, img_lines, img_result;
    vector<Vec4i> lines;
    vector<vector<Vec4i>> separated_lines;
    vector<Point> lane;
    string dir;

    VideoCapture video(0);
    if (!video.isOpened()) {
        cout << "Not open Camera\n";
        return -1;
    }

    while (1) {
        if (!video.read(img_frame)) break;

        img_filter = roadLaneDetector.filter_colors(img_frame); // ���� ���͸��� ���� ������ �����մϴ�.
        cvtColor(img_filter, img_filter, COLOR_BGR2GRAY); // �׷��̽����� �̹����� ��ȯ�մϴ�.
        Canny(img_filter, img_edges, 50, 150); // ĳ�� ���� ������ �����մϴ�.
        img_mask = roadLaneDetector.limit_region(img_edges); // ���� ������ �����մϴ�.
        lines = roadLaneDetector.houghLines(img_mask); // Hough ��ȯ���� ���������� ���� ������ ����

        if (lines.size() > 0) { // ����� ������ �ִ� ���
            separated_lines = roadLaneDetector.separateLine(img_mask, lines); // ���� �� ������ �������� �и��մϴ�.
            lane = roadLaneDetector.regression(separated_lines, img_frame); // ������ �����մϴ�.
            img_result = roadLaneDetector.drawLine(img_frame, lane, dir); // ���� �� �̵� ������ �̹����� �׸��ϴ�.

            // ���� ��Ż ���θ� �����մϴ�.
            if (roadLaneDetector.isLaneDeparture(lane, img_frame.cols)) {
                putText(img_result, "Lane Departure!", Point(50, 50), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2, LINE_AA);
            }
        }

        imshow("result", img_result);

        if (waitKey(1) == 27) break;
    }

    return 0;
}