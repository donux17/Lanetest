#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <string>
#include <vector>
#include "opencv2/opencv.hpp"
#include "RoadLaneDetector.h"

int main() {
    RoadLaneDetector roadLaneDetector;
    Mat img_frame, img_filter, img_edges, img_mask, img_result;
    vector<Vec4i> lines;
    vector<vector<Vec4i>> separated_lines;
    vector<Point> lane;
    string dir;

    VideoCapture video(0);
    if (!video.isOpened()) {
        cout << "�������� �� �� ����\n";
        return -1;
    }

    while (1) {
        if (!video.read(img_frame)) break;

        img_filter = roadLaneDetector.filter_colors(img_frame); // ���� ���͸��� ���� ������ �����մϴ�.
        cvtColor(img_filter, img_filter, COLOR_BGR2GRAY); // �׷��̽����� �̹����� ��ȯ�մϴ�.
        Canny(img_filter, img_edges, 50, 150); // ĳ�� ���� ������ �����մϴ�.

        img_mask = roadLaneDetector.limit_region(img_edges); // ���� ������ �����մϴ�.
        lines = roadLaneDetector.houghLines(img_mask); // ���� ��ȯ�� ����Ͽ� ������ �����մϴ�.

        if (lines.size() > 0) { // ����� ������ �ִ� ���
            separated_lines = roadLaneDetector.separateLine(img_mask, lines); // ���� �� ������ �������� �и��մϴ�.
            lane = roadLaneDetector.regression(separated_lines, img_frame); // ������ �����մϴ�.
            dir = roadLaneDetector.predictDir(); // ������ �̵� ������ �����մϴ�.
            img_result = roadLaneDetector.drawLine(img_frame, lane, dir); // ���� �� �̵� ������ �̹����� �׸��ϴ�.
        }

        imshow("result", img_result);

        if (waitKey(1) == 27) break;
    }

    return 0;
}