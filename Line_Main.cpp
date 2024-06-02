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

        img_filter = roadLaneDetector.filter_colors(img_frame); // 색상 필터링을 통해 차선을 감지합니다.
        cvtColor(img_filter, img_filter, COLOR_BGR2GRAY); // 그레이스케일 이미지로 변환합니다.
        Canny(img_filter, img_edges, 50, 150); // 캐니 엣지 검출을 수행합니다.
        img_mask = roadLaneDetector.limit_region(img_edges); // 관심 영역을 설정합니다.
        lines = roadLaneDetector.houghLines(img_mask); // Hough 변환으로 에지에서의 직선 성분을 추출

        if (lines.size() > 0) { // 검출된 직선이 있는 경우
            separated_lines = roadLaneDetector.separateLine(img_mask, lines); // 왼쪽 및 오른쪽 차선으로 분리합니다.
            lane = roadLaneDetector.regression(separated_lines, img_frame); // 차선을 추정합니다.
            img_result = roadLaneDetector.drawLine(img_frame, lane, dir); // 차선 및 이동 방향을 이미지에 그립니다.

            // 차선 이탈 여부를 감지합니다.
            if (roadLaneDetector.isLaneDeparture(lane, img_frame.cols)) {
                putText(img_result, "Lane Departure!", Point(50, 50), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2, LINE_AA);
            }
        }

        imshow("result", img_result);

        if (waitKey(1) == 27) break;
    }

    return 0;
}