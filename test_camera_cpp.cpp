#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    // Open camera with index 2
    cv::VideoCapture cap(2);
    if (!cap.isOpened()) {
        std::cerr << "Cannot open camera" << std::endl;
        return -1;
    }

    cv::Mat frame;
    while (true) {
        // Capture frame
        cap >> frame;
        if (frame.empty()) {
            std::cerr << "Failed to grab frame" << std::endl;
            break;
        }

        // Display frame
        cv::imshow("USB-C Camera", frame);

        // Exit if 'q' is pressed
        char key = (char)cv::waitKey(1);
        if (key == 'q' || key == 'Q') {
            break;
        }
    }

    // Release camera and close windows
    cap.release();
    cv::destroyAllWindows();

    return 0;
}
