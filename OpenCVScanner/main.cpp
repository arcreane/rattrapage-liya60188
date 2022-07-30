#include <opencv2/opencv.hpp>
#include "opencv2/imgproc.hpp"
#include <opencv2/highgui.hpp>
#include <iostream>

using namespace std;
using namespace cv;

String imageName;
Mat src, grayscaleImg, cannyEdgeImg, gaussBlurImg, matrix, imgwarp, finalImg;
vector<Point> initialpoints, doc_points;


///// Identifying the document to scan /////
Mat edgesDetection(Mat image){
    //Grayscale
    cvtColor(image, grayscaleImg, COLOR_BGR2GRAY);
    //Blurring
    GaussianBlur(grayscaleImg, gaussBlurImg, Size(3, 3), 3, 3);
    //Edge detection
    Canny(gaussBlurImg, cannyEdgeImg, 122, 231);

    return cannyEdgeImg;
}

///// Get contour of the document /////
vector<Point> getcontours(Mat image){
    vector<vector<Point>> contours, contour;
    vector<Vec4i> hierarchy;
    findContours(image, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    
    vector<vector<Point>> conpoly(contours.size());
    vector<Point> drawing;
    int maxArea = 0;

    //To not detect the noise in the image
    for (int i = 0; i < contours.size(); i++)		
    {
        int area = contourArea(contours[i]);
        if (area >= 1000)
        {
            //Count the number of curves or points
            float peri = arcLength(contours[i], true);		
            approxPolyDP(contours[i], conpoly[i], 0.02 * peri, true);

            if (area > maxArea && conpoly[i].size() == 4)
            {
                drawing = { conpoly[i][0] , conpoly[i][1], conpoly[i][2], conpoly[i][3] };
                maxArea = area;
            }
        }
    }

    return drawing;
}

///// Drawing the points on the image /////
void drawpoints(Mat image, vector<Point> points, Scalar Color)
{
    for (int i = 0; i < points.size(); i++)
    {
        circle(src, points[i], 5, Color, FILLED);
    }
}

///// Reorder the points /////
vector<Point> reorder(vector<Point> points)
{
    vector<Point> newpoints;
    vector<int> sumPoints, subPoints;

    for (int i = 0; i < 4; i++)
    {
        sumPoints.push_back(points[i].x + points[i].y);
        subPoints.push_back(points[i].x - points[i].y);
    }
    newpoints.push_back(points[min_element(sumPoints.begin(), sumPoints.end()) - sumPoints.begin()]);
    newpoints.push_back(points[max_element(subPoints.begin(), subPoints.end()) - subPoints.begin()]);
    newpoints.push_back(points[min_element(subPoints.begin(), subPoints.end()) - subPoints.begin()]);
    newpoints.push_back(points[max_element(sumPoints.begin(), sumPoints.end()) - sumPoints.begin()]);

    return newpoints;
}

///// Warping to get the top-down view of the document /////
Mat warping(Mat img, vector<Point> point, float w, float h)
{
    Point2f src[4] = { point[0], point[1] ,point[2], point[3] };
    Point2f dist[4] = { { 0.0f,0.0f }, { w,0.0f }, { 0.0f,h }, { w,h } };

    matrix = getPerspectiveTransform(src, dist);
    warpPerspective(img, imgwarp, matrix, Point(w, h));
    return imgwarp;
}

int main(int, char**) {
    ///// Menu /////
    do {
        cout << "Select the image you want to scan :\n"
            "- Brouillon1\n"
            "- Brouillon2\n" 
            "- Multimedia1\n"
            "- Multimedia2\n" << endl;
        cin >> imageName;
        if (imageName == "Brouillon1" || imageName == "brouillon1") {
            src = imread("Brouillon1.jpg");
        }
        else if (imageName == "Brouillon2" || imageName == "brouillon2") {
            src = imread("Brouillon2.jpg");
        }
        else if (imageName == "Multimedia1" || imageName == "multimedia1") {
            src = imread("Multimedia1.jpg");
        }
        else if (imageName == "Multimedia2" || imageName == "multimedia2") {
            src = imread("Multimedia2.jpg");
        }
        if (src.empty()) {
            cout << "Could not open or find the image\n" << endl;
        }
        waitKey(0);
    } while (src.empty());
    
    ///// Resizing the image /////
    resize(src, src, Size(src.cols / 2.5, src.rows / 2.5));
    ///// Identify document /////
    Mat imgTreshold = edgesDetection(src);
    ///// Get contours of document /////
    initialpoints = getcontours(imgTreshold);
    ///// Reorder points /////
    doc_points = reorder(initialpoints);
    ///// Warp /////
    float w = 420, h = 596;
    finalImg = warping(src, doc_points, w, h);


    imshow("Original", src);
    imshow("Final Image", finalImg);
    waitKey(0);

    return 0;
}