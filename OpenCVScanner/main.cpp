#include <opencv2/opencv.hpp>
#include "opencv2/imgproc.hpp"
#include <opencv2/highgui.hpp>
#include <iostream>

using namespace std;
using namespace cv;

String imageName;
Mat src, imgGrayscale, imgCannyEdge, imgGaussblur, dilatedImg, imgTreshold, matrix, imgwarp, img_final;
vector<Point> initialpoints, docpoints;
float w = 420, h = 596;

///// Identifying the document to scan /////
Mat identify(Mat image){
    //converting to greyscale
    cvtColor(image, imgGrayscale, COLOR_BGR2GRAY);
    //blurring
    GaussianBlur(imgGrayscale, imgGaussblur, Size(3, 3), 3, 3);
    //edge detection
    Canny(imgGaussblur, imgCannyEdge, 122, 231);
    Mat kernel = getStructuringElement(MORPH_RECT, Size(2, 2));
    dilate(imgCannyEdge, dilatedImg, kernel);
    return dilatedImg;

}

///// Get contour of the document /////
vector<Point> getcontours(Mat image){
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(image, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    vector<vector<Point>> conpoly(contours.size());

    vector<Point> bigger;
    int maxArea = 0;

    //Not detect the noise in the image
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
                bigger = { conpoly[i][0] , conpoly[i][1], conpoly[i][2], conpoly[i][3] };
                maxArea = area;
            }
        }
    }
    return bigger;
}

void drawpoints(vector<Point> points, Scalar Color)
{
    for (int i = 0; i < points.size(); i++)
    {
        circle(src, points[i], 5, Color, FILLED);
        putText(src, to_string(i), points[i], FONT_HERSHEY_DUPLEX, 1, Color, 3);
    }
}

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

Mat warping(Mat img, vector<Point> point, float w, float h)
{
    Point2f src[4] = { point[0], point[1] ,point[2], point[3] };
    Point2f dist[4] = { { 0.0f,0.0f }, { w,0.0f }, { 0.0f,h }, { w,h } };

    matrix = getPerspectiveTransform(src, dist);
    warpPerspective(img, imgwarp, matrix, Point(w, h));
    return imgwarp;
}

int main(int, char**) {

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
    
    resize(src, src, Size(src.cols / 2.5, src.rows / 2.5));

    ///// Identify document /////
    imgTreshold = identify(src);
    

    ///// Get contours of document /////
    initialpoints = getcontours(imgTreshold);
    //drawpoints(initialpoints, Scalar(0, 0, 255));
    docpoints = reorder(initialpoints);
    
    ///// Warp /////
    img_final = warping(src, docpoints, w, h);
    drawpoints(docpoints, Scalar(255, 0, 255));

    //// Crop the image /////
    int cropVal = 10;
    Rect roi(cropVal, cropVal, w - (2 * cropVal), h - (2 * cropVal));
    Mat imgcrop = img_final(roi);

    imshow("Original", src);
    //imshow("Image defined", dilatedImg);
    //imshow("Image final", imgwarp);
    imshow("Final", imgcrop);
    waitKey(0);

    return 0;
}