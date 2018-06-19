//
// Created by root on 6/17/18.
//

#ifndef VD_101_PATHTRACER_H
#define VD_101_PATHTRACER_H
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"


#include <iostream>
#include <ctype.h>
#include <algorithm> // for copy
#include <iterator> // for ostream_iterator
#include <vector>
#include <ctime>
#include <sstream>
#include <fstream>
#include <string>
#include <opencv/cv.h>

using namespace cv;
using namespace std;

#define MIN_NUM_FEAT 200

struct PathTracer {
    vector<Point2f> trajectory;

    PathTracer(Mat img1, Mat img2, bool show=true) : show(show), counter(0), scale(1.0), focal(1000){
        //read the first two frames from the dataset
        Mat img1g, img2g;
        img1 = quantizeImage(img1);
        img2 = quantizeImage(img2);
        // we work with grayscale images
        cvtColor(img1, img1g, COLOR_BGR2GRAY);
        cvtColor(img2, img2g, COLOR_BGR2GRAY);
        prevImage = img2g;

        // feature detection, tracking
        vector<Point2f> points1, points2;        //vectors to store the coordinates of the feature points
        featureDetection(img1g, points1);        //detect features in img_1
        vector<uchar> status;
        featureTracking(img1g,img2g,points1,points2, status); //track those features to img_2

        pp =cv::Point2d(img1g.cols/2, img1g.rows/2);
        //recovering the pose and the essential matrix

        E = findEssentialMat(points2, points1, focal, pp, RANSAC, 0.999, 1.0, mask);
        recoverPose(E, points2, points1, R, t, focal, pp, mask);

        Mat prevImage = img2g;
        Mat currImage;
        prevFeatures = points2;

        R_f = R.clone();
        t_f = t.clone();


        if(show) {
            namedWindow("Trajectory", WINDOW_AUTOSIZE);// Create a window for display.
        }

        traj = Mat::zeros(600, 600, CV_8UC3);
    }

    void addFrame(Mat frame) {
        Mat img;
        frame = quantizeImage(frame);
        cvtColor(frame, img, COLOR_BGR2GRAY);
        vector<uchar> status;
        featureTracking(prevImage, img, prevFeatures, currFeatures, status);

        E = findEssentialMat(currFeatures, prevFeatures, focal, pp, RANSAC, 0.999, 1.0, mask);
        recoverPose(E, currFeatures, prevFeatures, R, t, focal, pp, mask);

        Mat prevPts(2, prevFeatures.size(), CV_64F), currPts(2, currFeatures.size(), CV_64F);


        for (int i = 0; i <
            prevFeatures.size(); i++) {   //this (x,y) combination makes sense as observed from the source code of triangulatePoints on GitHub
            prevPts.at<double>(0, i) = prevFeatures.at(i).x;
            prevPts.at<double>(1, i) = prevFeatures.at(i).y;

            currPts.at<double>(0, i) = currFeatures.at(i).x;
            currPts.at<double>(1, i) = currFeatures.at(i).y;
        }

        scale = getAbsoluteScale(counter++, 0, t.at<double>(2));

        if ((scale > 0.1) && (t.at<double>(2) > t.at<double>(0)) && (t.at<double>(2) > t.at<double>(1))) {

            t_f = t_f + scale * (R_f * t);
            R_f = R * R_f;

        } else {
            //cout << "scale below 0.1, or incorrect translation" << endl;
        }

        // a redetection is triggered in case the number of feautres being trakced go below a particular threshold
        if (prevFeatures.size() < MIN_NUM_FEAT) {
            //cout << "Number of tracked features reduced to " << prevFeatures.size() << endl;
            //cout << "trigerring redection" << endl;
            featureDetection(prevImage, prevFeatures);
            featureTracking(prevImage, img, prevFeatures, currFeatures, status);

        }

        prevImage = img.clone();
        prevFeatures = currFeatures;
        auto point = Point2f(t_f.at<double>(0) + 300, t_f.at<double>(2) + 100);
        trajectory.push_back(point);

        if(show) {
            circle(traj, point, 1, CV_RGB(255, 0, 0), 2);
            //rectangle(traj, Point(10, 30), Point(550, 50), CV_RGB(0, 0, 0), CV_FILLED);

        }
    }

    Point2f real_position = Point2f(300, 100);
    int angle = 90;
    void RealAction(vizdoom::Button b) {
        auto distance = 10;

        switch(b) {
            case vizdoom::MOVE_RIGHT:{
                real_position.x += cos((angle + 90)* M_PI /180.0) * distance; real_position.y += sin((angle + 90)* M_PI /180.0) * distance;
                break;
            }
            case vizdoom::MOVE_LEFT: {
                real_position.x += cos((angle - 90)* M_PI /180.0) * distance; real_position.y += sin((angle - 90)* M_PI /180.0) * distance;
                break;
            }
            case vizdoom::MOVE_BACKWARD: {
                real_position.x -= cos(angle * M_PI /180.0) * distance; real_position.y -= sin(angle * M_PI /180.0) * distance;
                break;
            }
            case vizdoom::MOVE_FORWARD:{
                real_position.x += cos(angle * M_PI /180.0) * distance; real_position.y += sin(angle * M_PI /180.0) * distance;
                break;
            }
            case vizdoom::TURN_RIGHT:{
                angle += 7;
                break;
            }
            case vizdoom::TURN_LEFT: {
                angle -= 7;
               break;
            }
            }
            if(show) {
                circle(traj, real_position, 1, CV_RGB(255, 0, 255), 2);
                imshow("Trajectory", traj);
            }
    }

private:
    bool show;
    int counter;
    double scale, focal;
    cv::Point2d pp;
    vector<Point2f> prevFeatures, currFeatures;
    Mat prevImage, traj, mask;
    Mat E, R, t;
    Mat R_f, t_f; //the final rotation and tranlation vectors

    cv::Mat quantizeImage(const cv::Mat& inImage, int numBits=4)
    {

        cv::Mat retImage;
        //
        retImage = inImage.clone();
        /*
        uchar maskBit = 0xFF;

        // keep numBits as 1 and (8 - numBits) would be all 0 towards the right
        maskBit = maskBit << (8 - numBits);

        for(int j = 0; j < retImage.rows; j++)
            for(int i = 0; i < retImage.cols; i++)
            {
                cv::Vec3b valVec = retImage.at<cv::Vec3b>(j, i);
                valVec[0] = valVec[0] & maskBit;
                valVec[1] = valVec[1] & maskBit;
                valVec[2] = valVec[2] & maskBit;
                retImage.at<cv::Vec3b>(j, i) = valVec;
            }
        */
        //blur(retImage,retImage, Size(10,10));
        //imshow("dbg", retImage);
        return retImage ;
    }

    //this function automatically gets rid of points for which tracking fails
    void featureTracking(Mat img_1, Mat img_2, vector<Point2f>& points1, vector<Point2f>& points2, vector<uchar>& status)	{
        vector<float> err;
        Size winSize=Size(21,21);
        TermCriteria termcrit=TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.01);

        calcOpticalFlowPyrLK(img_1, img_2, points1, points2, status, err, winSize, 3, termcrit, 0, 0.001);

        //getting rid of points for which the KLT tracking failed or those who have gone outside the frame
        int indexCorrection = 0;
        for( int i=0; i<status.size(); i++)
        {  Point2f pt = points2.at(i- indexCorrection);
            if ((status.at(i) == 0)||(pt.x<0)||(pt.y<0))	{
                if((pt.x<0)||(pt.y<0))	{
                    status.at(i) = 0;
                }
                points1.erase (points1.begin() + (i - indexCorrection));
                points2.erase (points2.begin() + (i - indexCorrection));
                indexCorrection++;
            }

        }

    }


    void featureDetection(Mat img_1, vector<Point2f>& points1)	{   //uses FAST as of now, modify parameters as necessary
        vector<KeyPoint> keypoints_1;
        int fast_threshold = 20;
        bool nonmaxSuppression = true;
        FAST(img_1, keypoints_1, fast_threshold, nonmaxSuppression);
        KeyPoint::convert(keypoints_1, points1, vector<int>());
    }

    double getAbsoluteScale(int frame_id, int sequence_id, double z_cal)	{
/*
        string line;
        int i = 0;
        ifstream myfile ("/home/avisingh/Datasets/KITTI_VO/00.txt");
        double x =0, y=0, z = 0;
        double x_prev, y_prev, z_prev;
        if (myfile.is_open())
        {
            while (( getline (myfile,line) ) && (i<=frame_id))
            {
                z_prev = z;
                x_prev = x;
                y_prev = y;
                std::istringstream in(line);
                //cout << line << '\n';
                for (int j=0; j<12; j++)  {
                    in >> z ;
                    if (j==7) y=z;
                    if (j==3)  x=z;
                }

                i++;
            }
            myfile.close();
        }

        else {
            cout << "Unable to open file";
            return 0;
        }*/

        return 10; //sqrt((x-x_prev)*(x-x_prev) + (y-y_prev)*(y-y_prev) + (z-z_prev)*(z-z_prev)) ;
    }
};


#endif //VD_101_PATHTRACER_H
