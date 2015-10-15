/**
* This file is part of LSD-SLAM.
*
* Copyright 2013 Jakob Engel <engelj at in dot tum dot de> (Technical University of Munich)
* For more information see <http://vision.in.tum.de/lsdslam> 
*
* LSD-SLAM is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* LSD-SLAM is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with LSD-SLAM. If not, see <http://www.gnu.org/licenses/>.
*/

#include "LiveSLAMWrapper.h"
#include <iostream>
#include <fstream>
#include <list>
#include <string>
#include <boost/thread.hpp>
#include "util/settings.h"
#include "util/globalFuncs.h"
#include "SlamSystem.h"
#include "DataStructures/types.h"

#include "IOWrapper/ROS/rosReconfigure.h"

#include <X11/Xlib.h>
#include "dense_fullba_rgbd/LSDParamsConfig.h"
#include "ros/ros.h"
#include "ros/package.h"
#include "cv_bridge/cv_bridge.h"

#include "visensor_node/visensor_imu.h"
#include "visensor_node/visensor_calibration.h"

using namespace lsd_slam;
using namespace std ;
using namespace cv ;

CALIBRATION_PAR calib_par ;
ros::Subscriber sub_image[2];
ros::Subscriber sub_imu;
LiveSLAMWrapper* globalLiveSLAM = NULL ;

bool initCalibrationPar(string caliFilePath)
{
    //read calibration parameters
    std::ifstream f(caliFilePath.c_str());
    if (!f.good())
    {
        f.close();
        printf(" %s not found!\n", caliFilePath.c_str());
        return false;
    }
    std::string l1, l2;
    std::getline(f,l1);
    std::getline(f,l2);
    f.close();

    if(std::sscanf(l1.c_str(), "%f %f %f %f %f %f %f %f",
                   &calib_par.fx, &calib_par.fy, &calib_par.cx, &calib_par.cy,
                   &calib_par.d[0], &calib_par.d[1], &calib_par.d[2], &calib_par.d[3]) != 8 )
    {
        puts("calibration file format error 1") ;
        return false ;
    }
    if(std::sscanf(l2.c_str(), "%d %d", &calib_par.width, &calib_par.height ) != 2)
    {
        puts("calibration file format error 2") ;
        return false ;
    }

    return true ;
}

void image0CallBack(const sensor_msgs::ImageConstPtr& msg)
{
    ros::Time tImage = msg->header.stamp;
    cv::Mat   image  = cv_bridge::toCvShare(msg, std::string("mono8"))->image;
    globalLiveSLAM->image0_queue_mtx.lock();
    globalLiveSLAM->image0Buf.push_back(ImageMeasurement(tImage, image));
    globalLiveSLAM->image0_queue_mtx.unlock();
}

void image1CallBack(const sensor_msgs::ImageConstPtr& msg)
{
    ros::Time tImage = msg->header.stamp;
    cv::Mat   image  = cv_bridge::toCvShare(msg, std::string("mono8"))->image;
    globalLiveSLAM->image1_queue_mtx.lock();
    globalLiveSLAM->image1Buf.push_back(ImageMeasurement(tImage, image));
    globalLiveSLAM->image1_queue_mtx.unlock();
}

void imuCallBack(const visensor_node::visensor_imu& imu_msg )
{
    globalLiveSLAM->imu_queue_mtx.lock();
    globalLiveSLAM->imuQueue.push_back( imu_msg );
    globalLiveSLAM->imu_queue_mtx.unlock();
}

void process_image()
{
    globalLiveSLAM->Loop();
}

void process_BA()
{
    globalLiveSLAM->BALoop();
}

Mat cameraMatrix(3, 3, CV_64FC1 ) ;
Mat discoffs(1, 5, CV_64FC1 ) ;
Mat map1, map2 ;
char filePath[256] = "/home/ygling2008/tum_dataset/rgbd_dataset_freiburg1_xyz/" ;
list<string> rgbFileList;
list<string> depthFileList;

void initPara()
{
    calib_par.width = 640 ;
    calib_par.height = 480 ;
//    calib_par.fx = 517.3 ;
//    calib_par.fy = 516.5 ;
//    calib_par.cx = 318.6 ;
//    calib_par.cy = 255.3 ;
//    calib_par.d[0] = 0.2624 ;
//    calib_par.d[1] = -0.9531 ;
//    calib_par.d[2] = -0.0054 ;
//    calib_par.d[3] = 0.0026 ;
//    calib_par.d[4] = 1.1633 ;
    calib_par.fx = 525.0 ;
    calib_par.fy = 525.0 ;
    calib_par.cx = 319.5 ;
    calib_par.cy = 239.5 ;
    calib_par.d[0] = 0;
    calib_par.d[1] = 0 ;
    calib_par.d[2] = 0 ;
    calib_par.d[3] = 0 ;
    calib_par.d[4] = 0 ;

    cameraMatrix.at<double>(0, 0) = calib_par.fx ;
    cameraMatrix.at<double>(0, 1) = 0 ;
    cameraMatrix.at<double>(0, 2) = calib_par.cx ;
    cameraMatrix.at<double>(1, 0) = 0 ;
    cameraMatrix.at<double>(1, 1) = calib_par.fy ;
    cameraMatrix.at<double>(1, 2) = calib_par.cy ;
    cameraMatrix.at<double>(2, 0) = 0 ;
    cameraMatrix.at<double>(2, 1) = 0 ;
    cameraMatrix.at<double>(2, 2) = 1 ;

    for ( int i = 0 ; i < 5 ; i++ ){
        discoffs.at<double>(0, i) = calib_par.d[i];
    }
     initUndistortRectifyMap(cameraMatrix, discoffs, Mat(), Mat(), Size(calib_par.width, calib_par.height), CV_32FC1, map1, map2);
}


void initFileList()
{
    char tmp[256];
    char t1[256], t2[256], t3[256], t4[256] ;
    FILE *fp = NULL;

    strcpy(tmp, filePath ) ;
    strcat(tmp, "match.txt" ) ;
    fp = fopen(tmp, "r");
    if (fp == NULL){
        puts("imgList Path error");
        return;
    }
    while (fgets(tmp, 256, fp) != NULL)
    {
        sscanf(tmp, "%s %s %s %s\n", &t1, &t2, &t3, &t4);
        rgbFileList.push_back( string(t2) ) ;
        depthFileList.push_back( string(t4) );
    }
    fclose(fp);
}

int readImageID = 0 ;

void readImage()
{
    if ( readImageID > rgbFileList.size() ){
        puts("End of Images") ;
        return ;
    }
    char tmp[256];

    //rgb image
    strcpy(tmp, filePath ) ;
    string rgbFileName = rgbFileList.front() ;
    rgbFileList.pop_front();
    strcat(tmp, rgbFileName.c_str() ) ;
    Mat rgbImage = imread(tmp, CV_LOAD_IMAGE_COLOR ) ;
    Mat grayImage ;
    cvtColor(rgbImage, grayImage, CV_RGB2GRAY);

    //depth image
    strcpy(tmp, filePath ) ;
    string depthFileName = depthFileList.front() ;
    depthFileList.pop_front();
    strcat(tmp, depthFileName.c_str() ) ;
    Mat dd = imread(tmp, CV_LOAD_IMAGE_ANYDEPTH ) ;
    Mat depthImage(calib_par.height, calib_par.width, CV_32F ) ;
    for ( int i = 0 ; i < calib_par.height; i++ )
    {
        for ( int j = 0 ; j < calib_par.width ; j++ )
        {
            depthImage.at<float>(i, j) = (float)dd.at<unsigned short>(i, j)/5000.0 ;
        }
    }

    ros::Time tImage = ros::Time::now() ;
//    imshow("img", grayImage ) ;
    globalLiveSLAM->image1Buf.push_back(ImageMeasurement(tImage, grayImage));
    globalLiveSLAM->image0Buf.push_back(ImageMeasurement(tImage, depthImage));

//    imshow("grayImage", grayImage ) ;
//    imshow("depthImage", depthImage/3.0 ) ;
//    waitKey(0) ;
    readImageID++ ;
}

int main( int argc, char** argv )
{
    XInitThreads();

    ros::init(argc, argv, "dense_fullba_rgbd");
    ros::NodeHandle nh ;

    LiveSLAMWrapper slamNode(packagePath, nh, calib_par );
    globalLiveSLAM = &slamNode ;

    initFileList() ;
    initPara() ;
    readImage() ;

//    dynamic_reconfigure::Server<dense_new::LSDParamsConfig> srv(ros::NodeHandle("~"));
//	srv.setCallback(dynConfCb);

//    string packagePath = ros::package::getPath("dense_new")+"/";
//    string caliFilePath = packagePath + "calib/LSD_calib.cfg" ;

//    if ( initCalibrationPar(caliFilePath) == false ){
//        return 0 ;
//    }

//    sub_image[0] = nh.subscribe("/cam0", 100, &image0CallBack );
//    sub_image[1] = nh.subscribe("/cam1", 100, &image1CallBack );
//    sub_imu = nh.subscribe("/cust_imu0", 1000, &imuCallBack ) ;

//    //Output3DWrapper* outputWrapper = new ROSOutput3DWrapper(calib_par.width, calib_par.height, nh);

//    globalLiveSLAM->popAndSetGravity();
//    boost::thread ptrProcessImageThread = boost::thread(&process_image);
//    boost::thread ptrProcessBAThread = boost::thread(&process_BA);

//    ros::spin() ;
//    ptrProcessImageThread.join();
//    ptrProcessBAThread.join();

	return 0;
}
