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

#include "SlamSystem.h"
#include "util/settings.h"
#include "DataStructures/Frame.h"
#include "Tracking/SE3Tracker.h"
#include "Tracking/TrackingReference.h"
#include "LiveSLAMWrapper.h"
#include "util/globalFuncs.h"
#include "IOWrapper/ImageDisplay.h"
#include "DataStructures/FrameMemory.h"
#include <deque>

// for mkdir
#include <sys/types.h>
#include <sys/stat.h>
#include "opencv2/opencv.hpp"

using namespace lsd_slam;
using namespace Eigen;

SlamSystem::SlamSystem(int w, int h, Eigen::Matrix3f K, ros::NodeHandle &n)
{
//    if(w%16 != 0 || h%16!=0)
//    {
//        printf("image dimensions must be multiples of 16! Please crop your images / video accordingly.\n");
//        assert(false);
//    }

	this->width = w;
	this->height = h;
	this->K = K;
    this->nh = n ;
	trackingIsGood = true;
    lock_densetracking = false ;
    currentKeyFrame =  nullptr;
	createNewKeyFrame = false;

	tracker = new SE3Tracker(w,h,K);
    trackingReference = new TrackingReference();
    int maxDisparity = 64 ;
    int blockSize = 21 ;
    bm_ = cv::StereoBM( cv::StereoBM::BASIC_PRESET, maxDisparity, blockSize) ;
    initRosPub() ;

    head = 0 ;
    tail = -1 ;
    numOfState = 0 ;
    frameInfoListHead = frameInfoListTail = 0 ;

	lastTrackingClosenessScore = 0;
	msTrackFrame = msOptimizationIteration = msFindConstraintsItaration = msFindReferences = 0;
	nTrackFrame = nOptimizationIteration = nFindConstraintsItaration = nFindReferences = 0;
	nAvgTrackFrame = nAvgOptimizationIteration = nAvgFindConstraintsItaration = nAvgFindReferences = 0;
	gettimeofday(&lastHzUpdate, NULL);

}

SlamSystem::~SlamSystem()
{
	delete trackingReference;
	delete tracker;

    for( int i = 0 ; i < slidingWindowSize ; i++ ){
        slidingWindow[i].reset();
    }
    //lastTrackedFrame.reset();
    currentKeyFrame.reset();
	FrameMemory::getInstance().releaseBuffes();
	Util::closeAllWindows();
}

void SlamSystem::debugDisplayDepthMap()
{
//	double scale = 1;
//	if(currentKeyFrame != 0 && currentKeyFrame != 0)
//		scale = currentKeyFrame->getScaledCamToWorld().scale();
//	// debug plot depthmap
//	char buf1[200];

//    snprintf(buf1,200,"dens %2.0f%%; good %2.0f%%; scale %2.2f; res %2.1f/; usg %2.0f%%; Map: %d F, %d KF, %d E, %.1fm Pts",
//			100*currentKeyFrame->numPoints/(float)(width*height),
//			100*tracking_lastGoodPerBad,
//			scale,
//			tracking_lastResidual,
//            100*tracking_lastUsage );


//	if(onSceenInfoDisplay)
//        printMessageOnCVImage(map->debugImageDepth, buf1 );
//	if (displayDepthMap)
//		Util::displayImage( "DebugWindow DEPTH", map->debugImageDepth, false );

//	int pressedKey = Util::waitKey(1);
//	handleKey(pressedKey);
}

void SlamSystem::initRosPub()
{
    pub_path = nh.advertise<visualization_msgs::Marker>("/denseVO/path", 1000);
    pub_cloud = nh.advertise<sensor_msgs::PointCloud>("/denseVO/cloud", 1000);
    pub_odometry = nh.advertise<nav_msgs::Odometry>("/denseVO/odometry", 1000);
    pub_pose = nh.advertise<geometry_msgs::PoseStamped>("/denseVO/pose", 1000);
    pub_resudualMap = nh.advertise<sensor_msgs::Image>("denseVO/residualMap", 100 );
    pub_reprojectMap = nh.advertise<sensor_msgs::Image>("denseVO/reprojectMap", 100 );
    pub_gradientMapForDebug = nh.advertise<sensor_msgs::Image>("denseVO/debugMap", 100 );

    path_line.header.frame_id    = "world";
    path_line.header.stamp       = ros::Time::now();
    path_line.ns                 = "dense_vo";
    path_line.action             = visualization_msgs::Marker::ADD;
    path_line.pose.orientation.w = 1.0;
    path_line.type               = visualization_msgs::Marker::LINE_STRIP;
    path_line.scale.x            = 0.01 ;
    path_line.color.a            = 1.0;
    path_line.color.r            = 1.0;
    path_line.id                 = 1;
    path_line.points.push_back( geometry_msgs::Point());
    pub_path.publish(path_line);
}

void SlamSystem::generateDubugMap(Frame* currentFrame, cv::Mat& gradientMapForDebug )
{
    int n = currentFrame->height() ;
    int m = currentFrame->width() ;
    const float* pIdepth = currentFrame->idepth(0) ;
    for ( int i = 0 ; i < n ; i++ )
    {
        for( int j = 0 ; j < m ; j++ )
        {
            if (  *pIdepth > 0 ){
                gradientMapForDebug.at<cv::Vec3b>(i, j)[0] = 0;
                gradientMapForDebug.at<cv::Vec3b>(i, j)[1] = 255;
                gradientMapForDebug.at<cv::Vec3b>(i, j)[2] = 0;
            }
            pIdepth++ ;
        }
    }
}

void SlamSystem::setDepthInit(cv::Mat img0, cv::Mat img1, double timeStamp, int id)
{
    cv::Mat disparity, depth ;
    bm_(img1, img0, disparity, CV_32F);
    calculateDepthImage(disparity, depth, 0.11, K(0, 0) );

    currentKeyFrame.reset(new Frame(id, width, height, K, timeStamp, img1.data,
                                    Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero() ));
    Frame* frame = currentKeyFrame.get() ;
    frame->setDepthFromGroundTruth( (float*)depth.data );
    if ( printDebugInfo ){
        cv::cvtColor(img1, gradientMapForDebug, CV_GRAY2BGR ) ;
        generateDubugMap(frame, gradientMapForDebug ) ;
        sensor_msgs::Image msg;
        msg.header.stamp = ros::Time() ;
        sensor_msgs::fillImage(msg, sensor_msgs::image_encodings::BGR8, height, width, width*3,
                               gradientMapForDebug.data );
        pub_gradientMapForDebug.publish(msg) ;
    }
    frame->R_bk_2_b0.setIdentity() ;
    frame->T_bk_2_b0.setZero() ;
    frame->v_bk.setZero() ;
    RefToFrame = SE3();
    slidingWindow[tail] = currentKeyFrame ;
//    std::cout << RefToFrame.rotationMatrix() << std::endl ;
//    std::cout << RefToFrame.translation() << std::endl ;
}

void SlamSystem::copyStateData( int preStateID )
{
//  //copy the lastest data to the second lastest

//  //1. basic data
//  slidingWindow[preStateID]->R_bk_2_b0 = slidingWindow[tail]->R_bk_2_b0;
//  slidingWindow[preStateID]->T_bk_2_b0 = slidingWindow[tail]->T_bk_2_b0;
//  slidingWindow[preStateID]->v_bk = slidingWindow[tail]->v_bk;

//  slidingWindow[preStateID]->alpha_c_k = slidingWindow[tail]->alpha_c_k;
//  slidingWindow[preStateID]->beta_c_k = slidingWindow[tail]->beta_c_k;
//  slidingWindow[preStateID]->R_k1_k = slidingWindow[tail]->R_k1_k;
//  slidingWindow[preStateID]->P_k = slidingWindow[tail]->P_k;
//  slidingWindow[preStateID]->timeIntegral = slidingWindow[tail]->timeIntegral;

//  slidingWindow[preStateID]->keyFrameFlag = slidingWindow[tail]->keyFrameFlag;
//  slidingWindow[preStateID]->imuLinkFlag = slidingWindow[tail]->imuLinkFlag;
//  slidingWindow[preStateID]->timestamp = slidingWindow[tail]->timestamp;

//  int n = height;
//  int m = width;
//  for (int i = 0; i < maxPyramidLevel; i++)
//  {
//    memcpy(slidingWindow[preStateID]->intensity[i], slidingWindow[tail]->intensity[i], n*m*sizeof(unsigned char));
//    if (slidingWindow[tail]->keyFrameFlag )
//    {
//      memcpy(slidingWindow[preStateID]->depthImage[i], slidingWindow[tail]->depthImage[i], n*m*sizeof(float));
//      memcpy(slidingWindow[preStateID]->gradientX[i], slidingWindow[tail]->gradientX[i], n*m*sizeof(double));
//      memcpy(slidingWindow[preStateID]->gradientY[i], slidingWindow[tail]->gradientY[i], n*m*sizeof(double));

//      slidingWindow[preStateID]->totalNumOfValidPixels[i] = slidingWindow[tail]->totalNumOfValidPixels[i];

//      int sz = slidingWindow[tail]->pixelInfo[i].Aij.size();
//      slidingWindow[preStateID]->pixelInfo[i].Aij.clear();
//      slidingWindow[preStateID]->pixelInfo[i].Aij.resize(sz);
//      slidingWindow[preStateID]->pixelInfo[i].AijTAij.clear();
//      slidingWindow[preStateID]->pixelInfo[i].AijTAij.resize(sz);
//      for (int j = 0; j < sz; j++)
//      {
//        slidingWindow[preStateID]->pixelInfo[i].Aij[j] = slidingWindow[tail]->pixelInfo[i].Aij[j];
//        slidingWindow[preStateID]->pixelInfo[i].AijTAij[j] = slidingWindow[tail]->pixelInfo[i].AijTAij[j];
//      }
////        slidingWindow[preStateID]->pixelInfo[i].Aij.swap( slidingWindow[tail]->pixelInfo[i].Aij );
////        slidingWindow[preStateID]->pixelInfo[i].AijTAij.swap( slidingWindow[tail]->pixelInfo[i].AijTAij );
//      slidingWindow[preStateID]->pixelInfo[i].piList = slidingWindow[tail]->pixelInfo[i].piList ;
//      slidingWindow[preStateID]->pixelInfo[i].intensity = slidingWindow[tail]->pixelInfo[i].intensity ;
//      slidingWindow[preStateID]->pixelInfo[i].goodPixel = slidingWindow[tail]->pixelInfo[i].goodPixel ;
//    }
//    n >>= 1;
//    m >>= 1;
//  }

  //2. maintain reprojection list
  for (int i = numOfState - 2; i >= 0; i--)
  {
    int k = head + i;
    if (k >= slidingWindowSize){
      k -= slidingWindowSize;
    }
    //delete the old link
    list<int>::iterator iter;
    for (iter = slidingWindow[k]->cameraLinkList.begin(); iter != slidingWindow[k]->cameraLinkList.end();)
    {
      if (*iter == preStateID){
        iter = slidingWindow[k]->cameraLinkList.erase(iter);
      }
      else{
        iter++;
      }
    }

    //rebuild the new link
    for (iter = slidingWindow[k]->cameraLinkList.begin(); iter != slidingWindow[k]->cameraLinkList.end(); iter++)
    {
      if (*iter == tail)
      {
        slidingWindow[k]->cameraLink[preStateID].R_bi_2_bj = slidingWindow[k]->cameraLink[tail].R_bi_2_bj;
        slidingWindow[k]->cameraLink[preStateID].T_bi_2_bj = slidingWindow[k]->cameraLink[tail].T_bi_2_bj;
        slidingWindow[k]->cameraLink[preStateID].P_inv = slidingWindow[k]->cameraLink[tail].P_inv;
        *iter = preStateID;
      }
    }
  }

  numOfState--;
  //swap the pointer
  slidingWindow[preStateID] = slidingWindow[tail] ;
  tail = preStateID;

  preStateID = tail - 1;
  if (preStateID < 0){
    preStateID += slidingWindowSize;
  }
  slidingWindow[preStateID]->imuLinkFlag = false;
}


void SlamSystem::twoWayMarginalize()
{
    if (twoWayMarginalizatonFlag == false)
    {
      //marginalized the oldest frame
      if (numOfState == slidingWindowSize)
      {
        vector<Vector3d>T(slidingWindowSize);
        vector<Vector3d>vel(slidingWindowSize);
        vector<Matrix3d>R(slidingWindowSize);
        for (int i = 0; i < slidingWindowSize; i++)
        {
            if ( slidingWindow[i] != nullptr )
            {
                R[i] = slidingWindow[i]->R_bk_2_b0;
                T[i] = slidingWindow[i]->T_bk_2_b0;
                vel[i] = slidingWindow[i]->v_bk;
            }
        }

        margin.size++;
        //1. IMU constraints
        if ( slidingWindow[head]->imuLinkFlag )
        {
          int k = head ;
          int k1 = k + 1;
          if (k1 >= slidingWindowSize){
            k1 -= slidingWindowSize;
          }
          MatrixXd H_k1_2_k(9, 18);
          MatrixXd H_k1_2_k_T;
          VectorXd residualIMU(9);

          Vector3d t1 = R[k].transpose()*(T[k1] - T[k] + gravity_b0* SQ(slidingWindow[k]->timeIntegral)*0.5);
          Vector3d t2 = R[k].transpose()*(R[k1] * vel[k1] + gravity_b0* slidingWindow[k]->timeIntegral);

          H_k1_2_k.setZero();
          H_k1_2_k.block(0, 0, 3, 3) = -R[k].transpose();
          H_k1_2_k.block(0, 3, 3, 3) = -Matrix3d::Identity()*slidingWindow[k]->timeIntegral;
          H_k1_2_k.block(0, 6, 3, 3) = vectorToSkewMatrix(t1);
          H_k1_2_k.block(3, 3, 3, 3) = -Matrix3d::Identity();
          H_k1_2_k.block(3, 6, 3, 3) = vectorToSkewMatrix(t2);
          H_k1_2_k.block(6, 6, 3, 3) = -R[k1].transpose() * R[k];

          H_k1_2_k.block(0, 9, 3, 3) = R[k].transpose();
          H_k1_2_k.block(3, 12, 3, 3) = R[k].transpose() *  R[k1];
          H_k1_2_k.block(3, 15, 3, 3) = -R[k].transpose() *  R[k1] * vectorToSkewMatrix(vel[k1]);
          H_k1_2_k.block(6, 15, 3, 3) = Matrix3d::Identity();

          residualIMU.segment(0, 3) = t1 - vel[k] * slidingWindow[k]->timeIntegral - slidingWindow[k]->alpha_c_k;
          residualIMU.segment(3, 3) = t2 - vel[k] - slidingWindow[k]->beta_c_k;
          residualIMU.segment(6, 3) = 2.0 * (Quaterniond(slidingWindow[k]->R_k1_k.transpose()) * Quaterniond(R[k].transpose() *  R[k1])).vec();

          H_k1_2_k_T = H_k1_2_k.transpose();
          H_k1_2_k_T *= slidingWindow[k]->P_k.inverse();
          H_k1_2_k_T = H_k1_2_k.transpose();
          H_k1_2_k_T *= slidingWindow[k]->P_k.inverse();

          margin.Ap.block(0, 0, 18, 18) += H_k1_2_k_T  *  H_k1_2_k;
          margin.bp.segment(0, 18) += H_k1_2_k_T * residualIMU;
        }

        //2. camera constraints
        int currentStateID = head;
        MatrixXd H_i_2_j(9, 18);
        MatrixXd H_i_2_j_T;
        VectorXd residualCamera(9);
        MatrixXd tmpP_inv(9, 9);
        MatrixXd tmpHTH;
        VectorXd tmpHTb;
        list<int>::iterator iter = slidingWindow[currentStateID]->cameraLinkList.begin();
        for (; iter != slidingWindow[currentStateID]->cameraLinkList.end(); iter++)
        {
          int linkID = *iter;

          H_i_2_j.setZero();
          H_i_2_j.block(0, 0, 3, 3) = Matrix3d::Identity();
          H_i_2_j.block(6, 6, 3, 3) = Matrix3d::Identity();
          H_i_2_j.block(0, 9, 3, 3) = -Matrix3d::Identity();
          H_i_2_j.block(0, 15, 3, 3) = R[linkID] * vectorToSkewMatrix( slidingWindow[currentStateID]->cameraLink[linkID].T_bi_2_bj );
          H_i_2_j.block(6, 15, 3, 3) = -R[currentStateID].transpose() *  R[linkID];

          residualCamera.segment(0, 3) = T[currentStateID] - T[linkID] - R[linkID]* slidingWindow[currentStateID]->cameraLink[linkID].T_bi_2_bj;
          residualCamera.segment(3, 3).setZero();
          residualCamera.segment(6, 3) = 2.0 * (Quaterniond(slidingWindow[currentStateID]->cameraLink[linkID].R_bi_2_bj.transpose()) * Quaterniond(R[linkID].transpose()*R[currentStateID])).vec();

          tmpP_inv.setZero();
          tmpP_inv.block(0, 0, 3, 3) = slidingWindow[currentStateID]->cameraLink[linkID].P_inv.block(0, 0, 3, 3) ;
          tmpP_inv.block(0, 6, 3, 3) = slidingWindow[currentStateID]->cameraLink[linkID].P_inv.block(0, 3, 3, 3) ;
          tmpP_inv.block(6, 0, 3, 3) = slidingWindow[currentStateID]->cameraLink[linkID].P_inv.block(3, 0, 3, 3) ;
          tmpP_inv.block(6, 6, 3, 3) = slidingWindow[currentStateID]->cameraLink[linkID].P_inv.block(3, 3, 3, 3) ;
          double r_v = residualCamera.segment(0, 3).norm() ;
          if ( r_v > huber_r_v ){
            tmpP_inv /= r_v ;
          }
          double r_w = residualCamera.segment(6, 3).norm() ;
          if ( r_w > huber_r_w ){
            tmpP_inv /= r_w ;
          }


//          double r_v = residualCamera.segment(0, 3).norm() ;
//          if ( r_v > huber_r_v ){
//            tmpP_inv.block(0, 0, 3, 3) = v_cov_inv / r_v ;
//          }
//          else {
//            tmpP_inv.block(0, 0, 3, 3) = v_cov_inv ;
//          }
//          double r_w = residualCamera.segment(6, 3).norm() ;
//          if ( r_w > huber_r_w ){
//            tmpP_inv.block(6, 6, 3, 3) = w_cov_inv / r_w ;
//          }
//          else {
//            tmpP_inv.block(6, 6, 3, 3) = w_cov_inv ;
//          }

          H_i_2_j_T = H_i_2_j.transpose();
          H_i_2_j_T *= tmpP_inv;

          tmpHTH = H_i_2_j_T  *  H_i_2_j;
          tmpHTb = H_i_2_j_T * residualCamera;

          //                                        tmpHTH /= 9.0 ;
          //                                        tmpHTb /= 9.0 ;

          int currentStateIDIndex = 0 ;
          int linkIDIndex = linkID - head;
          if (linkIDIndex < 0){
            linkIDIndex += slidingWindowSize;
          }

          margin.Ap.block(currentStateIDIndex * 9, currentStateIDIndex * 9, 9, 9) += tmpHTH.block(0, 0, 9, 9);
          margin.Ap.block(currentStateIDIndex * 9, linkIDIndex * 9, 9, 9) += tmpHTH.block(0, 9, 9, 9);
          margin.Ap.block(linkIDIndex * 9, currentStateIDIndex * 9, 9, 9) += tmpHTH.block(9, 0, 9, 9);
          margin.Ap.block(linkIDIndex * 9, linkIDIndex * 9, 9, 9) += tmpHTH.block(9, 9, 9, 9);

          margin.bp.segment(currentStateIDIndex * 9, 9) += tmpHTb.segment(0, 9);
          margin.bp.segment(linkIDIndex * 9, 9) += tmpHTb.segment(9, 9);
        }

        //3. marginalization
        margin.popEndState();

        //pop the oldest state
        head++;
        if (head >= slidingWindowSize){
          head -= slidingWindowSize;
        }
        numOfState--;
      }
      else
      {
        margin.size++;
      }
    }
    else
    {
      //marginalized the second newest frame
      vector<Vector3d>T(slidingWindowSize);
      vector<Vector3d>vel(slidingWindowSize);
      vector<Matrix3d>R(slidingWindowSize);
      for (int i = 0; i < slidingWindowSize; i++)
      {
          if ( slidingWindow[i] != nullptr )
          {
              R[i] = slidingWindow[i]->R_bk_2_b0;
              T[i] = slidingWindow[i]->T_bk_2_b0;
              vel[i] = slidingWindow[i]->v_bk;
          }
      }

      margin.size++;
      int preStateID = tail - 1;
      if (preStateID < 0){
        preStateID += slidingWindowSize;
      }

      MatrixXd tmpHTH;
      VectorXd tmpHTb;

      MatrixXd H_k1_2_k(9, 18);
      MatrixXd H_k1_2_k_T;
      VectorXd residualIMU(9);

      MatrixXd H_i_2_j(9, 18);
      MatrixXd H_i_2_j_T;
      VectorXd residualCamera(9);
      MatrixXd tmpP_inv(9, 9);
      //1.  IMU constrains
      for (int i = numOfState - 2; i >= numOfState - 3; i--)
      {
        int k = head + i;
        if (k >= slidingWindowSize){
          k -= slidingWindowSize;
        }
        if (slidingWindow[k]->imuLinkFlag == false){
          continue;
        }
        int k1 = k + 1;
        if (k1 >= slidingWindowSize){
          k1 -= slidingWindowSize;
        }
        Vector3d t1 = R[k].transpose()*(T[k1] - T[k] + gravity_b0* SQ(slidingWindow[k]->timeIntegral)*0.5);
        Vector3d t2 = R[k].transpose()*(R[k1] * vel[k1] + gravity_b0* slidingWindow[k]->timeIntegral);

        H_k1_2_k.setZero();
        H_k1_2_k.block(0, 0, 3, 3) = -R[k].transpose();
        H_k1_2_k.block(0, 3, 3, 3) = -Matrix3d::Identity()*slidingWindow[k]->timeIntegral;
        H_k1_2_k.block(0, 6, 3, 3) = vectorToSkewMatrix(t1);
        H_k1_2_k.block(3, 3, 3, 3) = -Matrix3d::Identity();
        H_k1_2_k.block(3, 6, 3, 3) = vectorToSkewMatrix(t2);
        H_k1_2_k.block(6, 6, 3, 3) = -R[k1].transpose() * R[k];

        H_k1_2_k.block(0, 9, 3, 3) = R[k].transpose();
        H_k1_2_k.block(3, 12, 3, 3) = R[k].transpose() *  R[k1];
        H_k1_2_k.block(3, 15, 3, 3) = -R[k].transpose() *  R[k1] * vectorToSkewMatrix(vel[k1]);
        H_k1_2_k.block(6, 15, 3, 3) = Matrix3d::Identity();

        residualIMU.segment(0, 3) = t1 - vel[k] * slidingWindow[k]->timeIntegral - slidingWindow[k]->alpha_c_k;
        residualIMU.segment(3, 3) = t2 - vel[k] - slidingWindow[k]->beta_c_k;
        residualIMU.segment(6, 3) = 2.0 * (Quaterniond(slidingWindow[k]->R_k1_k.transpose()) * Quaterniond(R[k].transpose() *  R[k1])).vec();

        H_k1_2_k_T = H_k1_2_k.transpose();
        H_k1_2_k_T *= slidingWindow[k]->P_k.inverse();

        margin.Ap.block(i * 9, i * 9, 18, 18) += H_k1_2_k_T  *  H_k1_2_k;
        margin.bp.segment(i * 9, 18) += H_k1_2_k_T * residualIMU;
      }


      //2. camera constrains
      for (int i = numOfState-3; i >= 0; i-- )
      {
        int currentStateID = head + i;
        if (currentStateID >= slidingWindowSize){
          currentStateID -= slidingWindowSize;
        }
        if (slidingWindow[currentStateID]->keyFrameFlag == false){
          continue;
        }

        list<int>::iterator iter = slidingWindow[currentStateID]->cameraLinkList.begin();
        for (; iter != slidingWindow[currentStateID]->cameraLinkList.end(); iter++)
        {
          int linkID = *iter;

          if (linkID != preStateID){
            continue;
          }

          //H_i_2_j.setZero();
          //H_i_2_j.block(0, 0, 6, 6).setIdentity();
          //H_i_2_j.block(0, 6, 3, 3) = -Matrix3d::Identity();
          //H_i_2_j.block(3, 9, 3, 3) = -slidingWindow[currentStateID]->R_bk_2_b0.transpose() *  states[linkID].R_bk_2_b0;

          //residualCamera.segment(0, 3) = slidingWindow[currentStateID]->T_bk_2_b0 - states[linkID].T_bk_2_b0 - iter->T_bi_2_bj;
          //residualCamera.segment(3, 3) = 2.0 * (Quaterniond(iter->R_bi_2_bj) * Quaterniond(states[linkID].R_bk_2_b0.transpose() *  slidingWindow[currentStateID]->R_bk_2_b0)).vec();

          //H_i_2_j_T = H_i_2_j.transpose();
          //H_i_2_j_T *= iter->P_inv ;

          H_i_2_j.setZero();
          H_i_2_j.block(0, 0, 3, 3) = Matrix3d::Identity();
          H_i_2_j.block(6, 6, 3, 3) = Matrix3d::Identity();
          H_i_2_j.block(0, 9, 3, 3) = -Matrix3d::Identity();
          H_i_2_j.block(0, 15, 3, 3) = R[linkID] * vectorToSkewMatrix( slidingWindow[currentStateID]->cameraLink[linkID].T_bi_2_bj );
          H_i_2_j.block(6, 15, 3, 3) = -R[currentStateID].transpose() *  R[linkID];

          residualCamera.segment(0, 3) = T[currentStateID] - T[linkID] - R[linkID]*slidingWindow[currentStateID]->cameraLink[linkID].T_bi_2_bj;
          residualCamera.segment(3, 3).setZero();
          residualCamera.segment(6, 3) = 2.0 * (Quaterniond(slidingWindow[currentStateID]->cameraLink[linkID].R_bi_2_bj.transpose()) * Quaterniond(R[linkID].transpose()*R[currentStateID])).vec();

          tmpP_inv.setZero();
          tmpP_inv.block(0, 0, 3, 3) = slidingWindow[currentStateID]->cameraLink[linkID].P_inv.block(0, 0, 3, 3) ;
          tmpP_inv.block(0, 6, 3, 3) = slidingWindow[currentStateID]->cameraLink[linkID].P_inv.block(0, 3, 3, 3) ;
          tmpP_inv.block(6, 0, 3, 3) = slidingWindow[currentStateID]->cameraLink[linkID].P_inv.block(3, 0, 3, 3) ;
          tmpP_inv.block(6, 6, 3, 3) = slidingWindow[currentStateID]->cameraLink[linkID].P_inv.block(3, 3, 3, 3) ;
          double r_v = residualCamera.segment(0, 3).norm() ;
          if ( r_v > huber_r_v ){
            tmpP_inv /= r_v ;
          }
          double r_w = residualCamera.segment(6, 3).norm() ;
          if ( r_w > huber_r_w ){
            tmpP_inv /= r_w ;
          }


//          double r_v = residualCamera.segment(0, 3).norm() ;
//          if ( r_v > huber_r_v ){
//            tmpP_inv.block(0, 0, 3, 3) = v_cov_inv / r_v ;
//          }
//          else {
//            tmpP_inv.block(0, 0, 3, 3) = v_cov_inv ;
//          }
//          double r_w = residualCamera.segment(6, 3).norm() ;
//          if ( r_w > huber_r_w ){
//            tmpP_inv.block(6, 6, 3, 3) = w_cov_inv / r_w ;
//          }
//          else {
//            tmpP_inv.block(6, 6, 3, 3) = w_cov_inv ;
//          }

          H_i_2_j_T = H_i_2_j.transpose();
          H_i_2_j_T *= tmpP_inv;

          tmpHTH = H_i_2_j_T  *  H_i_2_j;
          tmpHTb = H_i_2_j_T * residualCamera;

          //                                        tmpHTH /= 9.0 ;
          //                                        tmpHTb /= 9.0 ;

          int currentStateIDIndex = currentStateID - head;
          if (currentStateIDIndex < 0){
            currentStateIDIndex += slidingWindowSize;
          }
          int linkIDIndex = linkID - head;
          if (linkIDIndex < 0){
            linkIDIndex += slidingWindowSize;
          }

          margin.Ap.block(currentStateIDIndex * 9, currentStateIDIndex * 9, 9, 9) += tmpHTH.block(0, 0, 9, 9);
          margin.Ap.block(currentStateIDIndex * 9, linkIDIndex * 9, 9, 9) += tmpHTH.block(0, 9, 9, 9);
          margin.Ap.block(linkIDIndex * 9, currentStateIDIndex * 9, 9, 9) += tmpHTH.block(9, 0, 9, 9);
          margin.Ap.block(linkIDIndex * 9, linkIDIndex * 9, 9, 9) += tmpHTH.block(9, 9, 9, 9);

          margin.bp.segment(currentStateIDIndex * 9, 9) += tmpHTb.segment(0, 9);
          margin.bp.segment(linkIDIndex * 9, 9) += tmpHTb.segment(9, 9);
        }
      }


      //3. marginalization
      margin.popFrontState();

      //double t = (double)cvGetTickCount();
      copyStateData( preStateID );
      //printf("copy time: %f\n", ((double)cvGetTickCount() - t) / (cvGetTickFrequency() * 1000));
    }
}

void SlamSystem::setNewMarginalzationFlag()
{
    if ( slidingWindow[tail]->keyFrameFlag ){
      twoWayMarginalizatonFlag = false;
    }
    else {
      twoWayMarginalizatonFlag = true;
    }
}

void SlamSystem::BA()
{
    vector<Vector3d>T(slidingWindowSize);
    vector<Vector3d>vel(slidingWindowSize);
    vector<Matrix3d>R(slidingWindowSize);
    for (int i = 0; i < slidingWindowSize; i++)
    {
        if ( slidingWindow[i] != nullptr )
        {
            R[i] = slidingWindow[i]->R_bk_2_b0;
            T[i] = slidingWindow[i]->T_bk_2_b0;
            vel[i] = slidingWindow[i]->v_bk;
        }
    }

    int sizeofH = 9 * numOfState;
    MatrixXd HTH(sizeofH, sizeofH);
    VectorXd HTb(sizeofH);
    MatrixXd tmpHTH;
    VectorXd tmpHTb;

    MatrixXd H_k1_2_k(9, 18);
    MatrixXd H_k1_2_k_T;
    VectorXd residualIMU(9);

    MatrixXd H_i_2_j(9, 18);
    MatrixXd H_i_2_j_T;
    VectorXd residualCamera(9);
    MatrixXd tmpP_inv(9, 9);

    for (int iterNum = 0; iterNum < maxIterationBA; iterNum++)
    {
      HTH.setZero();
      HTb.setZero();

      //1. prior constraints
      int m_sz = margin.size;
      VectorXd dx = VectorXd::Zero(STATE_SZ(numOfState - 1));

      if (m_sz != (numOfState - 1)){
        assert("prior matrix goes wrong!!!");
      }

      for (int i = numOfState - 2; i >= 0; i--)
      {
        int k = head + i;
        if (k >= slidingWindowSize){
          k -= slidingWindowSize;
        }
        //dp, dv, dq
        dx.segment(STATE_SZ(i), 3) = T[k] - slidingWindow[k]->T_bk_2_b0;
        dx.segment(STATE_SZ(i) + 3, 3) = vel[k] - slidingWindow[k]->v_bk;
        dx.segment(STATE_SZ(i) + 6, 3) = Quaterniond(Matrix3d(slidingWindow[k]->R_bk_2_b0.transpose() * R[k])).vec() * 2.0;
      }
      HTH.block(0, 0, STATE_SZ(m_sz), STATE_SZ(m_sz)) += margin.Ap.block(0, 0, STATE_SZ(m_sz), STATE_SZ(m_sz));
      HTb.segment(0, STATE_SZ(m_sz)) -= margin.Ap.block(0, 0, STATE_SZ(m_sz), STATE_SZ(m_sz))*dx;
      HTb.segment(0, STATE_SZ(m_sz)) -= margin.bp.segment(0, STATE_SZ(m_sz));

      //2. imu constraints
      for (int i = numOfState-2; i >= 0; i-- )
      {
        int k = head + i;
        if (k >= slidingWindowSize){
          k -= slidingWindowSize;
        }
        if ( slidingWindow[k]->imuLinkFlag == false){
          continue;
        }
        int k1 = k + 1;
        if (k1 >= slidingWindowSize){
          k1 -= slidingWindowSize;
        }
        Vector3d t1 = R[k].transpose()*( T[k1] - T[k] + gravity_b0* SQ( slidingWindow[k]->timeIntegral )*0.5);
        Vector3d t2 = R[k].transpose()*( R[k1] * vel[k1] + gravity_b0* slidingWindow[k]->timeIntegral );

        H_k1_2_k.setZero();
        H_k1_2_k.block(0, 0, 3, 3) = -R[k].transpose();
        H_k1_2_k.block(0, 3, 3, 3) = -Matrix3d::Identity()* slidingWindow[k]->timeIntegral;
        H_k1_2_k.block(0, 6, 3, 3) = vectorToSkewMatrix( t1 );
        H_k1_2_k.block(3, 3, 3, 3) = -Matrix3d::Identity();
        H_k1_2_k.block(3, 6, 3, 3) = vectorToSkewMatrix( t2 );
        H_k1_2_k.block(6, 6, 3, 3) = -R[k1].transpose() * R[k];

        H_k1_2_k.block(0, 9, 3, 3) = R[k].transpose();
        H_k1_2_k.block(3, 12, 3, 3) = R[k].transpose() *  R[k1];
        H_k1_2_k.block(3, 15, 3, 3) = -R[k].transpose() *  R[k1] * vectorToSkewMatrix( vel[k1] );
        H_k1_2_k.block(6, 15, 3, 3) = Matrix3d::Identity();

        residualIMU.segment(0, 3) = t1 - vel[k]* slidingWindow[k]->timeIntegral - slidingWindow[k]->alpha_c_k;
        residualIMU.segment(3, 3) = t2 - vel[k] - slidingWindow[k]->beta_c_k;
        residualIMU.segment(6, 3) = 2.0 * ( Quaterniond( slidingWindow[k]->R_k1_k.transpose() )
                                            * Quaterniond( R[k].transpose() *  R[k1] ) ).vec();
//        std::cout << "k\n" << k << std::endl ;
//        std::cout << "k1\n" << k1 << std::endl ;
//        std::cout << "gravity_b0\n" << gravity_b0 << std::endl ;
//        std::cout << "slidingWindow[k]->timeIntegral\n" << slidingWindow[k]->timeIntegral << std::endl ;

//        std::cout << "residualIMU\n" << residualIMU << std::endl ;
//        std::cout << "t2\n" << t2 << std::endl ;
//        std::cout << "vel[k] \n" << vel[k]  << std::endl ;
//        std::cout << "slidingWindow[k]->beta_c_k \n" << slidingWindow[k]->beta_c_k  << std::endl ;

         H_k1_2_k_T = H_k1_2_k.transpose();
        H_k1_2_k_T *= slidingWindow[k]->P_k.inverse();

        HTH.block(i * 9, i * 9, 18, 18) += H_k1_2_k_T  *  H_k1_2_k;
        HTb.segment(i * 9, 18) -= H_k1_2_k_T * residualIMU;
      }

      //3. camera constraints
//      int numList = 0;

      for (int i = 0; i < numOfState; i++)
      {
        int currentStateID = head + i;
        if (currentStateID >= slidingWindowSize){
          currentStateID -= slidingWindowSize;
        }
        if (slidingWindow[currentStateID]->keyFrameFlag == false){
          continue;
        }

        list<int>::iterator iter = slidingWindow[currentStateID]->cameraLinkList.begin();
        for (; iter != slidingWindow[currentStateID]->cameraLinkList.end(); iter++ )
        {
          int linkID = *iter;
          //numList++ ;

          //H_i_2_j.setZero();
          //H_i_2_j.block(0, 0, 6, 6).setIdentity();
          //H_i_2_j.block(0, 6, 3, 3) = -Matrix3d::Identity();
          //H_i_2_j.block(3, 9, 3, 3) = -slidingWindow[currentStateID]->R_bk_2_b0.transpose() *  slidingWindow[linkID]->R_bk_2_b0;

          //residualCamera.segment(0, 3) = slidingWindow[currentStateID]->T_bk_2_b0 - slidingWindow[linkID]->T_bk_2_b0 - iter->T_bi_2_bj;
          //residualCamera.segment(3, 3) = 2.0 * (Quaterniond(iter->R_bi_2_bj) * Quaterniond(slidingWindow[linkID]->R_bk_2_b0.transpose() *  slidingWindow[currentStateID]->R_bk_2_b0)).vec();

          //H_i_2_j_T = H_i_2_j.transpose();
          //H_i_2_j_T *= iter->P_inv ;

          H_i_2_j.setZero();
          H_i_2_j.block(0, 0, 3, 3) = Matrix3d::Identity();
          H_i_2_j.block(6, 6, 3, 3) = Matrix3d::Identity();
          H_i_2_j.block(0, 9, 3, 3) = -Matrix3d::Identity();
          H_i_2_j.block(0, 15, 3, 3) = R[linkID] * vectorToSkewMatrix( slidingWindow[currentStateID]->cameraLink[linkID].T_bi_2_bj );
          H_i_2_j.block(6, 15, 3, 3) = -R[currentStateID].transpose() *  R[linkID];

          residualCamera.segment(0, 3) = T[currentStateID] - T[linkID] - R[linkID]*slidingWindow[currentStateID]->cameraLink[linkID].T_bi_2_bj;
          residualCamera.segment(3, 3).setZero();
          residualCamera.segment(6, 3) = 2.0 * (Quaterniond(slidingWindow[currentStateID]->cameraLink[linkID].R_bi_2_bj.transpose()) * Quaterniond(R[linkID].transpose()*R[currentStateID])).vec();

          tmpP_inv.setZero();
          tmpP_inv.block(0, 0, 3, 3) = slidingWindow[currentStateID]->cameraLink[linkID].P_inv.block(0, 0, 3, 3) ;
          tmpP_inv.block(0, 6, 3, 3) = slidingWindow[currentStateID]->cameraLink[linkID].P_inv.block(0, 3, 3, 3) ;
          tmpP_inv.block(6, 0, 3, 3) = slidingWindow[currentStateID]->cameraLink[linkID].P_inv.block(3, 0, 3, 3) ;
          tmpP_inv.block(6, 6, 3, 3) = slidingWindow[currentStateID]->cameraLink[linkID].P_inv.block(3, 3, 3, 3) ;
          double r_v = residualCamera.segment(0, 3).norm() ;
          if ( r_v > huber_r_v ){
            tmpP_inv /= r_v ;
          }
          double r_w = residualCamera.segment(6, 3).norm() ;
          if ( r_w > huber_r_w ){
            tmpP_inv /= r_w ;
          }

//          double r_v = residualCamera.segment(0, 3).norm() ;
//          if ( r_v > huber_r_v ){
//            tmpP_inv.block(0, 0, 3, 3) = v_cov_inv / r_v ;
//          }
//          else {
//            tmpP_inv.block(0, 0, 3, 3) = v_cov_inv ;
//          }
//          double r_w = residualCamera.segment(6, 3).norm() ;
//          if ( r_w > huber_r_w ){
//            tmpP_inv.block(6, 6, 3, 3) = w_cov_inv / r_w ;
//          }
//          else {
//            tmpP_inv.block(6, 6, 3, 3) = w_cov_inv ;
//          }

          H_i_2_j_T = H_i_2_j.transpose();
          H_i_2_j_T *= tmpP_inv;

          tmpHTH = H_i_2_j_T  *  H_i_2_j;
          tmpHTb = H_i_2_j_T * residualCamera;

          int currentStateIDIndex = currentStateID - head;
          if ( currentStateIDIndex < 0){
            currentStateIDIndex += slidingWindowSize;
          }
          int linkIDIndex = linkID - head  ;
          if (linkIDIndex < 0){
            linkIDIndex += slidingWindowSize;
          }

          HTH.block(currentStateIDIndex * 9, currentStateIDIndex * 9, 9, 9) += tmpHTH.block(0, 0, 9, 9);
          HTH.block(currentStateIDIndex * 9, linkIDIndex * 9, 9, 9) += tmpHTH.block(0, 9, 9, 9);
          HTH.block(linkIDIndex * 9, currentStateIDIndex * 9, 9, 9) += tmpHTH.block(9, 0, 9, 9);
          HTH.block(linkIDIndex * 9, linkIDIndex * 9, 9, 9) += tmpHTH.block(9, 9, 9, 9);

          HTb.segment(currentStateIDIndex * 9, 9) -= tmpHTb.segment(0, 9);
          HTb.segment(linkIDIndex * 9, 9) -= tmpHTb.segment(9, 9);
        }
      }
//      printf("[numList in BA]=%d\n", numList ) ;

      //solve the BA
      //cout << "HTH\n" << HTH << endl;

      LLT<MatrixXd> lltOfHTH = HTH.llt();
      ComputationInfo info = lltOfHTH.info();
      if (info == Success)
      {
        VectorXd dx = lltOfHTH.solve(HTb);
 //       cout << iterNum << endl ;
 //       cout << dx.transpose() << endl ;

        //cout << "iteration " << iterNum << "\n" << dx << endl;
#ifdef DEBUG_INFO
        geometry_msgs::Vector3 to_pub ;
        to_pub.x = dx.norm() ;
        //printf("%d %f\n",iterNum, to_pub.x ) ;
        pub_BA.publish( to_pub ) ;
#endif

        VectorXd errorUpdate(9);
        for (int i = 0; i < numOfState; i++)
        {
          int k = head + i;
          if (k >= slidingWindowSize){
            k -= slidingWindowSize;
          }
          errorUpdate = dx.segment(i * 9, 9);
          T[k] += errorUpdate.segment(0, 3);
          vel[k] += errorUpdate.segment(3, 3);

          Quaterniond q(R[k]);
          Quaterniond dq;
          dq.x() = errorUpdate(6) * 0.5;
          dq.y() = errorUpdate(7) * 0.5;
          dq.z() = errorUpdate(8) * 0.5;
          dq.w() = sqrt(1 - SQ(dq.x()) * SQ(dq.y()) * SQ(dq.z()));
          R[k] = (q * dq).normalized().toRotationMatrix();
        }
        //cout << T[head].transpose() << endl;
      }
      else
      {
        ROS_WARN("LLT error!!!");
        iterNum = maxIterationBA;
        //cout << HTH << endl;
        //FullPivLU<MatrixXd> luHTH(HTH);
        //printf("rank = %d\n", luHTH.rank() ) ;
        //HTH.rank() ;
      }
    }

    // Include correction for information vector
    int m_sz = margin.size;
    VectorXd r0 = VectorXd::Zero(STATE_SZ(numOfState - 1));
    for (int i = numOfState - 2; i >= 0; i--)
    {
      int k = head + i;
      if (k >= slidingWindowSize){
        k -= slidingWindowSize;
      }
      //dp, dv, dq
      r0.segment(STATE_SZ(i), 3) = T[k] - slidingWindow[k]->T_bk_2_b0;
      r0.segment(STATE_SZ(i) + 3, 3) = vel[k] - slidingWindow[k]->v_bk;
      r0.segment(STATE_SZ(i) + 6, 3) = Quaterniond(Matrix3d(slidingWindow[k]->R_bk_2_b0.transpose() * R[k])).vec() * 2.0;
    }
    margin.bp.segment(0, STATE_SZ(m_sz)) += margin.Ap.block(0, 0, STATE_SZ(m_sz), STATE_SZ(m_sz))*r0;

    //after all the iterations done
    for (int i = 0; i < slidingWindowSize; i++)
    {
        if ( slidingWindow[i] != nullptr )
        {
            slidingWindow[i]->T_bk_2_b0 = T[i];
            slidingWindow[i]->v_bk = vel[i];
            slidingWindow[i]->R_bk_2_b0 = R[i];
        }
    }
}


void SlamSystem::insertFrame(int imageSeqNumber, cv::Mat img, ros::Time imageTimeStamp, Matrix3d R, Vector3d T, Vector3d vel )
{
    tail++ ;
    numOfState++;
    if (tail >= slidingWindowSize){
      tail -= slidingWindowSize;
    }
    slidingWindow[tail].reset(
                new Frame( tail, width, height, K, imageTimeStamp.toSec(), img.data, R, T, vel )
                );
}

void SlamSystem::insertCameraLink(Frame* keyFrame, Frame* currentFrame,
        const Matrix3d& R_k_2_c, const Vector3d& T_k_2_c, const MatrixXd& lastestATA )
{
    int id = currentFrame->id() ;
    keyFrame->cameraLinkList.push_back(id);
    keyFrame->cameraLink[id].R_bi_2_bj = R_k_2_c;
    keyFrame->cameraLink[id].T_bi_2_bj = T_k_2_c;
    keyFrame->cameraLink[id].P_inv = lastestATA;
  //keyFrame->cameraLink[currentFrame->id].T_trust = T_trust ;
}


void SlamSystem::processIMU(double dt, const Vector3d&linear_acceleration, const Vector3d &angular_velocity)
{
    Quaterniond dq;

     dq.x() = angular_velocity(0)*dt*0.5;
     dq.y() = angular_velocity(1)*dt*0.5;
     dq.z() = angular_velocity(2)*dt*0.5;
     dq.w() = sqrt(1 - SQ(dq.x()) * SQ(dq.y()) * SQ(dq.z()));

     Matrix3d deltaR(dq);
     //R_c_0 = R_c_0 * deltaR;
     //T_c_0 = ;
     Frame *current = slidingWindow[tail].get();

     Matrix<double, 9, 9> F = Matrix<double, 9, 9>::Zero();
     F.block<3, 3>(0, 3) = Matrix3d::Identity();
     F.block<3, 3>(3, 6) = -current->R_k1_k* vectorToSkewMatrix(linear_acceleration);
     F.block<3, 3>(6, 6) = -vectorToSkewMatrix(angular_velocity);

     Matrix<double, 6, 6> Q = Matrix<double, 6, 6>::Zero();
     Q.block<3, 3>(0, 0) = acc_cov;
     Q.block<3, 3>(3, 3) = gyr_cov;

     Matrix<double, 9, 6> G = Matrix<double, 9, 6>::Zero();
     G.block<3, 3>(3, 0) = -current->R_k1_k;
     G.block<3, 3>(6, 3) = -Matrix3d::Identity();

     current->P_k = (Matrix<double, 9, 9>::Identity() + dt * F) * current->P_k * (Matrix<double, 9, 9>::Identity() + dt * F).transpose() + (dt * G) * Q * (dt * G).transpose();
     //current->R_k1_k = current->R_k1_k*deltaR;
     current->alpha_c_k += current->beta_c_k*dt + current->R_k1_k*linear_acceleration * dt * dt * 0.5 ;
     current->beta_c_k += current->R_k1_k*linear_acceleration*dt;
     current->R_k1_k = current->R_k1_k*deltaR;
     current->timeIntegral += dt;
}

void SlamSystem::trackFrame(cv::Mat img1, unsigned int frameID,
                            ros::Time imageTimeStamp, Eigen::Matrix3d deltaR)
{
	// Create new frame
    std::shared_ptr<Frame> trackingNewFrame(
                new Frame( frameID, width, height, K, imageTimeStamp.toSec(), img1.data,
                           Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero() )
                );
    if (  trackingReference->keyframe != currentKeyFrame.get() ){
         trackingReference->importFrame( currentKeyFrame.get() );
    }

    //initial guess
    SE3 RefToFrame_initialEstimate ;
    RefToFrame_initialEstimate.setRotationMatrix(  deltaR.transpose()*RefToFrame.rotationMatrix() );
    RefToFrame_initialEstimate.translation() =
            deltaR.transpose()*RefToFrame.translation() ;

    //track
	struct timeval tv_start, tv_end;
	gettimeofday(&tv_start, NULL);
    RefToFrame = tracker->trackFrame( trackingReference, trackingNewFrame.get(),
                               RefToFrame_initialEstimate );
	gettimeofday(&tv_end, NULL);

//    Eigen::Matrix3d R_k_2_c = RefToFrame.rotationMatrix();
//    Eigen::Vector3d T_k_2_c = RefToFrame.translation();
//    Matrix3d R_bk1_2_b0 = trackingReference->keyframe->R_bk_2_b0 * R_k_2_c.transpose();
//    Vector3d T_bk1_2_b0 = trackingReference->keyframe->T_bk_2_b0 + R_bk1_2_b0*T_k_2_c ;
//    pubOdometry(-T_bk1_2_b0, R_bk1_2_b0, pub_odometry, pub_pose );
//    pubPath(-T_bk1_2_b0, path_line, pub_path );

    //debug information
    //msTrackFrame = 0.9*msTrackFrame + 0.1*((tv_end.tv_sec-tv_start.tv_sec)*1000.0f + (tv_end.tv_usec-tv_start.tv_usec)/1000.0f);
    msTrackFrame = (tv_end.tv_sec-tv_start.tv_sec)*1000.0f + (tv_end.tv_usec-tv_start.tv_usec)/1000.0f ;
    printf("msTrackFrame = %0.f\n", msTrackFrame ) ;
	nTrackFrame++;
	tracking_lastResidual = tracker->lastResidual;
	tracking_lastUsage = tracker->pointUsage;
	tracking_lastGoodPerBad = tracker->lastGoodCount / (tracker->lastGoodCount + tracker->lastBadCount);
	tracking_lastGoodPerTotal = tracker->lastGoodCount / (trackingNewFrame->width(SE3TRACKING_MIN_LEVEL)*trackingNewFrame->height(SE3TRACKING_MIN_LEVEL));

	// Keyframe selection
    if ( trackingReference->keyframe->numFramesTrackedOnThis > MIN_NUM_MAPPED )
	{
        Sophus::Vector3d dist = RefToFrame.translation() * currentKeyFrame->meanIdepth;
        float minVal = 1.0f;

        lastTrackingClosenessScore = getRefFrameScore(dist.dot(dist), tracker->pointUsage, KFDistWeight, KFUsageWeight);
        if (lastTrackingClosenessScore > minVal || tracker->trackingWasGood == false )
		{
			createNewKeyFrame = true;

           // if(enablePrintDebugInfo && printKeyframeSelectionInfo)
           //     printf("SELECT %d on %d! dist %.3f + usage %.3f = %.3f > 1\n",trackingNewFrame->id(),trackingNewFrame->getTrackingParent()->id(), dist.dot(dist), tracker->pointUsage, lastTrackingClosenessScore );
        }
		else
		{
        //	if(enablePrintDebugInfo && printKeyframeSelectionInfo)
        //       printf("SKIPPD %d on %d! dist %.3f + usage %.3f = %.3f < 1\n",trackingNewFrame->id(),trackingNewFrame->getTrackingParent()->id(), dist.dot(dist), tracker->pointUsage, lastTrackingClosenessScore );
		}
	}
    frameInfoList_mtx.lock();
    int tmpTail = frameInfoListTail+1 ;
    if ( tmpTail >= frameInfoListSize ){
        tmpTail -= frameInfoListSize;
    }
    FRAMEINFO& tmpFrameInfo = frameInfoList[tmpTail] ;
    tmpFrameInfo.t = imageTimeStamp ;
    tmpFrameInfo.R_k_2_c = RefToFrame.rotationMatrix().cast<double>();
    tmpFrameInfo.T_k_2_c = RefToFrame.translation().cast<double>();
    tmpFrameInfo.trust = true ;
    tmpFrameInfo.keyFrameFlag = createNewKeyFrame ;
    tmpFrameInfo.lastestATA = MatrixXd::Identity(6, 6)*1000000 ;
    frameInfoListTail = tmpTail ;
    frameInfoList_mtx.unlock();

    if ( createNewKeyFrame == true ){
        tracking_mtx.lock();
        lock_densetracking = true ;
        tracking_mtx.unlock();
    }
}

