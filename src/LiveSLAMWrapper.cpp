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
#include <vector>
#include <list>
#include <iostream>
#include "util/SophusUtil.h"
#include "util/globalFuncs.h"
#include "SlamSystem.h"
#include "IOWrapper/ImageDisplay.h"
#include "cv_bridge/cv_bridge.h"


namespace lsd_slam
{


LiveSLAMWrapper::LiveSLAMWrapper(std::string packagePath, ros::NodeHandle& _nh, const CALIBRATION_PAR &calib_par)
{
    fx = calib_par.fx;
    fy = calib_par.fy;
    cx = calib_par.cx;
    cy = calib_par.cy;
    width = calib_par.width;
    height = calib_par.height;
    nh = _nh ;

    isInitialized = false;
    Sophus::Matrix3f K_sophus;
    K_sophus << fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0;

	outFileName = packagePath+"estimated_poses.txt";
    outFile = nullptr;

	// make Odometry
    monoOdometry = new SlamSystem(width, height, K_sophus, _nh);

	imageSeqNumber = 0;
    image0Buf.clear();
    image1Buf.clear();
    imuQueue.clear();
}


LiveSLAMWrapper::~LiveSLAMWrapper()
{
	if(monoOdometry != 0)
		delete monoOdometry;
	if(outFile != 0)
	{
		outFile->flush();
		outFile->close();
		delete outFile;
	}
    image0Buf.clear();
    image1Buf.clear();
    imuQueue.clear();
}

void LiveSLAMWrapper::popAndSetGravity()
{
    std::list<ImageMeasurement>::reverse_iterator reverse_iterImage ;
    ros::Time tImage ;

    pImage0Iter = image0Buf.begin();
    pImage1Iter = image1Buf.begin();
    cv::Mat image0 = pImage0Iter->image.clone();
    cv::Mat image1 = pImage1Iter->image.clone();
    image0Buf.pop_front();
    image1Buf.pop_front();
    monoOdometry->gravity_b0.setZero() ;


    monoOdometry->insertFrame(imageSeqNumber, image1, tImage,
                              Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero() );

    monoOdometry->currentKeyFrame = monoOdometry->slidingWindow[0] ;
    monoOdometry->currentKeyFrame->setDepthFromGroundTruth( (float*)image0.data ) ;

    monoOdometry->currentKeyFrame->keyFrameFlag = true ;
    monoOdometry->currentKeyFrame->cameraLinkList.clear() ;
    monoOdometry->RefToFrame = Sophus::SE3() ;

    monoOdometry->margin.initPrior();
}


void LiveSLAMWrapper::BALoop()
{
    ros::Rate BARate(2000) ;
    list<ImageMeasurement>::iterator iterImage ;
    std::list<visensor_node::visensor_imu>::iterator iterIMU ;
    cv::Mat image0 ;
    cv::Mat image1 ;
    cv::Mat gradientMapForDebug(height, width, CV_8UC3) ;
    sensor_msgs::Image msg;
    double t ;

    while ( nh.ok() )
    {
        monoOdometry->frameInfoList_mtx.lock();
        int ttt = (monoOdometry->frameInfoListTail - monoOdometry->frameInfoListHead);
        if ( ttt < 0 ){
            ttt += frameInfoListSize ;
        }
        //printf("[BA thread] sz=%d\n", ttt ) ;
        if ( ttt < 1 ){
            monoOdometry->frameInfoList_mtx.unlock();
            BARate.sleep() ;
            continue ;
        }
        for ( int sz ; ; )
        {
            monoOdometry->frameInfoListHead++ ;
            if ( monoOdometry->frameInfoListHead >= frameInfoListSize ){
                monoOdometry->frameInfoListHead -= frameInfoListSize ;
            }
            sz = monoOdometry->frameInfoListTail - monoOdometry->frameInfoListHead ;
            if ( sz == 0 ){
                break ;
            }
            if ( monoOdometry->frameInfoList[monoOdometry->frameInfoListHead].keyFrameFlag ){
                break ;
            }
        }
        ros::Time imageTimeStamp = monoOdometry->frameInfoList[monoOdometry->frameInfoListHead].t ;
        monoOdometry->frameInfoList_mtx.unlock();

        //Pop out the image list
        image1_queue_mtx.lock();
        iterImage = image1Buf.begin() ;
        while ( iterImage->t < imageTimeStamp ){
            iterImage = image1Buf.erase( iterImage ) ;
        }
        image1 = iterImage->image.clone();
        image1_queue_mtx.unlock();

        image0_queue_mtx.lock();
        iterImage = image0Buf.begin() ;
        while ( iterImage->t < imageTimeStamp ){
            iterImage = image0Buf.erase( iterImage ) ;
        }
        image0 = iterImage->image.clone();
        image0_queue_mtx.unlock();

        imu_queue_mtx.lock();
        iterIMU = imuQueue.begin() ;
        Vector3d linear_acceleration;
        Vector3d angular_velocity;

        //std::cout << "imageTime=" << imageTimeStamp << std::endl;
        while ( iterIMU->header.stamp < imageTimeStamp )
        {
            linear_acceleration(0) = iterIMU->linear_acceleration.x;
            linear_acceleration(1) = iterIMU->linear_acceleration.y;
            linear_acceleration(2) = iterIMU->linear_acceleration.z;
            angular_velocity(0) = iterIMU->angular_velocity.x;
            angular_velocity(1) = iterIMU->angular_velocity.y;
            angular_velocity(2) = iterIMU->angular_velocity.z;

            //linear_acceleration = -linear_acceleration;
            //angular_velocity = -angular_velocity ;

//            double pre_t = iterIMU->header.stamp.toSec();
//            iterIMU = imuQueue.erase(iterIMU);

//            //std::cout << imuQueue.size() <<" "<< iterIMU->header.stamp << std::endl;

//            double next_t = iterIMU->header.stamp.toSec();
//            double dt = next_t - pre_t ;

            double pre_t = iterIMU->header.stamp.toSec();
            iterIMU = imuQueue.erase(iterIMU);
            double next_t = iterIMU->header.stamp.toSec();
            double dt = next_t - pre_t ;

//            std::cout << linear_acceleration.transpose() << std::endl ;
//            std::cout << angular_velocity.transpose() << std::endl ;
            monoOdometry->processIMU( dt, linear_acceleration, angular_velocity );
        }
        imu_queue_mtx.unlock();

        //propagate the last frame info to the current frame
        Frame* lastFrame = monoOdometry->slidingWindow[monoOdometry->tail].get();
        float dt = lastFrame->timeIntegral;

        Vector3d T_bk1_2_b0 = lastFrame->T_bk_2_b0 - 0.5 * gravity_b0 * dt *dt
                + lastFrame->R_bk_2_b0*(lastFrame->v_bk * dt  + lastFrame->alpha_c_k);
        Vector3d v_bk1 = lastFrame->R_k1_k.transpose() *
                (lastFrame->v_bk - lastFrame->R_bk_2_b0.transpose() * gravity_b0 * dt
                 + lastFrame->beta_c_k);
        Matrix3d R_bk1_2_b0 = lastFrame->R_bk_2_b0 * lastFrame->R_k1_k;

        monoOdometry->insertFrame(imageSeqNumber, image1, imageTimeStamp, R_bk1_2_b0, T_bk1_2_b0, v_bk1);
        Frame* currentFrame = monoOdometry->slidingWindow[monoOdometry->tail].get();
        Frame* keyFrame = monoOdometry->currentKeyFrame.get();
        if ( monoOdometry->frameInfoList[monoOdometry->frameInfoListHead].keyFrameFlag )
        {
            //prepare key frame
            cv::Mat disparity, depth ;
            monoOdometry->bm_(image1, image0, disparity, CV_32F);
            calculateDepthImage(disparity, depth, 0.11, fx );
            currentFrame->setDepthFromGroundTruth( (float*)depth.data ) ;

            //pub debugMap
            cv::cvtColor(image1, gradientMapForDebug, CV_GRAY2BGR);
            monoOdometry->generateDubugMap(currentFrame, gradientMapForDebug);
            msg.header.stamp = imageTimeStamp;
            sensor_msgs::fillImage(msg, sensor_msgs::image_encodings::BGR8, height,
                                   width, width*3, gradientMapForDebug.data );
            monoOdometry->pub_gradientMapForDebug.publish(msg) ;

            //set key frame
            monoOdometry->currentKeyFrame = monoOdometry->slidingWindow[monoOdometry->tail] ;
            monoOdometry->currentKeyFrame->keyFrameFlag = true ;
            monoOdometry->currentKeyFrame->cameraLinkList.clear() ;
            //reset the initial guess
            monoOdometry->RefToFrame = Sophus::SE3() ;

            //unlock dense tracking
            monoOdometry->tracking_mtx.lock();
            monoOdometry->lock_densetracking = false;
            monoOdometry->tracking_mtx.unlock();
        }
        if ( monoOdometry->frameInfoList[monoOdometry->frameInfoListHead].trust )
        {
//            cout << "insert camera link" << endl ;
//            cout << monoOdometry->frameInfoList[monoOdometry->frameInfoListHead].T_k_2_c << endl ;

            monoOdometry->insertCameraLink(keyFrame, currentFrame,
                          monoOdometry->frameInfoList[monoOdometry->frameInfoListHead].R_k_2_c,
                          monoOdometry->frameInfoList[monoOdometry->frameInfoListHead].T_k_2_c,
                          monoOdometry->frameInfoList[monoOdometry->frameInfoListHead].lastestATA );
        }

        cout << "[-BA]current Position: " << currentFrame->T_bk_2_b0.transpose() << endl;
        cout << "[-BA]current Velocity: " << currentFrame->v_bk.transpose() << endl;

        //BA
        t = (double)cvGetTickCount()  ;
        monoOdometry->BA();
        printf("BA cost time: %f\n", ((double)cvGetTickCount() - t) / (cvGetTickFrequency() * 1000) );
        t = (double)cvGetTickCount()  ;

        cout << "[BA-]current Position: " << currentFrame->T_bk_2_b0.transpose() << endl;
        cout << "[BA-]current Velocity: " << currentFrame->v_bk.transpose() << endl;

        //marginalziation
        monoOdometry->twoWayMarginalize();
        monoOdometry->setNewMarginalzationFlag();

        //    pubOdometry(-T_bk1_2_b0, R_bk1_2_b0, pub_odometry, pub_pose );
        //    pubPath(-T_bk1_2_b0, path_line, pub_path );

        pubOdometry(-monoOdometry->slidingWindow[monoOdometry->tail]->T_bk_2_b0,
                monoOdometry->slidingWindow[monoOdometry->tail]->R_bk_2_b0,
                monoOdometry->pub_odometry, monoOdometry->pub_pose );
        pubPath(-monoOdometry->slidingWindow[monoOdometry->tail]->T_bk_2_b0,
                monoOdometry->frameInfoList[monoOdometry->frameInfoListHead].keyFrameFlag,
                monoOdometry->path_line, monoOdometry->pub_path);
    }
}

void LiveSLAMWrapper::Loop()
{
    /*
    unsigned int image0BufSize ;
    unsigned int image1BufSize ;
    unsigned int imuBufSize ;
    */
    std::list<visensor_node::visensor_imu>::reverse_iterator reverse_iterImu ;
    std::list<ImageMeasurement>::iterator  pIter ;
    ros::Time imageTimeStamp ;
    cv::Mat   image0 ;
    cv::Mat   image1 ;
    ros::Rate r(1000.0);
    while ( nh.ok() )
    {
        monoOdometry->tracking_mtx.lock();
        bool tmpFlag = monoOdometry->lock_densetracking ;
        monoOdometry->tracking_mtx.unlock();
        //printf("tmpFlag = %d\n", tmpFlag ) ;
        if ( tmpFlag == true ){
            r.sleep() ;
            continue ;
        }
        image0_queue_mtx.lock();
        image1_queue_mtx.lock();
        imu_queue_mtx.lock();
        pIter = pImage1Iter ;
        pIter++ ;
        if ( pIter == image1Buf.end() ){
            image0_queue_mtx.unlock();
            image1_queue_mtx.unlock();
            imu_queue_mtx.unlock();
            r.sleep() ;
            continue ;
        }
        pIter = pImage0Iter ;
        pIter++ ;
        if ( pIter == image0Buf.end() ){
            image0_queue_mtx.unlock();
            image1_queue_mtx.unlock();
            imu_queue_mtx.unlock();
            r.sleep() ;
            continue ;
        }
        imageTimeStamp = pIter->t ;
        reverse_iterImu = imuQueue.rbegin() ;
//        printf("%d %d\n", imuQueue.size() < 10, reverse_iterImu->header.stamp <= imageTimeStamp ) ;
        if ( imuQueue.size() < 1 || reverse_iterImu->header.stamp < imageTimeStamp ){
            image0_queue_mtx.unlock();
            image1_queue_mtx.unlock();
            imu_queue_mtx.unlock();
            r.sleep() ;
            continue ;
        }
        //std::cout << imageTimeStamp.toNSec() << "\n" ;
        //std::cout << "[dt-image] " << imageTimeStamp << std::endl ;
        //std::cout << "[dt-imu] " << reverse_iterImu->header.stamp << " " << imuQueue.size() << std::endl ;
        ros::Time preTime = pImage1Iter->t ;
        pImage1Iter++ ;
        pImage0Iter++ ;

        imu_queue_mtx.unlock();
        image1 = pImage1Iter->image.clone();
        image0 = pImage0Iter->image.clone();
        image1_queue_mtx.unlock();
        image0_queue_mtx.unlock();

        imu_queue_mtx.lock();
        Quaterniond q, dq ;
        q.setIdentity() ;
        while ( currentIMU_iter->header.stamp < imageTimeStamp )
        {
            double pre_t = currentIMU_iter->header.stamp.toSec();
            currentIMU_iter++ ;
            double next_t = currentIMU_iter->header.stamp.toSec();
            double dt = next_t - pre_t ;

            //prediction for dense tracking
            dq.x() = currentIMU_iter->angular_velocity.x*dt*0.5 ;
            dq.y() = currentIMU_iter->angular_velocity.y*dt*0.5 ;
            dq.z() = currentIMU_iter->angular_velocity.z*dt*0.5 ;
            dq.w() =  sqrt( 1 - SQ(dq.x()) * SQ(dq.y()) * SQ(dq.z()) ) ;
            q = (q * dq).normalized();
        }
        imu_queue_mtx.unlock();

		// process image
		//Util::displayImage("MyVideo", image.data);
        Matrix3d deltaR(q) ;

        //puts("444") ;


        ++imageSeqNumber;
        assert(image0.elemSize() == 1);
        assert(image1.elemSize() == 1);
        assert(fx != 0 || fy != 0);

        monoOdometry->trackFrame(image1, imageSeqNumber, imageTimeStamp, deltaR );
	}
}

void LiveSLAMWrapper::logCameraPose(const SE3& camToWorld, double time)
{
    Sophus::Quaterniond quat = camToWorld.unit_quaternion();
    Eigen::Vector3d trans = camToWorld.translation();

	char buffer[1000];
	int num = snprintf(buffer, 1000, "%f %f %f %f %f %f %f %f\n",
			time,
			trans[0],
			trans[1],
			trans[2],
			quat.x(),
			quat.y(),
			quat.z(),
			quat.w());

	if(outFile == 0)
		outFile = new std::ofstream(outFileName.c_str());
	outFile->write(buffer,num);
	outFile->flush();
}

}
