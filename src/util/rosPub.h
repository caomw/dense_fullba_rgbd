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

#pragma once

#include "Eigen/Core"
#include "Eigen/Dense"
#include "Eigen/Geometry"
#include "ros/ros.h"
#include "tf/transform_broadcaster.h"
#include "nav_msgs/Odometry.h"
#include "geometry_msgs/PoseWithCovarianceStamped.h"
#include "geometry_msgs/PoseStamped.h"
#include "geometry_msgs/PoseArray.h"
#include "nav_msgs/Path.h"
#include "visualization_msgs/Marker.h"
#include "cv_bridge/cv_bridge.h"
#include "image_transport/image_transport.h"
#include "sensor_msgs/image_encodings.h"
#include "sensor_msgs/PointCloud.h"
#include "sensor_msgs/fill_image.h"

using namespace Eigen;

inline void pubOdometry(const Vector3d& p, const Matrix3d& R, ros::Publisher& pub_odometry, ros::Publisher& pub_pose )
{
    nav_msgs::Odometry odometry;
    Eigen::Quaterniond q(R) ;

    odometry.header.stamp = ros::Time::now();
    odometry.header.frame_id = "world";
    odometry.pose.pose.position.x = p(0);
    odometry.pose.pose.position.y = p(1);
    odometry.pose.pose.position.z = p(2);
    odometry.pose.pose.orientation.x = q.x();
    odometry.pose.pose.orientation.y = q.y();
    odometry.pose.pose.orientation.z = q.z();
    odometry.pose.pose.orientation.w = q.w();
    pub_odometry.publish(odometry);

    geometry_msgs::PoseStamped pose_stamped;
    pose_stamped.header.stamp = ros::Time::now();
    pose_stamped.header.frame_id = "world";
    pose_stamped.pose = odometry.pose.pose;
    pub_pose.publish(pose_stamped);
}

inline void pubPath(const Vector3d& p,
                    int kind,
                    visualization_msgs::Marker& path_line,
                    ros::Publisher& pub_path )
{
    geometry_msgs::Point pose_p;
    std_msgs::ColorRGBA color_p ;
    pose_p.x = p(0);
    pose_p.y = p(1);
    pose_p.z = p(2);
    if ( kind == 0 ){
        color_p.r = 1.0 ;
        color_p.g = 0.0 ;
        color_p.b = 0.0 ;
        color_p.a = 1.0 ;
    }
    else
    {
        color_p.r = 0.0 ;
        color_p.g = 1.0 ;
        color_p.b = 0.0 ;
        color_p.a = 1.0 ;
    }
    path_line.colors.push_back(color_p);
    path_line.points.push_back(pose_p);
    path_line.scale.x = 0.01 ;
    pub_path.publish(path_line);
}
