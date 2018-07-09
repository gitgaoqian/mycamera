#include <iostream>  
#include </usr/include/opencv/cv.h>  
#include </usr/include/opencv/cxcore.h>  
#include </usr/include/opencv/highgui.h>
#include <image_transport/image_transport.h>
#include <camera_info_manager/camera_info_manager.h>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
using namespace std;
using namespace cv;

//定义全局变量

Mat map1_x, map2_y;
Mat frame;
Mat Image_L;
Mat Image_R;

int main(int argc, char** argv)
{
  ros::init(argc, argv, "image_publisher");
  ros::NodeHandle nh;
  image_transport::ImageTransport it(nh);
  image_transport::CameraPublisher pub_L = it.advertiseCamera("/camera/left/image_raw", 1);
  image_transport::CameraPublisher pub_R = it.advertiseCamera("/camera/right/image_raw", 1);
  //
  boost::shared_ptr<camera_info_manager::CameraInfoManager> cinfo_left;
  boost::shared_ptr<camera_info_manager::CameraInfoManager> cinfo_right;

  // parameters
  std::string  camera_name_left, camera_info_url_left,camera_name_right, camera_info_url_right;
   //加载camerainfo
  
  nh.param("camera_name_left", camera_name_left, std::string("camera_L"));//定义左相机的名称
  nh.param("camera_info_url_left", camera_info_url_left, std::string("file:///home/ros/stereocalib/stereo_calib1/left.yaml"));//加载左相机的参数
  nh.param("camera_name_right", camera_name_right, std::string("camera_R"));
  nh.param("camera_info_url_right", camera_info_url_right, std::string("file:///home/ros/stereocalib/stereo_calib1/right.yaml"));
  
  cinfo_left.reset(new camera_info_manager::CameraInfoManager(nh, camera_name_left, camera_info_url_left));
  cinfo_right.reset(new camera_info_manager::CameraInfoManager(nh, camera_name_right, camera_info_url_right));

  VideoCapture capture(0); //读取摄像头
  capture.set(CV_CAP_PROP_FRAME_WIDTH, 640);
  capture.set(CV_CAP_PROP_FRAME_HEIGHT, 240);
 
  
  ros::Rate loop_rate(15);
  while (nh.ok()) {

    capture>>frame;
    
    Image_L = frame(Rect(0,0,320,240));
    Image_R = frame(Rect(320,0,320,240));

 
    cv::waitKey(30);
    sensor_msgs::ImagePtr img_L = cv_bridge::CvImage(std_msgs::Header(), "bgr8", Image_L).toImageMsg();
    sensor_msgs::ImagePtr img_R = cv_bridge::CvImage(std_msgs::Header(), "bgr8", Image_R).toImageMsg();
    sensor_msgs::CameraInfoPtr ci_left(new sensor_msgs::CameraInfo(cinfo_left->getCameraInfo()));
    sensor_msgs::CameraInfoPtr ci_right(new sensor_msgs::CameraInfo(cinfo_right->getCameraInfo()));
    ci_left->header.frame_id = "camera_link";
    ci_right->header.frame_id = "camera_link";
    pub_L.publish(img_L,ci_left);
    pub_R.publish(img_R,ci_right);
    
    
    ros::spinOnce();
    loop_rate.sleep();
  }
}



