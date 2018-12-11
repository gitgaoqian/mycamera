#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <camera_info_manager/camera_info_manager.h>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
using namespace std;
using namespace cv;
 Mat image;
int main(int argc, char** argv)
{
  ros::init(argc, argv, "image_publisher");
  ros::NodeHandle nh;
  //camerpublishr类:http://docs.ros.org/api/image_transport/html/classimage__transport_1_1CameraPublisher.html#details
  image_transport::ImageTransport it(nh);
  image_transport::CameraPublisher pub = it.advertiseCamera("camera/image_raw", 1);
 //关于camerapublisher:既发布image又发布camera_info,camer_info主题名字和image主题在统一命名空间?
//image_transport::Publisher只发布image
  boost::shared_ptr<camera_info_manager::CameraInfoManager> cinfo_;
  
  // parameters
  std::string  camera_name_, camera_info_url_;
   //加载camerainfo
  
  nh.param("camera_name", camera_name_, std::string("head_camera"));
  nh.param("camera_info_url", camera_info_url_, std::string("file:///home/ros/yaml/head_camera.yaml"));
  cinfo_.reset(new camera_info_manager::CameraInfoManager(nh, camera_name_, camera_info_url_));

  VideoCapture capture(0);
  capture.set(CV_CAP_PROP_FRAME_WIDTH, 640);
  capture.set(CV_CAP_PROP_FRAME_HEIGHT, 480);

  ros::Rate loop_rate(5);
  while (nh.ok()) {

    capture>>image;
    //cv::imshow("Image",image);
    cv::waitKey(30);
    sensor_msgs::ImagePtr img = cv_bridge::CvImage(std_msgs::Header(), "bgr8", image).toImageMsg();
    sensor_msgs::CameraInfoPtr ci(new sensor_msgs::CameraInfo(cinfo_->getCameraInfo()));
    ci->header.frame_id = "camera_link";

    // publish the image
    pub.publish(img, ci);
    ros::spinOnce();
    loop_rate.sleep();
    
  }
}
