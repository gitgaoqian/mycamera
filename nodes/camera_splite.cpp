#include <iostream>  
#include </usr/include/opencv/cv.h>  
#include </usr/include/opencv/cxcore.h>  
#include </usr/include/opencv/highgui.h>
#include <image_transport/image_transport.h>
#include <camera_info_manager/camera_info_manager.h>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <dynamic_reconfigure/server.h>
#include <mycamera/dynamic_rateConfig.h>
using namespace std;
using namespace cv;


//定义全局变量

Mat map1_x, map2_y;
Mat frame;
Mat Image_L;
Mat Image_R;

int src_rate;
//此处更改参数值并且生效的方式有两种:1 讲src_rate 放在循环语句中,那么可以通过rosparam set来改变 2 设置一个动态参数,然后将这个动态参数赋值src_rate那么也可以通过rqt滑动条进行改变
void ConfigCb(mycamera::dynamic_rateConfig &config, uint32_t level)
{
  src_rate = config.src_rate;
}
int main(int argc, char** argv)
{
  ros::init(argc, argv, "camera_splite");
  ros::NodeHandle nh("~");//为了之后能够获取ros私有参数

            /* 设置动态参数,这样更改参数可以通过rqt_config滑动滑动条就行*/
  dynamic_reconfigure::Server<mycamera::dynamic_rateConfig> server;
  dynamic_reconfigure::Server<mycamera::dynamic_rateConfig>::CallbackType f;

  f = boost::bind(&ConfigCb, _1, _2);
  server.setCallback(f);

  image_transport::ImageTransport it(nh);
  //创建左右相机图像信息和相机信息的发布者
  image_transport::CameraPublisher pub_L = it.advertiseCamera("/camera/left/image_raw", 1);
  image_transport::CameraPublisher pub_R = it.advertiseCamera("/camera/right/image_raw", 1);

  // parameters
  std::string  camera_name_left, camera_info_url_left,camera_name_right, camera_info_url_right;
   //获取与camerainfo相关的ros参数服务器中的参数的几种方式
// //第一种在节点句柄命名空间:和roslaunch的结合,roslaunch对参数进行设置和赋值,节点中获取参数值
//   nh.getParam("camera_name_left",camera_name_left);
//   nh.getParam("camera_name_right",camera_name_right);
//   nh.getParam("camera_info_url_left",camera_info_url_left);
//   nh.getParam("camera_info_url_right",camera_info_url_right);

  //第二种在节点句柄命名空间:节点中直接设置参数并且赋值
  nh.param("camera_name_left", camera_name_left, std::string("camera_L"));//定义左相机的名称
  nh.param("camera_info_url_left", camera_info_url_left, std::string("file:///home/ros/stereocalib/stereo_calib1/left.yaml"));//加载左相机的参数
  nh.param("camera_name_right", camera_name_right, std::string("camera_R"));
  nh.param("camera_info_url_right", camera_info_url_right, std::string("file:///home/ros/stereocalib/stereo_calib1/right.yaml"));


  //boost智能指针？不懂
  boost::shared_ptr<camera_info_manager::CameraInfoManager> cinfo_left;
  boost::shared_ptr<camera_info_manager::CameraInfoManager> cinfo_right;
  //重新绑定camera_info_manager::CameraInfoManager
  cinfo_left.reset(new camera_info_manager::CameraInfoManager(nh, camera_name_left, camera_info_url_left));
  cinfo_right.reset(new camera_info_manager::CameraInfoManager(nh, camera_name_right, camera_info_url_right));

//给图像加上相机坐标系
  std_msgs::Header header;

  VideoCapture capture(0); //读取摄像头
  capture.set(CV_CAP_PROP_FRAME_WIDTH, 640);
  capture.set(CV_CAP_PROP_FRAME_HEIGHT, 240);
 
  while (nh.ok()) {
    //获取launch中设置的参数的两种方式,第一种节点句柄获取私有参数
    //nh.getParam("src_rate",src_rate);
     //节点句柄带有默认值的
    //nh.param<int>("src_rate", src_rate, 15);
    //第二种节点命名空间直接获取
    ros::param::get("~src_rate",src_rate);//放到循环中,可以当rosparam set更改参数值,程序会跟着变化;但是由于上面程序进行了动态参数的设置,所以也可以不放在循环中,通过rqt_config滑动条改变
    //直接版带默认值的
    //ros::param::param<int>("~src_rate",src_rate,15);
   
    ros::Rate loop_rate(src_rate);//频率设定为相机帧数，1s　30张图片，之前设置的小所以会出现画面不流畅
    capture>>frame;
    //给每一个图像加上时间戳
    ros::Time time_stamp=ros::Time::now();  
    header.stamp=time_stamp;
    Image_L = frame(Rect(0,0,320,240));
    Image_R = frame(Rect(320,0,320,240));

    cv::waitKey(30);
    //转换opencv图像到ros图像消息
    sensor_msgs::ImagePtr img_L = cv_bridge::CvImage(header, "bgr8", Image_L).toImageMsg();
    sensor_msgs::ImagePtr img_R = cv_bridge::CvImage(header, "bgr8", Image_R).toImageMsg();
    sensor_msgs::CameraInfoPtr ci_left(new sensor_msgs::CameraInfo(cinfo_left->getCameraInfo()));
    sensor_msgs::CameraInfoPtr ci_right(new sensor_msgs::CameraInfo(cinfo_right->getCameraInfo()));
    //为camerainfo加上坐标系和时间戳
    ci_left->header.frame_id = "camera_link";
    ci_left->header.stamp=time_stamp;
    ci_right->header.frame_id = "camera_link";
    ci_right->header.stamp=time_stamp;
    //同时发布图像和相机信息
    pub_L.publish(img_L,ci_left);
    pub_R.publish(img_R,ci_right);
    
    
    ros::spinOnce();
    loop_rate.sleep();
  }
}