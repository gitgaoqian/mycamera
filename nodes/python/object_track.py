#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" camshift_node.py - Version 1.1 2013-12-20
基于颜色追踪返回距离，图像直接传过来整图即可，在本程序中完成左右分割\图像校正\立体匹配等
1运行camera_read.launch 2 运行object_tracker.launch:republish节点和object_track节点
"""

import rospy
import cv2
from cv2 import cv as cv
from rbx1_vision.ros2opencv2 import ROS2OpenCV2

import numpy as np
import camera_configs
from std_msgs.msg import String
from sensor_msgs.msg import Image
import numpy as np

class CamShiftNode(ROS2OpenCV2):
    def __init__(self, node_name):
        ROS2OpenCV2.__init__(self, node_name)

        self.node_name = node_name

        # The minimum saturation of the tracked color in HSV space,
        # as well as the min and max value (the V in HSV) and a
        # threshold on the backprojection probability image.
        self.smin = rospy.get_param("~smin", 0)  # 最小饱和度
        self.vmin = rospy.get_param("~vmin", 0)  # 最小亮度
        self.vmax = rospy.get_param("~vmax", 254)  # 最大亮度
        self.threshold = rospy.get_param("~threshold", 50)  #

        # Create a number of windows for displaying the histogram,
        # parameters controls, and backprojection image
        cv.NamedWindow("Histogram", cv.CV_WINDOW_NORMAL)
        cv.MoveWindow("Histogram", 700, 50)
        cv.NamedWindow("Parameters", 0)
        cv.MoveWindow("Parameters", 700, 325)
        cv.NamedWindow("Backproject", 0)
        cv.MoveWindow("Backproject", 700, 600)

        # Create the slider controls for saturation, value and threshold添加滑动条
        cv.CreateTrackbar("Saturation", "Parameters", self.smin, 255, self.set_smin)
        cv.CreateTrackbar("Min Value", "Parameters", self.vmin, 255, self.set_vmin)
        cv.CreateTrackbar("Max Value", "Parameters", self.vmax, 255, self.set_vmax)
        cv.CreateTrackbar("Threshold", "Parameters", self.threshold, 255, self.set_threshold)

        # Initialize a number of variables
        self.hist = None
        self.track_window = None
        self.show_backproj = False
    # These are the callbacks for the slider controls
    def set_smin(self, pos):
        self.smin = pos
    def set_vmin(self, pos):
        self.vmin = pos
    def set_vmax(self, pos):
        self.vmax = pos
    def set_threshold(self, pos):
        self.threshold = pos
    # The main processing function computes the histogram and backprojection
    def process_image(self, cv_image):#重写ROS2OpenCv的process_image方法
        try:
            # First blur the image
            frame = cv2.blur(cv_image, (5, 5))  # 平均滤波处理，图像模糊处理

            # Convert from RGB to HSV space
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Create a mask using the current saturation and value parameters创建掩模
            # 这里初始的掩模设置整个图形全白，对于目标图像（所选区域）的掩模对应的hsv值，我们可以通过滑动条来测试得到
            mask = cv2.inRange(hsv, np.array((0., self.smin, self.vmin)), np.array((180., 255., self.vmax)))

            # If the user is making a selection with the mouse,
            # calculate a new histogram to track
            if self.selection is not None:
                x0, y0, w, h = self.selection  # 选中的区域，提供camshift算法中目标图像的初始区域
                x1 = x0 + w
                y1 = y0 + h
                self.track_window = (x0, y0, x1, y1)
                hsv_roi = hsv[y0:y1, x0:x1]
                mask_roi = mask[y0:y1, x0:x1]
                self.hist = cv2.calcHist([hsv_roi], [0], mask_roi, [16], [0, 180])  # 统计目标对象的颜色直方图
                cv2.normalize(self.hist, self.hist, 0, 255, cv2.NORM_MINMAX)  # 归一化
                self.hist = self.hist.reshape(-1)
                self.show_hist()

            if self.detect_box is not None:
                self.selection = None

            # If we have a histogram, track it with CamShift
            if self.hist is not None:
                # Compute the backprojection from the histogram
                backproject_pre = cv2.calcBackProject([hsv], [0], self.hist, [0, 180], 1)  # 目标图像的直方图在输入图像的反向投影,得到概率图形.

                # Mask the backprojection with the mask created earlier
                backproject = backproject_pre & mask  # 反向投影和掩模相与

                # Threshold the backprojection
                ret, backproject = cv2.threshold(backproject, self.threshold, 255, cv.CV_THRESH_TOZERO)

                x, y, w, h = self.track_window
                if self.track_window is None or w <= 0 or h <= 0:
                    self.track_window = 0, 0, self.frame_width - 1, self.frame_height - 1

                # Set the criteria for the CamShift algorithm
                term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

                # Run the CamShift algorithm
                self.track_box, self.track_window = cv2.CamShift(backproject, self.track_window, term_crit)
                # opencv标记物体检测的中心
                x, y, w, h = self.track_window
                cv2.circle(mask, (int(round(x + w / 2)), int(round(y + h / 2))), 3, (0, 0, 255), -1)
                d = self.DistanceMeasure(x+w/2,y+h/2)
                print(d)
                # cv2.putText(mask,"depth:"+str(d[2]),(x,y))
                # Display the resulting backprojection
                cv2.imshow("mask", mask)
                cv2.imshow("Backproject", backproject)
        except:
            pass

        return cv_image
    def show_hist(self):  # 显示直方图
        bin_count = self.hist.shape[0]
        bin_w = 24
        img = np.zeros((256, bin_count * bin_w, 3), np.uint8)
        for i in xrange(bin_count):
            h = int(self.hist[i])
            cv2.rectangle(img, (i * bin_w + 2, 255), ((i + 1) * bin_w - 2, 255 - h),
                          (int(180.0 * i / bin_count), 255, 255), -1)
        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        cv2.imshow('Histogram', img)
    def hue_histogram_as_image(self, hist):
        """ Returns a nice representation of a hue histogram """
        histimg_hsv = cv.CreateImage((320, 200), 8, 3)

        mybins = cv.CloneMatND(hist.bins)
        cv.Log(mybins, mybins)
        (_, hi, _, _) = cv.MinMaxLoc(mybins)
        cv.ConvertScale(mybins, mybins, 255. / hi)

        w, h = cv.GetSize(histimg_hsv)
        hdims = cv.GetDims(mybins)[0]
        for x in range(w):
            xh = (180 * x) / (w - 1)  # hue sweeps from 0-180 across the image
            val = int(mybins[int(hdims * x / w)] * h / 255)
            cv2.rectangle(histimg_hsv, (x, 0), (x, h - val), (xh, 255, 64), -1)
            cv2.rectangle(histimg_hsv, (x, h - val), (x, h), (xh, 255, 255), -1)

        histimg = cv2.cvtColor(histimg_hsv, cv.CV_HSV2BGR)

        return histimg
    def DistanceMeasure(self,x,y):
            # 根据更正map对图片进行重构
            img1_rectified = cv2.remap(self.left_image, camera_configs.left_map1, camera_configs.left_map2, cv2.INTER_LINEAR)
            img2_rectified = cv2.remap(self.right_image, camera_configs.right_map1, camera_configs.right_map2, cv2.INTER_LINEAR)

            # 将图片置为灰度图，为StereoBM作准备
            imgL = cv2.cvtColor(img1_rectified, cv2.COLOR_BGR2GRAY)
            imgR = cv2.cvtColor(img2_rectified, cv2.COLOR_BGR2GRAY)

            # 两个trackbar用来调节不同的参数查看效果
            num = cv2.getTrackbarPos("num", "depth")
            blockSize = cv2.getTrackbarPos("blockSize", "depth")
            if blockSize % 2 == 0:
                blockSize += 1
            if blockSize < 5:
                blockSize = 5

            # 根据Block Maching方法生成差异图（opencv里也提供了SGBM/Semi-Global Block Matching算法，有兴趣可以试试）
            stereo = cv2.StereoSGBM(minDisparity = 16,numDisparities = 128,SADWindowSize = 3,uniquenessRatio = 10,
                          speckleWindowSize = 100,speckleRange = 16,disp12MaxDiff = -1,P1 = 8*3*3**2,P2 = 32*3*3**2,fullDP = False)
            disparity = stereo.compute(imgL, imgR)
            disp = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            # 将图片扩展至3d空间中，其z方向的值则为当前的距离
            threeD = cv2.reprojectImageTo3D(disparity.astype(np.float32) / 16., camera_configs.Q)
            return threeD[x][y]

if __name__ == '__main__':
    try:
        node_name = "object_track"
        CamShiftNode(node_name)
        try:
            rospy.init_node(node_name)
        except:
            pass
        rospy.spin()
    except KeyboardInterrupt:
        print "Shutting down vision node."
        cv.DestroyAllWindows()


