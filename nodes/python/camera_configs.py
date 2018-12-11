#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
reference:https://www.cnblogs.com/zhiyishou/p/5767592.html
author:ros  created on 2018-11-15
'''
import cv2
import numpy as np

left_camera_matrix = np.array([[213.437818, 0.000000, 172.803103],
                                [0.000000, 214.075437 ,127.027571],
                                [0.000000, 0.000000, 1.000000]])
left_distortion = np.array([[-0.374243, 0.118810 ,0.001474, 0.001239, 0.000000]])



right_camera_matrix = np.array([[210.689344, 0.000000, 168.144027],
[0.000000, 211.281242, 123.774837],
[0.000000, 0.000000, 1.000000]])
right_distortion = np.array([[-0.353463, 0.098363 ,0.002089, 0.003900, 0.000000]])

om = np.array([-0.00306, -0.03207, 0.00206]) # 旋转关系向量
R = cv2.Rodrigues(om)[0]  # 使用Rodrigues变换将om变换为R
T = np.array([-58.9993788834643,-0.166484432134018,1.80514541799934]) # 平移关系向量

size = (320, 240) # 图像尺寸

# 进行立体更正
R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(left_camera_matrix, left_distortion,
                                                                  right_camera_matrix, right_distortion, size, R,
                                                                  T)
# 计算更正map
left_map1, left_map2 = cv2.initUndistortRectifyMap(left_camera_matrix, left_distortion, R1, P1, size, cv2.CV_16SC2)
right_map1, right_map2 = cv2.initUndistortRectifyMap(right_camera_matrix, right_distortion, R2, P2, size, cv2.CV_16SC2)