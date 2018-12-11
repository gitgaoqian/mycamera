#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
author:ros
created on 2018-11-15
同时实现ROS图像转换和图像识别服务。
"""
import sys
import requests
import urllib2
import json
import base64
import urllib
import cv2
import numpy as np
import time
import threading
import os
import rospy
from cv_bridge import CvBridge,CvBridgeError
from sensor_msgs.msg import Image
import signal

class ImageDetect:
    def __init__(self):
        signal.signal(signal.SIGINT, self.ExitFunction)  # 用于处理中断异常
        #图像识别相关
        self.api_key = 'cc4xT4PcIR3oaqiTqRBn2EM6'
        self.secret_key = 'eB4wzZbqa2MAS61Vdp6vU5LjRMygtw4i'
        self.auth_url = self.auth_url = "https://openapi.baidu.com/oauth/2.0/token?grant_type=client_credentials&client_id=" \
                                        + self.api_key + "&client_secret=" + self.secret_key
        self.advance_general_url = 'https://aip.baidubce.com/rest/2.0/image-classify/v2/advanced_general'
        self.token = self.GetToken()
        #语音相关
        self.tts_token = self.GetVoiceToken()
        self.Mac = "64:00:6A:69:07:E2"
        self.tts_url = "http://tsn.baidu.com/text2audio"
        #ROS节点相关
        self.bridge = CvBridge()
        rospy.Subscriber("/camera/left/image_raw", Image, self.callback)
        self.T = threading.Thread(target=self.AdvanceGeneral, args=())
        self.T.setDaemon(True)
    def callback(self,data):
        cv_image = self.bridge.imgmsg_to_cv2(data,"bgr8")
        cv2.imwrite("/home/ros/baidu_ai/frame.jpg", cv_image)
    def GetVoiceToken(self):
        apiKey = "KLB4LNxGRAiX58sBLpZOycEn"
        secretKey = "2xCMsnTurRFGTrRZg04UKwYqy9RNsyXK "
        auth_url = "https://openapi.baidu.com/oauth/2.0/token?grant_type=client_credentials&client_id=" \
                   + apiKey + "&client_secret=" + secretKey
        res = urllib2.urlopen(auth_url)
        json_data = res.read()
        return json.loads(json_data)['access_token']
    def GetToken(self):
        res = urllib2.urlopen(self.auth_url)
        data = res.read()
        return json.loads(data)['access_token']
    def AdvanceGeneral(self):#返回图像内容的百度百科内容，对于公众人物、果蔬、动物等有很高识别率，但是对于书籍、电影、电视剧不会识别出这是啥书，那是啥电影.每天有限定的访问次数：
        while True:
            f = open('/home/ros/baidu_ai/frame.jpg', 'rb')
            img = base64.b64encode(f.read())
            # str_image =  np.array2string(self.cv_image)
            params = {"image": img,"baike_num": 1, "access_token": self.token}
            params = urllib.urlencode(params)
            request = urllib2.Request(url=self.advance_general_url, data=params)
            request.add_header('Content-Type', 'application/x-www-form-urlencoded')
            response = urllib2.urlopen(request)
            content = response.read()
            content = eval(content)  # 将结果字符串转换为字典格式
            if "error_msg" in content.keys():#如果发生错误，输出错误信息
                print content
            else:#如果没有返回错误码
                print(str(content["result"][0]["score"])+str(content["result"][0]["keyword"]))
                self.TTS(content)
                time.sleep(0.3)
            f.close()
    def TTS(self,content):
        text = ""
        score = content["result"][0]["score"]
        if(score < 0.5):
            return 0
        else:
            keyword = content["result"][0]["keyword"]
            # description = content["result"][0]["baike_info"]["description"]
            text = keyword
            param_dic = {'tex': text, 'ctp': '1', 'lan': 'zh', 'cuid': self.Mac, 'tok': self.tts_token}
            r = requests.get(url=self.tts_url, params=param_dic, stream=True)
            voice_fp = open('/home/ros/tts.wav', 'wb')
            voice_fp.write(r.raw.read())
            voice_fp.close()
            os.system('mplayer /home/ros/tts.wav')
    def ExitFunction(self, signalnum, frame): # 发生中断异常
        print "exit program"
        sys.exit(0)
def main(args):
    ID = ImageDetect()
    ID.T.setDaemon(True)
    ID.T.start()
    rospy.init_node('image_detect', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()
if __name__ == "__main__":
    main(sys.argv)
