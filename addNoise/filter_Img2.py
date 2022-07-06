# -*- coding:utf-8 -*-
# @Time : 2022/6/8 8:05 下午
# @Author : Bin Bin Xue
# @File : add_Noise
# @Project : cv_works

# 利用opencv设置五种滤波：
# 线性滤波：方框滤波、均值滤波、高斯滤波
# 非线性滤波：中值滤波、双边滤波
# https://blog.csdn.net/Mr_Nobody17/article/details/119955506

# 导入opencv库
from cv2 import cv2
# 导入图像处理算法模块
import skimage
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']


# 1.均值滤波
def mean_filter(src, var1=3,var2=3):
    ksize = (var1,var2)
    dst = cv2.blur(src, ksize=ksize)
    cv2.namedWindow('mean_filter', cv2.WINDOW_NORMAL)
    cv2.imshow('mean_filter', dst)
    cv2.waitKey(0)
    return dst


# 2.方框滤波
def box_filter(src, var1,var2,var3,var4):
    ksize =(var2,var3)
    dst = cv2.boxFilter(src, ddepth=var1, ksize=ksize, normalize=var4)
    cv2.namedWindow('box_filter', cv2.WINDOW_NORMAL)
    cv2.imshow('box_filter', dst)
    cv2.waitKey(0)
    return dst


# 3.高斯滤波
def gaussian_filter(src, var1=3,var2=3, var3=0, var4=0):
    ksize=(var1,var2)
    dst = cv2.GaussianBlur(src, ksize=ksize, sigmaX=var3, sigmaY=var4)
    cv2.namedWindow('gaussian_filter', cv2.WINDOW_NORMAL)
    cv2.imshow('gaussian_filter', dst)
    cv2.waitKey(0)
    return dst


# 4.中值滤波
def median_filter(src, ksize=3):
    dst = cv2.medianBlur(src, ksize=int(ksize))
    cv2.namedWindow('median_filter', cv2.WINDOW_NORMAL)
    cv2.imshow('median_filter', dst)
    cv2.waitKey(0)
    return dst


# 5.双边滤波
def bilateralFilter(src, d=50, sigmaColor=100, sigmaSpace=100):
    dst = cv2.bilateralFilter(src, d=(d), sigmaColor=int(sigmaColor), sigmaSpace=int(sigmaSpace))
    cv2.namedWindow('bilateral_filter', cv2.WINDOW_NORMAL)
    cv2.imshow('bilasteral_filter', dst)
    cv2.waitKey(0)
    return dst
