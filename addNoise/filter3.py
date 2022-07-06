# -*- coding:utf-8 -*-
# @Time : 2022/6/15 5:10 下午
# @Author : Bin Bin Xue
# @File : filter3
# @Project : cv_works
import os

from cv2 import cv2

type = '.jpeg'

savePos = 'result2/'
readPos = 'result/'

name1 = 'nature_0_Gauss'
name2 = 'nature_0_P&S'
name3 = 'nature_0_Gamma'
name4 = 'nature_0_Uniform'

path1 = readPos + name1 + type
path2 = readPos + name2 + type
path3 = readPos + name3 + type
path4 = readPos + name4 + type

# 当前读取的图片名
list = [name1,name2,name3,name4]
# 当前读取的图片路径
# path = readPos + temp + type


if __name__ == '__main__':
    for temp in list:
        path = readPos + temp + type
        img = cv2.imread(path)
        # cv2.namedWindow('ori', cv2.WINDOW_NORMAL)
        # cv2.imshow('ori', img)
        # cv2.waitKey(0)
        for i in range(3, 100, 10):
            # 高斯滤波  sigmax=0会根据ksize的值自动进行计算，一般只要调核大小
            dst = cv2.GaussianBlur(img, (i, i), 0)
            cv2.namedWindow('gaussian_filter_' + str(i), cv2.WINDOW_NORMAL)
            cv2.imshow('gaussian_filter_' + str(i), dst)
            cv2.imwrite(os.path.join(savePos, str(temp)+'_gauss_'+str(i)+type),dst)
            print(str(temp)+'_gauss_'+str(i))
            # 均值滤波 对椒盐效果较好 只需要调核大小
            dst2 = cv2.blur(img, (i,i))
            cv2.namedWindow('mean_filter', cv2.WINDOW_NORMAL)
            cv2.imshow('mean_filter', dst2)
            cv2.imwrite(os.path.join(savePos, str(temp)+'_mean_'+str(i)+type),dst2)
            print(str(temp) + '_mean_' + str(i))
            # 中值滤波 k为方框，会进行归一化处理
            dst3 = cv2.medianBlur(img, i)
            cv2.namedWindow('median_filter', cv2.WINDOW_NORMAL)
            cv2.imshow('median_filter', dst3)
            cv2.imwrite(os.path.join(savePos, str(temp)+'_median_'+str(i)+type),dst3)
            print(str(temp) + '_median_' + str(i))

        # 双边滤波 d=5-9 sigmalColor&Space = 1-150
        # https://blog.csdn.net/qq_49478668/article/details/123488527
        for i in range(5,150,20):
            for j in range(5,220,20):
                dst4 = cv2.bilateralFilter(img,d=i,sigmaColor=j,sigmaSpace=j )
                cv2.namedWindow('bilateral_filter', cv2.WINDOW_NORMAL)
                cv2.imshow('bilasteral_filter', dst4)
                cv2.imwrite(os.path.join(savePos, str(temp)+'_bilasteral_'+str(i)+'_'+str(j)+type),dst4)
                print(str(temp) + '_bila_' + str(i)+'_'+str(j))

        # cv2.waitKey(0)
