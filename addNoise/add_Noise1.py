# -*- coding:utf-8 -*-
# @Time : 2022/6/8 8:05 下午
# @Author : Bin Bin Xue
# @File : add_Noise
# @Project : cv_works

# 噪声分类
# 1. 加性噪声，此类噪声与输⼊图像信号⽆关，含噪图像可表⽰为f(x, y)=g(x, y)+n(x, y),
# 噪声及光导摄像管的摄像机扫描图像时产⽣的噪声就属这类噪声；
# 2. 乘性噪声，此类噪声与图像信号有关，含噪图像可表⽰为f(x, y)=g(x, y)+n(x ,y)g(x, y),
# 飞点扫描器扫描图像时的噪声，电视图像中的相⼲噪声，胶⽚中的颗粒噪声就属于此类噪声。

# 添加噪声：高斯噪声、泊松噪声、椒盐噪声、瑞利噪声、伽马噪声、均匀噪声、随机噪声


# 导入opencv库
from cv2 import cv2
# 导入图像处理算法模块
import skimage
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']


# 七种噪声添加（高斯噪声、泊松噪声、椒盐噪声、瑞利噪声、伽马噪声、均匀噪声、随机噪声）

# 1.Gauss噪声（参数还包括均值和方差，均值对着分布中心，方差对着高度）
def add_noiseGaussNoise(img, mu=0.0, sigma=20.0):
    noiseGauss = np.random.normal(mu, sigma, img.shape)
    imgGaussNoise = img + noiseGauss
    imgGaussNoise = np.uint8(cv2.normalize(imgGaussNoise, None, 0, 255, cv2.NORM_MINMAX))
    return imgGaussNoise


def add_noiseGaussNoise2(img, mean=0, var=0.05):
    temp = img.copy()
    gauss_noiseAdd = skimage.util.random_noise(temp, mode='gaussian', mean=mean, var=var, clip=True)
    return gauss_noiseAdd


def clamp(pv):
    if pv > 255:
        return 255
    elif pv < 0:
        return 0
    else:
        return pv


def add_noiseGaussNoise3(img):
    noise_img = img.copy()
    h, w, c = noise_img.shape
    for row in range(h):
        for col in range(w):
            s = np.random.normal(0, 20, 3)
            b = noise_img[row, col, 0]
            g = noise_img[row, col, 1]
            r = noise_img[row, col, 2]
            noise_img[row, col, 0] = clamp(b + s[0])
            noise_img[row, col, 1] = clamp(b + s[1])
            noise_img[row, col, 2] = clamp(b + s[2])
    return noise_img


# 2.Rayleigh噪声
def add_noiseRayleigh(img, a=30.0):
    noiseRayleigh = np.random.rayleigh(a, size=img.shape)
    imgRayleighNoise = img + noiseRayleigh
    imgRayleighNoise = np.uint8(cv2.normalize(imgRayleighNoise, None, 0, 255, cv2.NORM_MINMAX))
    return imgRayleighNoise


# 3.Gamma噪声（b=1时为指数噪声，b>1时叠加多个指数噪声的即为Gamma噪声）
def add_noiseGamma(img, shape=10.0, scale=2.5):
    noiseGamma = np.random.gamma(shape=shape, scale=scale, size=img.shape)
    imgGammaNoise = img + noiseGamma
    imgGammaNoise = np.uint8(cv2.normalize(imgGammaNoise, None, 0, 255, cv2.NORM_MINMAX))
    return imgGammaNoise


# 4.Poisson噪声
def add_noisePoisson(img):
    noisePos = skimage.util.random_noise(img, mode='poisson', clip=True)
    return noisePos


# 5.均匀噪声
def add_noiseUniform(img):
    noiseUniform = skimage.util.random_noise(img, mode='speckle')
    return noiseUniform


# 6.椒盐噪声（由椒噪声和盐噪声组成）
def add_noiseSP(img, proportion):
    noise_img = img
    height, width = noise_img.shape[0], noise_img.shape[1]  # 获取高度宽度像素值
    num = int(height * width * proportion)  # 一个准备加入多少噪声小点
    for i in range(num):
        w = np.random.randint(0, width - 1)
        h = np.random.randint(0, height - 1)
        if np.random.randint(0, 1) == 0:
            noise_img[h, w] = 0
        else:
            noise_img[h, w] = 255
    return noise_img


def add_noiseSP2(img, amount=0.05, sp=0.5):
    noiseSP = skimage.util.random_noise(img, mode='s&p', amount=amount, salt_vs_pepper=sp)
    return noiseSP


# 7.随机噪声(随机在图像上将像素点的灰度值转为255，即白色）
def add_noiseRandom(img, noise_pro,gray):
    img_noise = img.copy()
    # 原图是RGB，三通道；灰度图是二通道
    if gray == 0:
        rows, cols, chn = img_noise.shape
        noise_num = noise_pro * rows * cols * chn
        for i in range(int(noise_num)):
            x = np.random.randint(0, rows)
            y = np.random.randint(0, cols)
            img_noise[x, y, :] = 255
    else:
        rows, cols = img_noise.shape
        noise_num = noise_pro * rows * cols
        for i in range(int(noise_num)):
            x = np.random.randint(0, rows)
            y = np.random.randint(0, cols)
            img_noise[x, y] = 255
    return img_noise
