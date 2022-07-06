# -*- coding:utf-8 -*-
# @Time : 2022/6/8 8:05 下午
# @Author : Bin Bin Xue
# @File : add_Noise
# @Project : cv_works

# 去噪方法分类：基于滤波器、基于模型、基于学习（深度学习）

# 滤波器方法：均值滤波、中值滤波、高斯滤波
# 根据噪声类型的不同，选择不同的滤波器过滤掉噪声。通常，对于椒盐噪声，选择中值滤波器（Median Filter），在去掉噪声的同时，
# 不会模糊图像；对于高斯噪声，选择高斯滤波器（Mean Filter）；均值滤波能够去掉噪声，但会对图像造成一定的模糊.

# 1、中值滤波
# 中值滤波是基于排序统计理论的一种能有效抑制噪声的非线性信号处理技术，中值滤波的基本原理是把数字图像或数字序列中一点的值
# 用该点的一个邻域中各点值的中值代替，让周围的像素值接近的真实值，从而消除孤立的噪声点。
# 2、高斯滤波
# 高斯滤波是一种线性平滑滤波，适用于消除高斯噪声，广泛应用于图像处理的减噪过程。高斯滤波就是对整幅图像进行加权平均的过程，
# 每一个像素点的值，都由其本身和邻域内的其他像素值经过加权平均后得到。
# 3、均值滤波
# 均值滤波是典型的线性滤波算法，它是指在图像上对目标像素给一个模板，该模板包括了其周围的临近像素(以目标像素为中心的周围8个像素，
# 构成一个滤波模板，即去掉目标像素本身)，再用模板中的全体像素的平均值来代替原来像素值。
# 边缘检测的目的是标识数字图像中亮度变化明显的点。高斯边缘检测是用高斯滤波的方式进行边缘检测。

# 导入opencv库
import copy
import math
from tkinter import Image

import cv2.cv2
from cv2 import cv2
# 导入图像处理算法模块
import skimage
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']


# 根据算法原理实现四个滤波算法

# 1.中值滤波
def medium_filter(img, ksize=3):
    rows, cols = img.shape[:2]
    pad_size= ksize // 2
    startSearchRow = pad_size
    endSearchRow = rows - pad_size - 1
    startSearchCol = pad_size
    endSearchCol = cols - pad_size - 1
    dst = np.zeros((rows, cols), dtype=np.uint8)
    # 中值滤波
    for y in range(startSearchRow, endSearchRow):
        for x in range(startSearchCol, endSearchCol):
            window = []
            for i in range(y - pad_size, y + pad_size + 1):
                for j in range(x - pad_size, x + pad_size + 1):
                    window.append(img[i][j])
            # 取中间值
            window = np.sort(window, axis=None)
            if len(window) % 2 == 1:
                medianValue = window[len(window) // 2]
            else:
                medianValue = int((window[len(window) // 2] + window[len(window) // 2 + 1]) / 2)
            dst[y][x] = medianValue
    return dst



# 2.均值滤波
def mean_filter(img,ksize):
    # 创建输出图像
    dst = np.copy(img)
    # 设置卷积核
    kernel  = np.ones((ksize,ksize))
    # 扩充图像边界
    pad_size=int((ksize-1)/2)
    dst = np.pad(dst,(pad_size,pad_size),mode="constant",constant_values=0)
    if len(img.shape)!=3:
        img=np.expand_dims(img, axis=-1)
    w,h=dst.shape
    for i in range(pad_size,w-pad_size):
        for j in range(pad_size,h-pad_size):
                dst[i,j]=np.sum(kernel*dst[i-pad_size: i+pad_size+1,j-pad_size:j+pad_size + 1])//(ksize**2)
    dst = dst[pad_size:w-pad_size,pad_size:h-pad_size]
    return dst


# 3.高斯滤波
# img为输入图像，k_size为核的大小，sigma为标准差
def gauss_filter(img,ksize=3,sigma=1.0):
    # 图像转成numpy数组形式
    img=np.asarray(np.unit8(img))
    # 得到各通道的像素值（三通道或两通道）
    if len(img.shape)!=3:
        img=np.expand_dims(img, axis=-1)
    H,W,D=img.shape

    #初始化高斯核矩阵和扩充图像（预防黑边的产生）
     # 取模版的一半
    tp= ksize//2
    # 生成目标图像（向外扩充加模版的一半尺寸）
    dst = np.zeros((H+tp*2,W+tp*2,D),dtype=np.float)
    dst[tp:tp+H,tp:tp+W]=img.copy().astype(np.float)

    #设置高斯核
    K = np.zeros((ksize,ksize),dtype=np.float)
    # 根据公式计算高斯核
    for i in range(-tp,-tp+ksize):
        for j in range(-tp,tp+ksize):
            K[j+tp,i+tp]=np.exp(-(i**2+j**2)/(2*(sigma ** 2)))
    K/=(2*np.pi*sigma*sigma)
    # 归一化
    K/=K.sum()
    tmp=dst.copy()

    # 高斯滤波处理图像
    for i in range(H):
        for j in range(W):
            for k in range(D):
                dst[tp+i,tp+j,k]=np.sum(K*tmp[i:i+ksize,j:j+ksize,k])
    dst = np.clip(dst,0,255)
    dst = dst[tp:tp+H,tp:tp+W].astype(np.unit8)
    return dst

# 4.双边滤波 直接用experiment(路径）创建该类，调用其中的方法进行双边滤波过滤
class BiFilter:

    def __init__(self, distance_sigma, range_sigma, radius):
        self.distance_sigma = distance_sigma
        self.range_sigma = range_sigma
        self.radius = radius
        self.spatial_weight_table = []
        self.range_weight_table = []

    def calculate_spatial_weight_table(self):
        for en_row in range(-self.radius, self.radius + 1):
            self.spatial_weight_table.append([])
            for en_col in range(-self.radius, self.radius + 1):
                distance = -(en_row ** 2 + en_col ** 2) / (2 * (self.distance_sigma ** 2))
                result = math.exp(distance)
                self.spatial_weight_table[en_row + self.radius].append(result)

    def calculate_range_weight_table(self):
        for en in range(0, 256):
            distance = -(en ** 2) / ((self.range_sigma ** 2) * 2)
            result = math.exp(distance)
            self.range_weight_table.append(result)

    def control_range(self, k):
        if k > 255:
            t = 255
        elif k < 0:
            t = 0
        else:
            t = k
        return t

    def extract(self, image):
        h = image.size[0]
        w = image.size[1]
        return h, w

    def to_pixel(self, pixels, i, j):
        rp = pixels[i, j][0]
        gp = pixels[i, j][1]
        bp = pixels[i, j][2]
        return rp, gp, bp

    def get_pixels(self, row, col, en_row, en_col, pixels):
        row_reset = row + en_row
        col_reset = col + en_col
        r = pixels[row_reset, col_reset][0]
        g = pixels[row_reset, col_reset][1]
        b = pixels[row_reset, col_reset][2]
        return r, g, b

    def convolution(self, radius, row, col, r2, g2, b2, r1, g1, b1):
        r_w = (self.spatial_weight_table[row + radius][col + radius] * self.range_weight_table[(abs(r2 - r1))])
        g_w = (self.spatial_weight_table[row + radius][col + radius] * self.range_weight_table[(abs(g2 - g1))])
        b_w = (self.spatial_weight_table[row + radius][col + radius] * self.range_weight_table[(abs(b2 - b1))])
        return r_w, g_w, b_w

    def Bilateral_Filtering(self, source_image):

        height, width = self.extract(source_image)
        radius = self.radius

        self.calculate_spatial_weight_table()
        self.calculate_range_weight_table()

        pixels = source_image.load()
        initial_data = []
        alter_image = copy.deepcopy()

        r_s = g_s = b_s = 0
        r_w_s = g_w_s = b_w_s = 0
        # 边缘像素不滤波（半径范围外的不进行滤波）
        for row in range(radius, height - radius):
            for col in range(radius, width - radius):
                # 对每个像素进行滤波
                rp, gp, bp = self.to_pixel(pixels, row, col)
                initial_data.append((rp, gp, bp))
                for en_row in range(-radius, radius + 1):
                    for en_col in range(-radius, radius + 1):
                        # 获得模块内的像素
                        rp_2, gp_2, bp_2 = self.get_pixels(row, col, en_row, en_col, pixels)
                        # 卷积计算
                        r_w, g_w, b_w = self.convolution(radius, en_row, en_col, rp_2, gp_2, bp_2, rp, gp, bp)
                        # 求和
                        r_w_s, g_w_s, b_w_s = self.addition(r_w_s, g_w_s, b_w_s, r_w, g_w, b_w)
                        # 鲜求和
                        r_s, g_s, b_s = self.addition2(r_s, g_s, b_s, r_w, g_w, b_w, rp_2, gp_2, bp_2)

                # 归一化过程 floor取最小整
                rp = self.uniform(rp, r_s, r_w_s)
                rp = self.uniform(gp, g_s, g_w_s)
                rp = self.uniform(bp, b_s, b_w_s)

                # 设置像素点(控制值域）
                reset_rgb = (self.control_range(rp), self.control_range(gp), self.control_range(bp))
                # 修改指定位置的像素（像素位置，值）
                alter_image.putpixel((row, col), reset_rgb)
                # 重置RGB各初始值
                r_s = g_s = b_s = 0
                r_w_s = g_w_s = b_w_s = 0
                r_w = g_w = b_w = 0
        # 返回修改后的图像
        return alter_image

    def addition(self, a, b, c, a1, b1, c1):
        return (a + a1), (b + b1), (c + c1)

    def addition2(self, a, b, c, a1, b1, c1, a2, b2, c2):
        return (a + a1 * float(a2)), (b + b1 * float(b2)), (c + c1 * float(c2))

    def uniform(self, rp, r_s, r_w_s):
        rp = int(math.floor(r_s / r_w_s))
        return rp


class Experiment:

    def __init__(self,image_route):
        self.image_route = image_route
        self.length = 25
        self.width = 20
        self.line = 2
        self.row = 2

    def image_process(self):
        # 图片路径转成统一的可用字符串
        img = str(self.image_route)
        # 打开图片
        img0 = cv2.imread(img)
        # 去掉.jpg，留下前缀名字
        img_name = self.extract(img)
        # 初始化图像
        self.initial_plot(img0)
        # 设置空间域sigma、值域sigma和模块半径
        ds, rs, radius = self.factor_setting()
        # 过滤图片并生成子图
        self.general_subplot(ds, rs, radius, img0, img_name)

    def extract(self, img):
        name = img[0:img.index('.')]
        return name

    def initial_plot(self, src):
        # 指定图像的高和宽
        plt.figure(figsize=(self.length, self.width))
        # 子图划分
        plt.subplot(self.line, self.row, 1)
        # 原始图取标题
        plt.title("SRC", fontsize=20)
        plt.imshow(src)
        plt.axis("off")

    def factor_setting(self):
        ds = 5
        rs = 15
        radius = 3
        return ds, rs, radius

    def general_subplot(self, ds, rs, radius, src, img_name):
        count = 1
        bf = BiFilter(ds, rs, radius)
        aft_image = bf.Bilateral_Filtering(src)
        aft_image.save(img_name + '_' + str(count) + '.jpg')
        print('第' + str(count) + '个图片已完成输出!')
        count += 1
        plt.subplot(self.line, self.row, count)
        plt.title('d=' + ds + ',r=' + rs + ',m=' + radius, fontsize=20)
        plt.imshow(aft_image)
        plt.axis('off')