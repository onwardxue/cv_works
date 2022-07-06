# -*- coding:utf-8 -*-
# @Time : 2022/6/8 8:05 下午
# @Author : Bin Bin Xue
# @File : add_Noise
# @Project : cv_works


# 导入opencv库
from cv2 import cv2
# 导入图像处理算法模块
import skimage
# 导入添加噪声方法
import add_Noise1 as an
import matplotlib.pyplot as plt
# 导入噪声过滤的方法
import filter_Img2 as fil
import numpy as np
from PIL import Image

# 设置图像显示中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
# 设置要实验的图片
name = "nature"
till = "nature_0_Gauss.jpeg"
route = name + ".jpeg"
# 设置图片保存地址
base_path = 'result/'


# 保存图片
# gray=0表示原始图像，=1表示灰度图
def saveImg(img, form, gray):
    paths = base_path + name + '_' + str(gray) + '_' + form + '.jpeg'
    plt.imshow(img)
    plt.axis('off')
    if(gray == 0):
        plt.savefig(paths,bbox_inches='tight',dpi=600)
    else:
        cv2.imwrite(paths,img)

# 询问是否要进行图片下载
def if_download(hint='是否下载该图片 Y/N\n'):
    a = input(hint)
    if a == 'Y' or a == 'y' or a == '1':
        return 1
    else:
        return 0


# 1.六种噪声添加（高斯噪声、泊松噪声、椒盐噪声、瑞利噪声、伽马噪声、均匀噪声）
# 噪声过滤方法研究
def addNoise(img, gray):
    noise1 = an.add_noiseGaussNoise2(img, mean=0, var=0.05)
    noise2 = an.add_noiseSP2(img, amount=0.2, sp=0.5)
    noise3 = an.add_noiseGamma(img, shape=10.0, scale=2)
    noise4 = an.add_noiseUniform(img)
    noise5 = an.add_noiseRayleigh(img)
    noise6 = an.add_noisePoisson(img)


    # 显示图像
    plt.subplot(241)
    plt.title('原始图像')
    plt.imshow(img, cmap='Greys_r')
    plt.subplot(242)
    plt.title('高斯噪声')
    plt.imshow(noise1, cmap='Greys_r')
    plt.subplot(243)
    plt.title('椒盐噪声')
    plt.imshow(noise2, cmap='Greys_r')
    plt.subplot(244)
    plt.title('伽马噪声')
    plt.imshow(noise3, cmap='Greys_r')
    plt.subplot(245)
    plt.title('均匀噪声')
    plt.imshow(noise4, cmap='Greys_r')
    plt.subplot(246)
    plt.title('瑞利噪声')
    plt.imshow(noise5, cmap='Greys_r')
    plt.subplot(247)
    plt.title('泊松噪声')
    plt.imshow(noise6, cmap='Greys_r')

    plt.show()


    if (if_download('是否要下载所有噪声图片？Y/N\n') == 1):
        saveImg(img, 'ori', gray)
        saveImg(noise1, 'Gauss', gray)
        saveImg(noise2, 'P&S', gray)
        saveImg(noise3, 'Gamma', gray)
        saveImg(noise4, 'Uniform', gray)
        saveImg(noise5, 'Rayleigh', gray)
        saveImg(noise6, 'Poisson', gray)



# 2.对于三种噪声参数的研究
def ex_par(img, gray):
    a = input('是否进行高斯噪声实验? 0/1\n')
    # 实验图像个数
    a1 = 0
    while int(a):
        mean, var = (input('请输入高斯实验的两个参数值（默认为0和0.05）：\n').split())
        noise1 = an.add_noiseGaussNoise2(img, mean=float(mean), var=float(var))
        a1 += 1
        # is_save = input('是否保存当前第' + str(a1) + '个图像？0/1\n')
        # if is_save == 1:
        #     saveImg(noise1, 'Gauss', gray)
        a = input('是否继续调整参数？0/1\n')
        plt.subplot(2, 3, a1)
        plt.title('gauss_' + 'var1=' + str(mean) + '_var2=' + str(var))
        plt.imshow(noise1, cmap='Greys_r')
    plt.show()
    if if_download() == 1:
        plt.savefig(base_path + '_高斯噪声实验_' + str(gray) + '_.jpeg')

    a = input('是否进行椒盐噪声实验? 0/1\n')
    a1 = 0
    while int(a):
        var1, var2 = (input('请输入椒盐噪声实验的两个参数值（默认为0.5和0.5）：\n').split())
        noise2 = an.add_noiseSP2(img, amount=float(var1), sp=float(var2))
        a1 += 1
        # is_save = input('是否保存当前第' + str(a1) + '个图像？0/1\n')
        # if is_save == 1:
        #     saveImg(noise2, 'S&P', gray)
        a = input('是否继续调整参数？0/1\n')
        plt.subplot(2, 3, a1)
        plt.title('gauss_' + 'var1=' + str(var1) + '_var2=' + str(var2))
        plt.imshow(noise2, cmap='Greys_r')
    plt.show()
    if if_download() == 1:
        plt.savefig(base_path + '_椒盐噪声实验_' + str(gray) + '_.jpeg')

    a = input('是否进行伽马噪声实验? 0/1\n')
    a1 = 0
    while int(a):
        var1, var2 = (input('请输入伽马噪声实验的两个参数值（默认为20.0和5）：\n').split())
        noise3 = an.add_noiseGamma(img, shape=float(var1), scale=float(var2))
        a1 += 1
        # is_save = input('是否保存当前第' + str(a1) + '个图像？0/1\n')
        # if is_save == 1:
        #     saveImg(noise3, 'Gamma', gray)
        a = input('是否继续调整参数？0/1\n')
        plt.subplot(2, 3, a1)
        plt.title('gauss_' + 'var1=' + str(var1) + '_var2=' + str(var2))
        plt.imshow(noise3, cmap='Greys_r')
    plt.show()
    if if_download() == 1:
        plt.savefig(base_path + '_伽马噪声实验_' + str(gray) + '_.jpeg')

    print('噪声实验结束')


# 进行图片加噪实验
def Noi_Exp():
    # 是否要灰度
    is_gray = input("是否要灰度？Y/N\n")
    # gray=0表示原始图像，=1表示灰度图
    gray = 0
    if is_gray == 'y' or is_gray == 'Y':
        # 读取图片
        img = cv2.imread(route, cv2.IMREAD_GRAYSCALE)
        gray = 1
    else:
        img = plt.imread(route)
    # 给图片分别添加六种噪声输出到2行4列的图中并保存图片
    addNoise(img, gray)
    # 是否要进入图片加噪调参环节
    is_next = input('是否要进入图片加噪调参环节？Y/N\n')
    a = 0
    if is_next == 'y' or is_next == 'Y':
        a = 1
        ex_par(img, gray)


def fiveFilter(src):
    # 均值滤波
    img_blur = fil.mean_filter(src, 3,3)
    cv2.imshow('blur', img_blur)
    # 方框滤波
    # img_boxFilter1 = fil.box_filter(src, -1,3,3, True)
    # 高斯滤波
    img_gaussianBlur = fil.gaussian_filter(src,3,3, 0, 0)
    # 中值滤波
    img_medianBlur = fil.median_filter(src, 3)
    # 双边滤波
    img_bilateralFilter = fil.bilateralFilter(src, 50, 100, 100)


def ex_fil(img, gray):
    print('提示：接下来将进行滤波实验，不同参数之间请按空格隔开')
    src = img.copy()
    # 是否进行均值滤波实验
    a = input('是否进行均值滤波实验? 0/1\n')
    a1 = 0
    while int(a):
        var1,var2= (input('请输入均值滤波实验的两个参数值（默认为3,3）：\n').split())
        img_blur = fil.mean_filter(src, int(var1),int(var2))
        a1 += 1
        # is_save = input('是否保存当前第' + str(a1) + '个图像？0/1\n')
        # if is_save == 1:
        #     saveImg(img_blur, 'img_blur', gray)
        a = input('是否继续调整参数？0/1\n')
        plt.subplot(2, 3, a1)
        plt.title('blur_' + 'var1=' + str(var1))
        plt.imshow(img_blur, cmap='Greys_r')
    plt.show()

    # # 是否进行方框滤波实验
    # a = input('是否进行方框滤波实验? 0/1\n')
    # a1 = 0
    # while int(a):
    #     var1, var2, var3,var4 = (input('请输入方框滤波实验的四个参数值（默认为-1,3,3,True）：\n').split())
    #     img_boxFilter1 = fil.box_filter(src, int(var1), int(var2), int(var3),bool(var4))
    #     a1 += 1
    #     a = input('是否继续调整参数？0/1\n')
    #     plt.subplot(2, 3, a1)
    #     plt.title('boxFilter_' + 'var1=' + str(var1) + 'var2=' + str(var2) + 'var3=' + str(var3))
    #     plt.imshow(img_boxFilter1, cmap='Greys_r')
    # plt.show()

    # 是否进行高斯滤波实验
    a = input('是否进行高斯滤波实验? 0/1\n')
    a1 = 0
    while int(a):
        var1, var2, var3,var4 = (input('请输入高斯滤波实验的四个参数值（默认为3,3,0,0）,其中内核大小要求为奇数：\n').split())
        img_gaussianBlur = fil.gaussian_filter(src, int(var1),int(var2),float(var3),float(var4))
        cv2.namedWindow('Gauss_filter'+str(a1+1), cv2.WINDOW_NORMAL)
        cv2.imshow('Gauss_filter', img_gaussianBlur)
        a1 += 1
        a = input('是否继续调整参数？0/1\n')
        plt.subplot(2, 3, a1)
        plt.title('gaussFilter_' + 'var1=' + str(var1) + 'var2=' + str(var2) + 'var3=' + str(var3))
        plt.imshow(img_gaussianBlur, cmap='Greys_r')
    plt.show()

    # 是否进行中值滤波实验
    a = input('是否进行中值滤波实验? 0/1\n')
    a1 = 0
    while int(a):
        var1 = input('请输入均值滤波实验的一个参数值（默认为(3)）：\n')
        img_medianBlur = fil.median_filter(src, int(var1))
        a1 += 1
        a = input('是否继续调整参数？0/1\n')
        plt.subplot(2, 3, a1)
        plt.title('medianBlur_' + 'var1=' + str(var1))
        plt.imshow(img_medianBlur, cmap='Greys_r')
    plt.show()

    # 是否进行双边滤波实验
    a = input('是否进行双边滤波实验? 0/1\n')
    a1 = 0
    while int(a):
        var1, var2, var3 = (input('请输入双边滤波实验的三个参数值（默认为50,100,100）：\n').split())
        img_bilateralFilter = cv2.bilateralFilter(src, 50, 100, 100)
        a1 += 1
        a = input('是否继续调整参数？0/1\n')
        plt.subplot(2, 3, a1)
        plt.title('gaussFilter_' + 'var1=' + str(var1) + 'var2=' + str(var2) + 'var3=' + str(var3))
        plt.imshow(img_bilateralFilter, cmap='Greys_r')
    plt.show()


def Fil_Exp():
    # 获取图片名
    till = input('请输入加噪目录(result)下需要过滤的图片名(包括格式）：\n')
    # 是否要灰度
    is_gray = input("是否要灰度？Y/N\n")
    # gray=0表示原始图像，=1表示灰度图
    gray = 0
    if is_gray == 'y' or is_gray == 'Y':
        # 从加噪声的目录(result)中读取图片
        img = cv2.imread((base_path + till), cv2.IMREAD_GRAYSCALE)
        gray = 1
    else:
        img = cv2.imread((base_path + till))
    cv2.imshow('ori', img)
    cv2.waitKey(0)
    # 用默认值对图片使用五种滤波进行过滤
    fiveFilter(img)
    cv2.destroyAllWindows()
    # 是否进入滤波调参
    if if_download(hint='是否进行滤波调参实验 Y/N\n') == 1:
        ex_fil(img, gray)


def main():
    k = 1
    while k:
        a = int(input('请选择实验项目：0-退出程序；1-图片加噪实验；2-图片过滤实验\n'))
        if a == 1:
            print('接下来将进行图片加噪实验,当看完图片要进行下一步时请点击Q键\n')
            Noi_Exp()
        elif a == 2:
            print('接下来将进行图片过滤实验,当看完图片要进行下一步时请点击Q键\n')
            Fil_Exp()
        elif a == 0:
            print('将退出程序！\n')
            break
        else:
            print('输入错误，请重新输入！\n')
        k = int(input('是否再次进行实验？0-退出，1-再次进行\n'))


if __name__ == '__main__':
    main()
