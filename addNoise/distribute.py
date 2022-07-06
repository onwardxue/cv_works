# -*- coding:utf-8 -*-
# @Time : 2022/6/13 3:51 下午
# @Author : Bin Bin Xue
# @File : distribute
# @Project : cv_works
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.stats as st

# 设置图像显示中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']


# 正态分布计算公式
def gf(x, mu=0, sigma=1.0):
    '''根据正态分布计算公式，由自变量x计算因变量的值
        Argument:
          x: array
            输入数据（自变量）
          mu: float
            均值
          sigma: float
            方差
    '''
    left = 1 / (np.sqrt(2 * math.pi) * np.sqrt(sigma))
    right = np.exp(-(x - mu) ** 2 / (2 * sigma))
    return left * right


# 高斯分布绘制
def gauss():
    # 自变量
    x = np.arange(-4, 5, 0.1)
    # 因变量（不同均值或方差）
    y_1 = gf(x, 0, 0.2)
    y_2 = gf(x, 0, 1.0)
    y_3 = gf(x, 0, 5.0)
    y_4 = gf(x, -2, 0.5)

    # 绘图
    plt.plot(x, y_1, color='green')
    plt.plot(x, y_2, color='blue')
    plt.plot(x, y_3, color='yellow')
    plt.plot(x, y_4, color='red')
    # 设置坐标系
    plt.xlim(-5.0, 5.0)
    plt.ylim(-0.2, 1)

    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].set_position(('data', 0))
    ax.yaxis.set_ticks_position('left')
    ax.spines['left'].set_position(('data', 0))

    plt.legend(labels=['$\mu = 0, \sigma^2=0.2$', '$\mu = 0, \sigma^2=1.0$', '$\mu = 0, \sigma^2=5.0$',
                       '$\mu = -2, \sigma^2=0.5$'])
    plt.show()


def gamma():
    # 确定绘图区域尺⼨
    fig = plt.figure(figsize=(8, 8))
    ax1 = fig.add_subplot(1, 1, 1)
    x = np.arange(0.01, 20, 0.01)

    # 绘制gamma分布曲线
    y1 = st.gamma.pdf(x, 1, scale=2)  # "α=1,β=2"
    y2 = st.gamma.pdf(x, 2, scale=2)  # "α=2,β=2"
    y3 = st.gamma.pdf(x, 3, scale=2)  # "α=3,β=2"
    y4 = st.gamma.pdf(x, 5, scale=1)  # "α=5,β=1"
    y5 = st.gamma.pdf(x, 9, scale=0.5)  # "α=9,β=0.5"

    # 设置图例
    ax1.plot(x, y1, label="α=1,β=2")
    ax1.plot(x, y2, label="α=2,β=2")
    ax1.plot(x, y3, label="α=3,β=2")
    ax1.plot(x, y4, label="α=5,β=1")
    ax1.plot(x, y5, label="α=9,β=0.5")

    # 设置坐标轴标题
    ax1.set_xlabel('x')
    ax1.set_ylabel('p(x)')
    ax1.set_title("Gamma distribute plot")
    ax1.legend(loc="best")
    plt.show()


# 瑞利分布
def rayleigh():
    # 确定绘图区域尺⼨
    fig = plt.figure(figsize=(8, 8))
    ax1 = fig.add_subplot(1, 1, 1)
    x = np.arange(0.01, 20, 0.01)

    # 绘制gamma分布曲线
    y1 = st.rayleigh.pdf(x, 1, scale=2)

    # 设置图例
    ax1.plot(x, y1)

    # 设置坐标轴标题
    ax1.set_xlabel('x')
    ax1.set_ylabel('p(x)')
    plt.show()

# 椒盐分布
def psn():
    # 确定绘图区域尺⼨
    fig = plt.figure(figsize=(5, 5))
    ax1 = fig.add_subplot(1, 1, 1)
    y = np.arange(0.1, 20, 0.1)

    # 绘制gamma分布曲线
    x = 5
    y = 5
    # 设置图例
    ax1.plot(x, y)

    # 设置坐标轴标题
    plt.vlines(2,0,6,color='red')
    plt.vlines(1,0,3,color='blue')
    plt.hlines(3,0,1,color='gray',linestyles='dotted')
    plt.hlines(6,0,2,color='gray',linestyles='dotted')
    plt.show()

# 均匀分布
def uni():
    # 确定绘图区域尺⼨
    fig = plt.figure(figsize=(5, 5))
    ax1 = fig.add_subplot(1, 1, 1)
    y = np.arange(0.1, 20, 0.1)

    # 绘制gamma分布曲线
    x = 5
    y = 5
    # 设置图例
    ax1.plot(x, y)

    # 设置坐标轴标题
    plt.vlines(3,0,3,color='red')
    plt.vlines(1,0,3,color='red')
    plt.hlines(3,1,3,color='red',)
    plt.hlines(3,0,1,color='gray',linestyles='dotted')

    plt.show()


def poi():
    X = range(0, 51)
    Y = []
    y2 = []
    y3 = []
    y4 = []
    for k in X:
        p = st.poisson.pmf(k, 20)
        p2 = st.poisson.pmf(k, 10)
        p3 = st.poisson.pmf(k, 5)
        p4 = st.poisson.pmf(k, 30)
        Y.append(p)
        y2.append(p2)
        y3.append(p3)
        y4.append(p4)

    plt.plot(X, Y, color="red", label="k=20")
    plt.plot(X, y2, color="blue", label="k=10")
    plt.plot(X, y3, color="green",label="k=5")
    plt.plot(X, y4, color="orange",label="k=30")


    plt.xlabel("z")
    plt.ylabel("p(z)")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    # gauss()
    # gamma()
    # rayleigh()
    # psn()
    # uni()
    poi()