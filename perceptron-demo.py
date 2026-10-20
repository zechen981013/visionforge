#只支持单个样本 
import numpy as np
import torch as t

def sigmoid(x):
    return 1/(1+t.exp(-x))

def step(x, threshold=150):
    """平滑激发函数：用 sigmoid 实现，x 以 threshold 为中心平滑过渡。
    x = threshold 时输出 0.5，x 越大越接近 1，越小越接近 0"""
    return sigmoid(x - threshold)


student = t.tensor([[2,3],[4,1],[2,2]])
price = t.tensor([[20],[30]])
print (price.shape)


money = student @ price +50
print("金额:", money.flatten())

activated = step(money)
print("激发结果:", activated.flatten())



