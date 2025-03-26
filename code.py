import numpy as np
import matplotlib.pyplot as plt


def first_order_forward(u, dx):
    """一阶前向差分（数组长度自动减1）
    公式: (u_{i+1} - u_{i}) / dx
    """
    return (u[1:] - u[:-1]) / dx  # 直接返回N-1长度的数组



def first_order_centered(u, dx):
    """一阶导数中心差分（数组长度自动减2）
    公式: (u_{i+1} - u_{i-1} )/2dx
    """
    return (u[2:] - u[:-2]) / (2 * dx)  # 直接返回N-2长度的数组


def second_order_back(u, dx):
    """
    二阶导数：一阶精度差分（3个点）
    公式: (u_{i} - 2u_{i-1} + u_{i-2})/dx²
    """

    return (u[2:] - 2 * u[1:-1] + u[:-2]) / ( dx ** 2)  # 直接返回N-2长度的数组


def second_order_centered(u, dx):
    """
    二阶导数：二阶中心差分（3点模板）
    公式: (u_{i+1} - 2u_i + u_{i-1})/dx²
    """
    return (u[2:] - 2 * u[1:-1] + u[:-2]) / (dx ** 2)  # 直接返回N-2长度的数组



