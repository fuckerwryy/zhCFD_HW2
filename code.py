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
# ======================
# 误差分析模块
# ======================

def compute_errors(f, df_exact, d2f_exact, x_range=(0.0, 2 * np.pi), num_points=100, dtype=np.float64):
    """
    计算不同网格间距下的误差
    """
    steps = np.array([16, 32, 64, 128, 256, 512])  # 网格点数
    h_list = (x_range[1] - x_range[0]) / (steps - 1)  # Δx值列表

    errors = {
        'first_order_forward': [],
        'first_order_centered': [],
        'second_order_back': [],
        'second_order_centered': []
    }

    for n in steps:
        x = np.linspace(x_range[0], x_range[1], n, dtype=dtype)
        dx = x[1] - x[0]
        u = f(x)

        # 计算数值导数
        du_forward = first_order_forward(u, dx)
        du_centered = first_order_centered(u, dx)
        d2u_back = second_order_back(u, dx)
        d2u_fourth = second_order_centered(u, dx)

        # 计算最大误差（忽略边界）
        errors['first_order_forward'].append(np.abs(du_forward[:] - df_exact(x)[:-1]).max())
        errors['first_order_centered'].append(np.abs(du_centered[:] - df_exact(x)[1:-1]).max())
        errors['second_order_back'].append(np.abs(d2u_back[:] - d2f_exact(x)[2:]).max())
        errors['second_order_centered'].append(np.abs(d2u_fourth[:] - d2f_exact(x)[1:-1]).max())

    return h_list, errors


