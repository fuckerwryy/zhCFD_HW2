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


def plot_errors(h_list, errors, title):
    """绘制误差随步长的变化曲线"""
    plt.figure(figsize=(10, 6))
    markers = {'first_order_forward': 'o', 'first_order_centered': 's',
               'second_order_back': 'D', 'second_order_centered': '^'}
    for key in errors:
        plt.loglog(h_list, errors[key], f'{markers[key]}-', label=key)
    plt.loglog(h_list, h_list, 'k--', label='O(h)')
    plt.loglog(h_list, h_list ** 2, 'k:', label='O(h²)')
    # 设置坐标轴
    ax = plt.gca()

    # 增加x轴刻度密度
    from matplotlib.ticker import LogLocator, LogFormatterSciNotation
    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=15))  # 主刻度
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))  # 次刻度
    ax.xaxis.set_major_formatter(LogFormatterSciNotation())  # 科学计数法格式

    # 强制显示所有步长标签
    ax.set_xticks(h_list)
    ax.set_xticklabels([f"{h:.2e}" for h in h_list], rotation=45, ha='right', fontsize=9)

    # 标签和标题
    plt.xlabel(' Δx', fontsize=12)
    plt.ylabel('maximum error', fontsize=12)
    plt.title(title, fontsize=14)

    # 网格和图例
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend(fontsize=10)

    # 自动调整布局
    plt.tight_layout()
    plt.show()

