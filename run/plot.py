from dataset import get_dataset
from model import get_model
import torch
from config import get_gowalla_config
import numpy as np
import matplotlib.pyplot as plt
import dgl
import scipy.sparse as sp
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
from matplotlib.ticker import MultipleLocator
import matplotlib.ticker as ticker
plt.rc('font', family='Times New Roman')
plt.rcParams['pdf.fonttype'] = 42

def darken_color(color, amount=0.5):
    """
    将给定颜色变暗一些
    amount: 取值范围从0(无变化)到1(完全黑)
    """
    import matplotlib.colors as mcolors
    c = mcolors.colorConverter.to_rgb(color)
    return max(c[0] - amount, 0), max(c[1] - amount, 0), max(c[2] - amount, 0)


def main():
    mean = np.array([0.892, 1.057, 0.852, 0.664, 0.629, 0.53, 0.527, 0.544])[::-1].copy()
    std = np.array([0.294, 0.315, 0.31, 0.35, 0.374, 0.334, 0.337, 0.308])[::-1].copy()
    value = np.array([0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001, 0])[::-1].copy()
    pdf = PdfPages('adv_weight.pdf')
    fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True, figsize=(10, 7))
    ax.plot(np.arange(len(value)), mean, 'x-', markersize=7, color='#7570A0', linewidth=2, markeredgecolor=darken_color('#7570A0', 0.2))
    ax.set_title('Analysis of hyper-parameter $w_a$', fontsize=19)
    ax.set_ylim(0.5, 1.2)
    ax.set_xticks(np.arange(len(value)))
    ax.set_xticklabels(value, fontsize=21)
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.tick_params(axis='y', labelsize=21)
    ax.set_ylabel('Recall@50 (%)', fontsize=21)
    ax.grid(True, which='major', linestyle='--', linewidth=0.8)
    ax.minorticks_on()
    ax.tick_params(which='major', direction='in')
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    plt.tight_layout()
    pdf.savefig()
    plt.close(fig)
    pdf.close()

    mean = np.array([0.934, 0.861, 0.852, 0.886, 0.846, 0.807, 0.904, 0.831])[::-1].copy()
    std = np.array([0.307, 0.359, 0.31, 0.308, 0.295, 0.276, 0.389, 0.291])[::-1].copy()
    value = np.array([0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001, 0])[::-1].copy()
    pdf = PdfPages('diverse_weight.pdf')
    fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True, figsize=(10, 7))
    ax.plot(np.arange(len(value)), mean, 'x-', markersize=7, color='#7570A0', linewidth=2, markeredgecolor=darken_color('#7570A0', 0.2))
    ax.set_title('Analysis of hyper-parameter $w_d$', fontsize=19)
    ax.set_ylim(0.6, 1.2)
    ax.set_xticks(np.arange(len(value)))
    ax.set_xticklabels(value, fontsize=21)
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.tick_params(axis='y', labelsize=21)
    ax.set_ylabel('Recall@50 (%)', fontsize=21)
    ax.grid(True, which='major', linestyle='--', linewidth=0.8)
    ax.minorticks_on()
    ax.tick_params(which='major', direction='in')
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    plt.tight_layout()
    pdf.savefig()
    plt.close(fig)
    pdf.close()

    mean = np.array([0.917, 0.852, 0.939, 0.932, 0.92, 0.767, 0.742, 0.696])[::-1].copy()
    std = np.array([0.324, 0.31, 0.427, 0.382, 0.387, 0.35, 0.387, 0.274])[::-1].copy()
    value = np.array([0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001, 0])[::-1].copy()
    pdf = PdfPages('l2_weight.pdf')
    fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True, figsize=(10, 7))
    ax.plot(np.arange(len(value)), mean, 'x-', markersize=7, color='#7570A0', linewidth=2, markeredgecolor=darken_color('#7570A0', 0.2))
    ax.set_ylim(0.6, 1.2)
    ax.set_xticks(np.arange(len(value)))
    ax.set_xticklabels(value, fontsize=21)
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.tick_params(axis='y', labelsize=21)
    ax.set_ylabel('Recall@50 (%)', fontsize=21)
    ax.grid(True, which='major', linestyle='--', linewidth=0.8)
    ax.minorticks_on()
    ax.tick_params(which='major', direction='in')
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    plt.tight_layout()
    pdf.savefig()
    plt.close(fig)
    pdf.close()



if __name__ == '__main__':
    main()