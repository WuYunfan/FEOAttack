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
    pdf = PdfPages('hyper-gowalla.pdf')
    base_mean = [0.423]
    mean = np.array([0.892, 1.057, 0.852, 0.664, 0.629, 0.53, 0.527, 0.544])[::-1].copy()
    value = np.array([0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001, 0])[::-1].copy()
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=4, constrained_layout=True, figsize=(30, 5))
    ax1.plot(np.arange(len(value)), mean, 'x-', markersize=7, color='#7570A0', linewidth=2,
             markeredgecolor=darken_color('#7570A0', 0.2), label='FEO')
    ax1.plot(np.arange(len(value)), base_mean * len(value), markersize=7, color='#F7B76D', linewidth=2,
             markeredgecolor=darken_color('#F7B76D', 0.2), label='None')
    ax1.set_title('Analysis of hyper-parameter $w_p$', fontsize=19)
    ax1.set_xticks(np.arange(len(value)))
    ax1.set_xticklabels(value, fontsize=21)
    # ax1.yaxis.set_major_locator(MultipleLocator(0.1))
    ax1.tick_params(axis='y', labelsize=21)
    ax1.set_xlabel('Promotion loss weight $w_p$', fontsize=21)
    ax1.set_ylabel('Recall@50 (%)', fontsize=21)
    ax1.grid(True, which='major', linestyle='--', linewidth=0.8)
    ax1.minorticks_on()
    ax1.tick_params(which='major', direction='in')
    ax1.xaxis.set_ticks_position('both')
    ax1.yaxis.set_ticks_position('both')
    ax1.legend(fontsize=21, loc='lower right', bbox_to_anchor=(1., 0.1))

    mean = np.array([0.881, 0.866, 0.917, 0.852, 0.939, 0.932, 0.92, 0.696])[::-1].copy()
    value = np.array([0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001, 0])[::-1].copy()
    ax2.plot(np.arange(len(value)), mean, 'x-', markersize=7, color='#7570A0', linewidth=2,
             markeredgecolor=darken_color('#7570A0', 0.2), label='FEO')
    ax2.plot(np.arange(len(value)), base_mean * len(value), markersize=7, color='#F7B76D', linewidth=2,
             markeredgecolor=darken_color('#F7B76D', 0.2), label='None')
    ax2.set_title('Analysis of hyper-parameter $w_m$', fontsize=19)
    ax2.set_xticks(np.arange(len(value)))
    ax2.set_xticklabels(value, fontsize=21)
    # ax2.yaxis.set_major_locator(MultipleLocator(0.1))
    ax2.tick_params(axis='y', labelsize=21)
    ax2.set_xlabel('$\\ell2$ loss weight $w_m$', fontsize=21)
    ax2.grid(True, which='major', linestyle='--', linewidth=0.8)
    ax2.minorticks_on()
    ax2.tick_params(which='major', direction='in')
    ax2.xaxis.set_ticks_position('both')
    ax2.yaxis.set_ticks_position('both')
    ax2.legend(fontsize=21, loc='lower right', bbox_to_anchor=(1., 0.1))

    mean = np.array([0.51, 1.014, 0.934, 0.861, 0.852, 0.886, 0.846, 0.831])[::-1].copy()
    value = np.array([0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001, 0])[::-1].copy()
    ax3.plot(np.arange(len(value)), mean, 'x-', markersize=7, color='#7570A0', linewidth=2,
             markeredgecolor=darken_color('#7570A0', 0.2), label='FEO')
    ax3.plot(np.arange(len(value)), base_mean * len(value), markersize=7, color='#F7B76D', linewidth=2,
             markeredgecolor=darken_color('#F7B76D', 0.2), label='None')
    ax3.set_title('Analysis of hyper-parameter $w_d$', fontsize=19)
    ax3.set_xticks(np.arange(len(value)))
    ax3.set_xticklabels(value, fontsize=21)
    # ax3.yaxis.set_major_locator(MultipleLocator(0.1))
    ax3.tick_params(axis='y', labelsize=21)
    ax3.set_xlabel('Diverse loss weight $w_d$', fontsize=21)
    ax3.grid(True, which='major', linestyle='--', linewidth=0.8)
    ax3.minorticks_on()
    ax3.tick_params(which='major', direction='in')
    ax3.xaxis.set_ticks_position('both')
    ax3.yaxis.set_ticks_position('both')
    ax3.legend(fontsize=21, loc='lower right', bbox_to_anchor=(1., 0.1))

    mean = np.array([0.92, 0.982, 0.998, 0.852, 0.844, 0.844, 0.844, 0.844, 0.844])[::-1].copy()
    value = np.array([1, 0.99, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0])[::-1].copy()
    ax4.plot(np.arange(len(value)), mean, 'x-', markersize=7, color='#7570A0', linewidth=2,
             markeredgecolor=darken_color('#7570A0', 0.2), label='FEO')
    ax4.plot(np.arange(len(value)), base_mean * len(value), markersize=7, color='#F7B76D', linewidth=2,
             markeredgecolor=darken_color('#F7B76D', 0.2), label='None')
    ax4.set_title('Analysis of hyper-parameter $p$', fontsize=19)
    ax4.set_xticks(np.arange(len(value)))
    ax4.set_xticklabels(value, fontsize=21)
    # ax4.yaxis.set_major_locator(MultipleLocator(0.1))
    ax4.tick_params(axis='y', labelsize=21)
    ax4.set_xlabel('Probability penalty coefficient $p$', fontsize=21)
    ax4.grid(True, which='major', linestyle='--', linewidth=0.8)
    ax4.minorticks_on()
    ax4.tick_params(which='major', direction='in')
    ax4.xaxis.set_ticks_position('both')
    ax4.yaxis.set_ticks_position('both')
    ax4.legend(fontsize=21, loc='lower right', bbox_to_anchor=(1., 0.1))
    plt.tight_layout()
    pdf.savefig()
    plt.close(fig)
    pdf.close()

    pdf = PdfPages('hyper-yelp.pdf')
    base_mean = [0.206]
    mean = np.array([0.849, 1.007, 0.489, 0.171, 0.163, 0.152, 0.143, 0.138])[::-1].copy()
    value = np.array([0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001, 0])[::-1].copy()
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=4, constrained_layout=True, figsize=(30, 5))
    ax1.plot(np.arange(len(value)), mean, 'x-', markersize=7, color='#7570A0', linewidth=2,
             markeredgecolor=darken_color('#7570A0', 0.2), label='FEO')
    ax1.plot(np.arange(len(value)), base_mean * len(value), markersize=7, color='#F7B76D', linewidth=2,
             markeredgecolor=darken_color('#F7B76D', 0.2), label='None')
    ax1.set_title('Analysis of hyper-parameter $w_p$', fontsize=19)
    ax1.set_xticks(np.arange(len(value)))
    ax1.set_xticklabels(value, fontsize=21)
    # ax1.yaxis.set_major_locator(MultipleLocator(0.1))
    ax1.tick_params(axis='y', labelsize=21)
    ax1.set_xlabel('Promotion loss weight $w_p$', fontsize=21)
    ax1.set_ylabel('Recall@50 (%)', fontsize=21)
    ax1.grid(True, which='major', linestyle='--', linewidth=0.8)
    ax1.minorticks_on()
    ax1.tick_params(which='major', direction='in')
    ax1.xaxis.set_ticks_position('both')
    ax1.yaxis.set_ticks_position('both')
    ax1.legend(fontsize=21, loc='lower right', bbox_to_anchor=(1., 0.1))

    mean = np.array([1.007, 0.979, 1.042, 0.466, 0.283, 0.287, 0.285, 0.293])[::-1].copy()
    value = np.array([0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001, 0])[::-1].copy()
    ax2.plot(np.arange(len(value)), mean, 'x-', markersize=7, color='#7570A0', linewidth=2,
             markeredgecolor=darken_color('#7570A0', 0.2), label='FEO')
    ax2.plot(np.arange(len(value)), base_mean * len(value), markersize=7, color='#F7B76D', linewidth=2,
             markeredgecolor=darken_color('#F7B76D', 0.2), label='None')
    ax2.set_title('Analysis of hyper-parameter $w_m$', fontsize=19)
    ax2.set_xticks(np.arange(len(value)))
    ax2.set_xticklabels(value, fontsize=21)
    # ax2.yaxis.set_major_locator(MultipleLocator(0.1))
    ax2.tick_params(axis='y', labelsize=21)
    ax2.set_xlabel('$\\ell2$ loss weight $w_m$', fontsize=21)
    ax2.grid(True, which='major', linestyle='--', linewidth=0.8)
    ax2.minorticks_on()
    ax2.tick_params(which='major', direction='in')
    ax2.xaxis.set_ticks_position('both')
    ax2.yaxis.set_ticks_position('both')
    ax2.legend(fontsize=21, loc='lower right', bbox_to_anchor=(1., 0.1))

    mean = np.array([1.007, 0.966, 0.991, 0.953, 1.027, 1.001, 0.892, 0.883])[::-1].copy()
    value = np.array([0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001, 0])[::-1].copy()
    ax3.plot(np.arange(len(value)), mean, 'x-', markersize=7, color='#7570A0', linewidth=2,
             markeredgecolor=darken_color('#7570A0', 0.2), label='FEO')
    ax3.plot(np.arange(len(value)), base_mean * len(value), markersize=7, color='#F7B76D', linewidth=2,
             markeredgecolor=darken_color('#F7B76D', 0.2), label='None')
    ax3.set_title('Analysis of hyper-parameter $w_d$', fontsize=19)
    ax3.set_xticks(np.arange(len(value)))
    ax3.set_xticklabels(value, fontsize=21)
    # ax3.yaxis.set_major_locator(MultipleLocator(0.1))
    ax3.tick_params(axis='y', labelsize=21)
    ax3.set_xlabel('Diverse loss weight $w_d$', fontsize=21)
    ax3.grid(True, which='major', linestyle='--', linewidth=0.8)
    ax3.minorticks_on()
    ax3.tick_params(which='major', direction='in')
    ax3.xaxis.set_ticks_position('both')
    ax3.yaxis.set_ticks_position('both')
    ax3.legend(fontsize=21, loc='lower right', bbox_to_anchor=(1., 0.1))

    mean = np.array([0.585, 0.903, 0.957, 1.007, 1.007, 1.007, 1.007, 1.007, 1.007])[::-1].copy()
    value = np.array([1, 0.99, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0])[::-1].copy()
    ax4.plot(np.arange(len(value)), mean, 'x-', markersize=7, color='#7570A0', linewidth=2,
             markeredgecolor=darken_color('#7570A0', 0.2), label='FEO')
    ax4.plot(np.arange(len(value)), base_mean * len(value), markersize=7, color='#F7B76D', linewidth=2,
             markeredgecolor=darken_color('#F7B76D', 0.2), label='None')
    ax4.set_title('Analysis of hyper-parameter $p$', fontsize=19)
    ax4.set_xticks(np.arange(len(value)))
    ax4.set_xticklabels(value, fontsize=21)
    # ax4.yaxis.set_major_locator(MultipleLocator(0.1))
    ax4.tick_params(axis='y', labelsize=21)
    ax4.set_xlabel('Probability penalty coefficient $p$', fontsize=21)
    ax4.grid(True, which='major', linestyle='--', linewidth=0.8)
    ax4.minorticks_on()
    ax4.tick_params(which='major', direction='in')
    ax4.xaxis.set_ticks_position('both')
    ax4.yaxis.set_ticks_position('both')
    ax4.legend(fontsize=21, loc='lower right', bbox_to_anchor=(1., 0.1))
    plt.tight_layout()
    pdf.savefig()
    plt.close(fig)
    pdf.close()



if __name__ == '__main__':
    main()