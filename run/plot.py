from dataset import get_dataset
from model import get_model
import torch
from config import get_gowalla_config
import numpy as np
import matplotlib.pyplot as plt
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
    pdf = PdfPages('hyper-gowalla-1.pdf')
    base_mean = [0.787]
    mean = np.array([0.686, 0.629, 0.680, 0.711, 1.002, 1.141])
    value = np.array([0, 0.0001, 0.001, 0.01, 0.1, 1])
    fig, ((ax11, ax12, ax13), (ax21, ax22, ax23)) = (
        plt.subplots(nrows=2, ncols=3, constrained_layout=True, figsize=(16, 8)))
    ax11.plot(np.arange(len(value)), mean, 'x-', markersize=7, color='#2E75B6', linewidth=2,
              markeredgecolor=darken_color('#2E75B6', 0.2), label='FEO')
    ax11.plot(np.arange(len(value)), base_mean * len(value), markersize=7, color='#D73027', linewidth=2,
              markeredgecolor=darken_color('#D73027', 0.2), label='Best Baseline')
    ax11.set_xticks(np.arange(len(value)))
    ax11.set_xticklabels(value, fontsize=21)
    ax11.set_ylim(0.3, 1.5)
    ax11.yaxis.set_major_locator(MultipleLocator(0.3))
    ax11.tick_params(axis='y', labelsize=21)
    ax11.set_xlabel('Adversarial loss weight $w_a$', fontsize=21)
    ax11.set_ylabel('Attack Recall@50 (%)', fontsize=21)
    ax11.grid(True, which='major', linestyle='--', linewidth=0.5, color='grey', dashes=(20, 10))
    ax11.minorticks_on()
    ax11.tick_params(which='major', direction='in')
    ax11.xaxis.set_ticks_position('both')
    ax11.yaxis.set_ticks_position('both')

    base_mean = [0.787]
    mean = np.array([1.142, 1.137, 1.141, 1.128, 0.705, 0.615])
    value = np.array([0, 0.0001, 0.001, 0.01, 0.1, 1])
    ax12.plot(np.arange(len(value)), mean, 'x-', markersize=7, color='#2E75B6', linewidth=2,
              markeredgecolor=darken_color('#2E75B6', 0.2), label='FEO')
    ax12.plot(np.arange(len(value)), base_mean * len(value), markersize=7, color='#D73027', linewidth=2,
              markeredgecolor=darken_color('#D73027', 0.2), label='Best Baseline')
    ax12.set_xticks(np.arange(len(value)))
    ax12.set_xticklabels(value, fontsize=21)
    ax12.set_ylim(0.3, 1.5)
    ax12.yaxis.set_major_locator(MultipleLocator(0.3))
    ax12.tick_params(axis='y', labelsize=21)
    ax12.set_xlabel('KL divergence loss weight $w_k$', fontsize=21)
    # ax12.set_ylabel('Attack Recall@50 (%)', fontsize=21)
    ax12.grid(True, which='major', linestyle='--', linewidth=0.5, color='grey', dashes=(20, 10))
    ax12.minorticks_on()
    ax12.tick_params(which='major', direction='in')
    ax12.xaxis.set_ticks_position('both')
    ax12.yaxis.set_ticks_position('both')

    attack = np.array([0.851, 0.801, 0.835, 0.925, 0.653, 0.583])
    recall = np.array([0.274, 0.188, 0.248, 0.145, 0.0683, 0.103])
    ax13.scatter(recall, attack, color='#2E75B6', edgecolor=darken_color('#2E75B6', 0.2), s=50)
    names = [f'$w_k$\n={w}' for w in [0, 0.0001, 0.001, 0.01, 0.1, 1]]
    for i, (xi, yi, name) in enumerate(zip(recall, attack, names)):
        ax13.text(xi - 0.01 if i != 0 else xi, yi + 0.01, name, fontsize=12, ha='left', va='bottom', color='#2E75B6', alpha=0.7)
    attack = np.array([0.299, 0.419, 0.582, 0.647, 0.708, 0.553, 0.559])
    recall = np.array([0.022, 0.005, 0.017, 0.047, 0.395, 0.296, 0.071])
    ax13.scatter(recall, attack, color='#D73027', edgecolor=darken_color('#D73027', 0.2), s=50)
    names = ['Random', 'Bandwagon', 'RevAdv', 'DPA2DL', 'RAPU-R', 'Leg-UP', 'UBA']
    for xi, yi, name in zip(recall, attack, names):
        ax13.text(xi - 0.01, yi + 0.01, name, fontsize=12, ha='left', va='bottom', color='#D73027', alpha=0.7)
    ax13.set_ylim(0.3, 1.5)
    ax13.yaxis.set_major_locator(MultipleLocator(0.3))
    ax13.tick_params(axis='x', labelsize=21)
    ax13.tick_params(axis='y', labelsize=21)
    ax13.set_xlabel('Detection Recall Rate', fontsize=21)
    # ax13.set_ylabel('Attack Recall@50 (%)', fontsize=21)  # 根据实际情况修改
    ax13.grid(True, which='major', linestyle='--', linewidth=0.5, color='grey', dashes=(20, 10))
    ax13.minorticks_on()
    ax13.tick_params(which='major', direction='in')
    ax13.xaxis.set_ticks_position('both')
    ax13.yaxis.set_ticks_position('both')

    base_mean = [0.787]
    mean = np.array([1.118, 1.141, 1.297, 1.285])
    value = np.array([1, 2, 5, 10])
    ax21.plot(np.arange(len(value)), mean, 'x-', markersize=7, color='#2E75B6', linewidth=2,
              markeredgecolor=darken_color('#2E75B6', 0.2), label='FEO')
    ax21.plot(np.arange(len(value)), base_mean * len(value), markersize=7, color='#D73027', linewidth=2,
              markeredgecolor=darken_color('#D73027', 0.2), label='Best Baseline')
    ax21.set_xticks(np.arange(len(value)))
    ax21.set_xticklabels(value, fontsize=21)
    ax21.set_ylim(0.3, 1.5)
    ax21.yaxis.set_major_locator(MultipleLocator(0.3))
    ax21.tick_params(axis='y', labelsize=21)
    ax21.set_xlabel('Maximum interaction limit $l$', fontsize=21)
    ax21.set_ylabel('Attack Recall@50 (%)', fontsize=21)
    ax21.grid(True, which='major', linestyle='--', linewidth=0.5, color='grey', dashes=(20, 10))
    ax21.minorticks_on()
    ax21.tick_params(which='major', direction='in')
    ax21.xaxis.set_ticks_position('both')
    ax21.yaxis.set_ticks_position('both')

    base_mean = [0.787]
    mean = np.array([1.282, 1.141, 0.92, 0.916, 1.076])
    value = np.array([5, 10, 20, 50, 100])
    ax22.plot(np.arange(len(value)), mean, 'x-', markersize=7, color='#2E75B6', linewidth=2,
              markeredgecolor=darken_color('#2E75B6', 0.2), label='FEO')
    ax22.plot(np.arange(len(value)), base_mean * len(value), markersize=7, color='#D73027', linewidth=2,
              markeredgecolor=darken_color('#D73027', 0.2), label='Best Baseline')
    ax22.set_xticks(np.arange(len(value)))
    ax22.set_xticklabels(value, fontsize=21)
    ax22.set_ylim(0.3, 1.5)
    ax22.yaxis.set_major_locator(MultipleLocator(0.3))
    ax22.tick_params(axis='y', labelsize=21)
    ax22.set_xlabel('Fake user generation step $s$', fontsize=21)
    # ax22.set_ylabel('Attack Recall@50 (%)', fontsize=21)
    ax22.grid(True, which='major', linestyle='--', linewidth=0.5, color='grey', dashes=(20, 10))
    ax22.minorticks_on()
    ax22.tick_params(which='major', direction='in')
    ax22.xaxis.set_ticks_position('both')
    ax22.yaxis.set_ticks_position('both')

    base_mean = [0.787]
    mean = np.array([1.226, 1.141, 1.119, 1.145, 1.105, 1.052])
    value = np.array([0.01, 0.05, 0.1, 0.2, 0.5, 1])
    ax23.plot(np.arange(len(value)), mean, 'x-', markersize=7, color='#2E75B6', linewidth=2,
              markeredgecolor=darken_color('#2E75B6', 0.2), label='FEO')
    ax23.plot(np.arange(len(value)), base_mean * len(value), markersize=7, color='#D73027', linewidth=2,
              markeredgecolor=darken_color('#D73027', 0.2), label='Best Baseline')
    ax23.set_xticks(np.arange(len(value)))
    ax23.set_xticklabels(value, fontsize=21)
    ax23.set_ylim(0.3, 1.5)
    ax23.yaxis.set_major_locator(MultipleLocator(0.3))
    ax23.tick_params(axis='y', labelsize=21)
    ax23.set_xlabel('Target hit ratio $h$', fontsize=21)
    # ax23.set_ylabel('Attack Recall@50 (%)', fontsize=21)
    ax23.grid(True, which='major', linestyle='--', linewidth=0.5, color='grey', dashes=(20, 10))
    ax23.minorticks_on()
    ax23.tick_params(which='major', direction='in')
    ax23.xaxis.set_ticks_position('both')
    ax23.yaxis.set_ticks_position('both')

    handles, labels = [], []
    for h, l in zip(*ax11.get_legend_handles_labels()):
        handles.append(h)
        labels.append(l)
    fig.legend(handles, labels, loc='upper center', ncol=len(handles), fontsize=21)
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    pdf.savefig()
    plt.close(fig)
    pdf.close()

    pdf = PdfPages('hyper-amazon-1.pdf')
    base_mean = [0.399]
    mean = np.array([0.441, 0.415, 0.514, 0.834, 1.363, 0.341])
    value = np.array([0, 0.0001, 0.001, 0.01, 0.1, 1])
    fig, ((ax11, ax12, ax13), (ax21, ax22, ax23)) = (
        plt.subplots(nrows=2, ncols=3, constrained_layout=True, figsize=(16, 8)))
    ax11.plot(np.arange(len(value)), mean, 'x-', markersize=7, color='#2E75B6', linewidth=2,
              markeredgecolor=darken_color('#2E75B6', 0.2), label='FEO')
    ax11.plot(np.arange(len(value)), base_mean * len(value), markersize=7, color='#D73027', linewidth=2,
              markeredgecolor=darken_color('#D73027', 0.2), label='Best Baseline')
    ax11.set_xticks(np.arange(len(value)))
    ax11.set_xticklabels(value, fontsize=21)
    ax11.set_ylim(0., 1.5)
    ax11.yaxis.set_major_locator(MultipleLocator(0.3))
    ax11.tick_params(axis='y', labelsize=21)
    ax11.set_xlabel('Adversarial loss weight $w_a$', fontsize=21)
    ax11.set_ylabel('Attack Recall@50 (%)', fontsize=21)
    ax11.grid(True, which='major', linestyle='--', linewidth=0.5, color='grey', dashes=(20, 10))
    ax11.minorticks_on()
    ax11.tick_params(which='major', direction='in')
    ax11.xaxis.set_ticks_position('both')
    ax11.yaxis.set_ticks_position('both')

    base_mean = [0.399]
    mean = np.array([1.439, 1.361, 1.363, 0.954, 0.141, 0.114])
    value = np.array([0, 0.0001, 0.001, 0.01, 0.1, 1])
    ax12.plot(np.arange(len(value)), mean, 'x-', markersize=7, color='#2E75B6', linewidth=2,
              markeredgecolor=darken_color('#2E75B6', 0.2), label='FEO')
    ax12.plot(np.arange(len(value)), base_mean * len(value), markersize=7, color='#D73027', linewidth=2,
              markeredgecolor=darken_color('#D73027', 0.2), label='Best Baseline')
    ax12.set_xticks(np.arange(len(value)))
    ax12.set_xticklabels(value, fontsize=21)
    ax12.set_ylim(0., 1.5)
    ax12.yaxis.set_major_locator(MultipleLocator(0.3))
    ax12.tick_params(axis='y', labelsize=21)
    ax12.set_xlabel('KL divergence loss weight $w_k$', fontsize=21)
    # ax12.set_ylabel('Attack Recall@50 (%)', fontsize=21)
    ax12.grid(True, which='major', linestyle='--', linewidth=0.5, color='grey', dashes=(20, 10))
    ax12.minorticks_on()
    ax12.tick_params(which='major', direction='in')
    ax12.xaxis.set_ticks_position('both')
    ax12.yaxis.set_ticks_position('both')

    attack = np.array([0.352, 0.368, 0.482, 0.386, 0.114, 0.067])
    recall = np.array([0.494, 0.377, 0.469, 0.154, 0.212, 0.155])
    ax13.scatter(recall, attack, color='#2E75B6', edgecolor=darken_color('#2E75B6', 0.2), s=50)
    names = [f'$w_k$\n={w}' for w in [0, 0.0001, 0.001, 0.01, 0.1, 1]]
    for xi, yi, name in zip(recall, attack, names):
        ax13.text(xi - 0.005, yi, name, fontsize=12, ha='left', va='bottom', color='#2E75B6', alpha=0.7)
    attack = np.array([0.004, 0.002, 0.336, 0.017])
    recall = np.array([0.169, 0.164, 0.076, 0.843])
    ax13.scatter(recall, attack, color='#D73027', edgecolor=darken_color('#D73027', 0.2), s=50)
    names = ['Random', 'Bandwagon', 'DPA2DL', 'RAPU-R']
    for xi, yi, name in zip(recall, attack, names):
        ax13.text(xi - 0.005, yi, name, fontsize=12, ha='left', va='bottom', color='#D73027', alpha=0.7)
    ax13.set_ylim(0., 0.6)
    ax13.yaxis.set_major_locator(MultipleLocator(0.1))
    ax13.tick_params(axis='x', labelsize=21)
    ax13.tick_params(axis='y', labelsize=21)
    ax13.set_xlabel('Detection Recall Rate', fontsize=21)
    # ax13.set_ylabel('Attack Recall@50 (%)', fontsize=21)  # 根据实际情况修改
    ax13.grid(True, which='major', linestyle='--', linewidth=0.5, color='grey', dashes=(20, 10))
    ax13.minorticks_on()
    ax13.tick_params(which='major', direction='in')
    ax13.xaxis.set_ticks_position('both')
    ax13.yaxis.set_ticks_position('both')

    base_mean = [0.399]
    mean = np.array([1.363, 0.900, 0.399, 0.176, 0.116])
    value = np.array([1, 2, 5, 10, 100])
    ax21.plot(np.arange(len(value)), mean, 'x-', markersize=7, color='#2E75B6', linewidth=2,
              markeredgecolor=darken_color('#2E75B6', 0.2), label='FEO')
    ax21.plot(np.arange(len(value)), base_mean * len(value), markersize=7, color='#D73027', linewidth=2,
              markeredgecolor=darken_color('#D73027', 0.2), label='Best Baseline')
    ax21.set_xticks(np.arange(len(value)))
    ax21.set_xticklabels(value, fontsize=21)
    ax21.set_ylim(0., 1.5)
    ax21.yaxis.set_major_locator(MultipleLocator(0.3))
    ax21.tick_params(axis='y', labelsize=21)
    ax21.set_xlabel('Maximum interaction limit $l$', fontsize=21)
    ax21.set_ylabel('Attack Recall@50 (%)', fontsize=21)
    ax21.grid(True, which='major', linestyle='--', linewidth=0.5, color='grey', dashes=(20, 10))
    ax21.minorticks_on()
    ax21.tick_params(which='major', direction='in')
    ax21.xaxis.set_ticks_position('both')
    ax21.yaxis.set_ticks_position('both')

    base_mean = [0.399]
    mean = np.array([2.205, 1.363, 0.769, 0.246, 0.248])
    value = np.array([50, 100, 200, 500, 1000])
    ax22.plot(np.arange(len(value)), mean, 'x-', markersize=7, color='#2E75B6', linewidth=2,
              markeredgecolor=darken_color('#2E75B6', 0.2), label='FEO')
    ax22.plot(np.arange(len(value)), base_mean * len(value), markersize=7, color='#D73027', linewidth=2,
              markeredgecolor=darken_color('#D73027', 0.2), label='Best Baseline')
    ax22.set_xticks(np.arange(len(value)))
    ax22.set_xticklabels(value, fontsize=21)
    ax22.set_ylim(0., 2.3)
    ax22.yaxis.set_major_locator(MultipleLocator(0.4))
    ax22.tick_params(axis='y', labelsize=21)
    ax22.set_xlabel('Fake user generation step $s$', fontsize=21)
    # ax22.set_ylabel('Attack Recall@50 (%)', fontsize=21)
    ax22.grid(True, which='major', linestyle='--', linewidth=0.5, color='grey', dashes=(20, 10))
    ax22.minorticks_on()
    ax22.tick_params(which='major', direction='in')
    ax22.xaxis.set_ticks_position('both')
    ax22.yaxis.set_ticks_position('both')

    base_mean = [0.399]
    mean = np.array([1.127, 1.453, 1.243, 1.904, 0.741, 0.219])
    value = np.array([0.01, 0.05, 0.1, 0.2, 0.5, 1])
    ax23.plot(np.arange(len(value)), mean, 'x-', markersize=7, color='#2E75B6', linewidth=2,
              markeredgecolor=darken_color('#2E75B6', 0.2), label='FEO')
    ax23.plot(np.arange(len(value)), base_mean * len(value), markersize=7, color='#D73027', linewidth=2,
              markeredgecolor=darken_color('#D73027', 0.2), label='Best Baseline')
    ax23.set_xticks(np.arange(len(value)))
    ax23.set_xticklabels(value, fontsize=21)
    ax23.set_ylim(0., 2.)
    ax23.yaxis.set_major_locator(MultipleLocator(0.3))
    ax23.tick_params(axis='y', labelsize=21)
    ax23.set_xlabel('Target hit ratio $h$', fontsize=21)
    # ax23.set_ylabel('Attack Recall@50 (%)', fontsize=21)
    ax23.grid(True, which='major', linestyle='--', linewidth=0.5, color='grey', dashes=(20, 10))
    ax23.minorticks_on()
    ax23.tick_params(which='major', direction='in')
    ax23.xaxis.set_ticks_position('both')
    ax23.yaxis.set_ticks_position('both')

    handles, labels = [], []
    for h, l in zip(*ax11.get_legend_handles_labels()):
        handles.append(h)
        labels.append(l)
    fig.legend(handles, labels, loc='upper center', ncol=len(handles), fontsize=21)
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    pdf.savefig()
    plt.close(fig)
    pdf.close()

if __name__ == '__main__':
    main()