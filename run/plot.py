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
    pdf = PdfPages('hyper-gowalla.pdf')
    base_mean = [0.787]
    mean = np.array([0.544, 0.527, 0.629, 0.854, 0.892, 0.443])
    value = np.array([0, 0.0001, 0.001, 0.01, 0.1, 1])
    fig, ((ax11, ax12, ax13), (ax21, ax22, ax23)) = (
        plt.subplots(nrows=2, ncols=3, constrained_layout=True, figsize=(20, 8)))
    ax11.plot(np.arange(len(value)), mean, 'x-', markersize=7, color='#7570A0', linewidth=2,
              markeredgecolor=darken_color('#7570A0', 0.2), label='FEO')
    ax11.plot(np.arange(len(value)), base_mean * len(value), markersize=7, color='#F7B76D', linewidth=2,
              markeredgecolor=darken_color('#F7B76D', 0.2), label='Best Baseline')
    ax11.set_xticks(np.arange(len(value)))
    ax11.set_xticklabels(value, fontsize=21)
    ax11.set_ylim(0.2, 1.2)
    ax11.yaxis.set_major_locator(MultipleLocator(0.2))
    ax11.tick_params(axis='y', labelsize=21)
    ax11.set_xlabel('Promotion loss weight $w_p$', fontsize=21)
    ax11.set_ylabel('Recall@50 (%)', fontsize=21)
    ax11.grid(True, which='major', linestyle='--', linewidth=0.5, color='black', dashes=(20, 10))
    ax11.minorticks_on()
    ax11.tick_params(which='major', direction='in')
    ax11.xaxis.set_ticks_position('both')
    ax11.yaxis.set_ticks_position('both')

    base_mean = [0.787]
    mean = np.array([0.696, 0.92, 0.938, 0.905, 0.881, 0.535])
    value = np.array([0, 0.0001, 0.001, 0.01, 0.1, 1])
    ax12.plot(np.arange(len(value)), mean, 'x-', markersize=7, color='#7570A0', linewidth=2,
              markeredgecolor=darken_color('#7570A0', 0.2), label='FEO')
    ax12.plot(np.arange(len(value)), base_mean * len(value), markersize=7, color='#F7B76D', linewidth=2,
              markeredgecolor=darken_color('#F7B76D', 0.2), label='Best Baseline')
    ax12.set_xticks(np.arange(len(value)))
    ax12.set_xticklabels(value, fontsize=21)
    ax12.set_ylim(0.2, 1.2)
    ax12.yaxis.set_major_locator(MultipleLocator(0.2))
    ax12.tick_params(axis='y', labelsize=21)
    ax12.set_xlabel('$\ell_2$ loss weight $w_m$', fontsize=21)
    ax12.set_ylabel('Recall@50 (%)', fontsize=21)
    ax12.grid(True, which='major', linestyle='--', linewidth=0.5, color='black', dashes=(20, 10))
    ax12.minorticks_on()
    ax12.tick_params(which='major', direction='in')
    ax12.xaxis.set_ticks_position('both')
    ax12.yaxis.set_ticks_position('both')

    base_mean = [0.787]
    mean = np.array([0.83, 0.846, 0.854, 0.934, 0.51, 0.542])
    value = np.array([0, 0.0001, 0.001, 0.01, 0.1, 1])
    ax13.plot(np.arange(len(value)), mean, 'x-', markersize=7, color='#7570A0', linewidth=2,
              markeredgecolor=darken_color('#7570A0', 0.2), label='FEO')
    ax13.plot(np.arange(len(value)), base_mean * len(value), markersize=7, color='#F7B76D', linewidth=2,
              markeredgecolor=darken_color('#F7B76D', 0.2), label='Best Baseline')
    ax13.set_xticks(np.arange(len(value)))
    ax13.set_xticklabels(value, fontsize=21)
    ax13.set_ylim(0.2, 1.2)
    ax13.yaxis.set_major_locator(MultipleLocator(0.2))
    ax13.tick_params(axis='y', labelsize=21)
    ax13.set_xlabel('Diverse loss weight $w_d$', fontsize=21)
    ax13.set_ylabel('Recall@50 (%)', fontsize=21)
    ax13.grid(True, which='major', linestyle='--', linewidth=0.5, color='black', dashes=(20, 10))
    ax13.minorticks_on()
    ax13.tick_params(which='major', direction='in')
    ax13.xaxis.set_ticks_position('both')
    ax13.yaxis.set_ticks_position('both')

    base_mean = [0.787]
    mean = np.array([0.844, 0.844, 0.854, 0.998, 0.982, 0.92])
    value = np.array([0, 0.8, 0.9, 0.95, 0.99, 1])
    ax21.plot(np.arange(len(value)), mean, 'x-', markersize=7, color='#7570A0', linewidth=2,
              markeredgecolor=darken_color('#7570A0', 0.2), label='FEO')
    ax21.plot(np.arange(len(value)), base_mean * len(value), markersize=7, color='#F7B76D', linewidth=2,
              markeredgecolor=darken_color('#F7B76D', 0.2), label='Best Baseline')
    ax21.set_xticks(np.arange(len(value)))
    ax21.set_xticklabels(value, fontsize=21)
    ax21.set_ylim(0.2, 1.2)
    ax21.yaxis.set_major_locator(MultipleLocator(0.2))
    ax21.tick_params(axis='y', labelsize=21)
    ax21.set_xlabel('Probability penalty coefficient $p$', fontsize=21)
    ax21.set_ylabel('Recall@50 (%)', fontsize=21)
    ax21.grid(True, which='major', linestyle='--', linewidth=0.5, color='black', dashes=(20, 10))
    ax21.minorticks_on()
    ax21.tick_params(which='major', direction='in')
    ax21.xaxis.set_ticks_position('both')
    ax21.yaxis.set_ticks_position('both')

    base_mean = [0.787]
    mean = np.array([0.978, 0.854, 0.561, 0.349, 0.448])
    value = np.array([5, 10, 20, 50, 100])
    ax22.plot(np.arange(len(value)), mean, 'x-', markersize=7, color='#7570A0', linewidth=2,
              markeredgecolor=darken_color('#7570A0', 0.2), label='FEO')
    ax22.plot(np.arange(len(value)), base_mean * len(value), markersize=7, color='#F7B76D', linewidth=2,
              markeredgecolor=darken_color('#F7B76D', 0.2), label='Best Baseline')
    ax22.set_xticks(np.arange(len(value)))
    ax22.set_xticklabels(value, fontsize=21)
    ax22.set_ylim(0.2, 1.2)
    ax22.yaxis.set_major_locator(MultipleLocator(0.2))
    ax22.tick_params(axis='y', labelsize=21)
    ax22.set_xlabel('Fake user generation step $s$', fontsize=21)
    ax22.set_ylabel('Recall@50 (%)', fontsize=21)
    ax22.grid(True, which='major', linestyle='--', linewidth=0.5, color='black', dashes=(20, 10))
    ax22.minorticks_on()
    ax22.tick_params(which='major', direction='in')
    ax22.xaxis.set_ticks_position('both')
    ax22.yaxis.set_ticks_position('both')

    base_mean = [0.787]
    mean = np.array([0.759, 0.854, 0.952, 1.001, 0.961, 0.868])
    value = np.array([0.01, 0.05, 0.1, 0.2, 0.5, 1])
    ax23.plot(np.arange(len(value)), mean, 'x-', markersize=7, color='#7570A0', linewidth=2,
              markeredgecolor=darken_color('#7570A0', 0.2), label='FEO')
    ax23.plot(np.arange(len(value)), base_mean * len(value), markersize=7, color='#F7B76D', linewidth=2,
              markeredgecolor=darken_color('#F7B76D', 0.2), label='Best Baseline')
    ax23.set_xticks(np.arange(len(value)))
    ax23.set_xticklabels(value, fontsize=21)
    ax23.set_ylim(0.2, 1.2)
    ax23.yaxis.set_major_locator(MultipleLocator(0.2))
    ax23.tick_params(axis='y', labelsize=21)
    ax23.set_xlabel('Target hit ratio $h$', fontsize=21)
    ax23.set_ylabel('Recall@50 (%)', fontsize=21)
    ax23.grid(True, which='major', linestyle='--', linewidth=0.5, color='black', dashes=(20, 10))
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

    pdf = PdfPages('hyper-amazon.pdf')
    base_mean = [0.399]
    mean = np.array([0.079, 0.793, 1.692, 0.161, 0.147, 0.135])
    value = np.array([0, 0.0001, 0.001, 0.01, 0.1, 1])
    fig, ((ax11, ax12, ax13), (ax21, ax22, ax23)) = (
        plt.subplots(nrows=2, ncols=3, constrained_layout=True, figsize=(20, 8)))
    ax11.plot(np.arange(len(value)), mean, 'x-', markersize=7, color='#7570A0', linewidth=2,
              markeredgecolor=darken_color('#7570A0', 0.2), label='FEO')
    ax11.plot(np.arange(len(value)), base_mean * len(value), markersize=7, color='#F7B76D', linewidth=2,
              markeredgecolor=darken_color('#F7B76D', 0.2), label='Best Baseline')
    ax11.set_xticks(np.arange(len(value)))
    ax11.set_xticklabels(value, fontsize=21)
    ax11.set_ylim(0., 2.)
    ax11.yaxis.set_major_locator(MultipleLocator(0.4))
    ax11.tick_params(axis='y', labelsize=21)
    ax11.set_xlabel('Promotion loss weight $w_p$', fontsize=21)
    ax11.set_ylabel('Recall@50 (%)', fontsize=21)
    ax11.grid(True, which='major', linestyle='--', linewidth=0.5, color='black', dashes=(20, 10))
    ax11.minorticks_on()
    ax11.tick_params(which='major', direction='in')
    ax11.xaxis.set_ticks_position('both')
    ax11.yaxis.set_ticks_position('both')

    base_mean = [0.399]
    mean = np.array([0.361, 1.287, 1.692, 0.146, 0.146, 0.186])
    value = np.array([0, 0.0001, 0.001, 0.01, 0.1, 1])
    ax12.plot(np.arange(len(value)), mean, 'x-', markersize=7, color='#7570A0', linewidth=2,
              markeredgecolor=darken_color('#7570A0', 0.2), label='FEO')
    ax12.plot(np.arange(len(value)), base_mean * len(value), markersize=7, color='#F7B76D', linewidth=2,
              markeredgecolor=darken_color('#F7B76D', 0.2), label='Best Baseline')
    ax12.set_xticks(np.arange(len(value)))
    ax12.set_xticklabels(value, fontsize=21)
    ax12.set_ylim(0., 2.)
    ax12.yaxis.set_major_locator(MultipleLocator(0.4))
    ax12.tick_params(axis='y', labelsize=21)
    ax12.set_xlabel('$\ell_2$ loss weight $w_m$', fontsize=21)
    ax12.set_ylabel('Recall@50 (%)', fontsize=21)
    ax12.grid(True, which='major', linestyle='--', linewidth=0.5, color='black', dashes=(20, 10))
    ax12.minorticks_on()
    ax12.tick_params(which='major', direction='in')
    ax12.xaxis.set_ticks_position('both')
    ax12.yaxis.set_ticks_position('both')

    base_mean = [0.399]
    mean = np.array([1.366, 1.554, 1.665, 0.196, 0.197, 0.234])
    value = np.array([0, 0.0001, 0.001, 0.01, 0.1, 1])
    ax13.plot(np.arange(len(value)), mean, 'x-', markersize=7, color='#7570A0', linewidth=2,
              markeredgecolor=darken_color('#7570A0', 0.2), label='FEO')
    ax13.plot(np.arange(len(value)), base_mean * len(value), markersize=7, color='#F7B76D', linewidth=2,
              markeredgecolor=darken_color('#F7B76D', 0.2), label='Best Baseline')
    ax13.set_xticks(np.arange(len(value)))
    ax13.set_xticklabels(value, fontsize=21)
    ax13.set_ylim(0., 2.)
    ax13.yaxis.set_major_locator(MultipleLocator(0.4))
    ax13.tick_params(axis='y', labelsize=21)
    ax13.set_xlabel('Diverse loss weight $w_d$', fontsize=21)
    ax13.set_ylabel('Recall@50 (%)', fontsize=21)
    ax13.grid(True, which='major', linestyle='--', linewidth=0.5, color='black', dashes=(20, 10))
    ax13.minorticks_on()
    ax13.tick_params(which='major', direction='in')
    ax13.xaxis.set_ticks_position('both')
    ax13.yaxis.set_ticks_position('both')

    base_mean = [0.399]
    mean = np.array([1.426, 1.365, 1.692, 1.704, 0.806, 0.339])
    value = np.array([0, 0.8, 0.9, 0.95, 0.99, 1])
    ax21.plot(np.arange(len(value)), mean, 'x-', markersize=7, color='#7570A0', linewidth=2,
              markeredgecolor=darken_color('#7570A0', 0.2), label='FEO')
    ax21.plot(np.arange(len(value)), base_mean * len(value), markersize=7, color='#F7B76D', linewidth=2,
              markeredgecolor=darken_color('#F7B76D', 0.2), label='Best Baseline')
    ax21.set_xticks(np.arange(len(value)))
    ax21.set_xticklabels(value, fontsize=21)
    ax21.set_ylim(0., 2.)
    ax21.yaxis.set_major_locator(MultipleLocator(0.4))
    ax21.tick_params(axis='y', labelsize=21)
    ax21.set_xlabel('Probability penalty coefficient $p$', fontsize=21)
    ax21.set_ylabel('Recall@50 (%)', fontsize=21)
    ax21.grid(True, which='major', linestyle='--', linewidth=0.5, color='black', dashes=(20, 10))
    ax21.minorticks_on()
    ax21.tick_params(which='major', direction='in')
    ax21.xaxis.set_ticks_position('both')
    ax21.yaxis.set_ticks_position('both')

    base_mean = [0.399]
    mean = np.array([1.756, 1.692, 0.754, 0.108, 0.478])
    value = np.array([50, 100, 200, 500, 1000])
    ax22.plot(np.arange(len(value)), mean, 'x-', markersize=7, color='#7570A0', linewidth=2,
              markeredgecolor=darken_color('#7570A0', 0.2), label='FEO')
    ax22.plot(np.arange(len(value)), base_mean * len(value), markersize=7, color='#F7B76D', linewidth=2,
              markeredgecolor=darken_color('#F7B76D', 0.2), label='Best Baseline')
    ax22.set_xticks(np.arange(len(value)))
    ax22.set_xticklabels(value, fontsize=21)
    ax22.set_ylim(0., 2.)
    ax22.yaxis.set_major_locator(MultipleLocator(0.4))
    ax22.tick_params(axis='y', labelsize=21)
    ax22.set_xlabel('Fake user generation step $s$', fontsize=21)
    ax22.set_ylabel('Recall@50 (%)', fontsize=21)
    ax22.grid(True, which='major', linestyle='--', linewidth=0.5, color='black', dashes=(20, 10))
    ax22.minorticks_on()
    ax22.tick_params(which='major', direction='in')
    ax22.xaxis.set_ticks_position('both')
    ax22.yaxis.set_ticks_position('both')

    base_mean = [0.399]
    mean = np.array([1.216, 1.692, 1.141, 0.427, 0.274, 0.029])
    value = np.array([0.01, 0.05, 0.1, 0.2, 0.5, 1])
    ax23.plot(np.arange(len(value)), mean, 'x-', markersize=7, color='#7570A0', linewidth=2,
              markeredgecolor=darken_color('#7570A0', 0.2), label='FEO')
    ax23.plot(np.arange(len(value)), base_mean * len(value), markersize=7, color='#F7B76D', linewidth=2,
              markeredgecolor=darken_color('#F7B76D', 0.2), label='Best Baseline')
    ax23.set_xticks(np.arange(len(value)))
    ax23.set_xticklabels(value, fontsize=21)
    ax23.set_ylim(0., 2.)
    ax23.yaxis.set_major_locator(MultipleLocator(0.4))
    ax23.tick_params(axis='y', labelsize=21)
    ax23.set_xlabel('Target hit ratio $h$', fontsize=21)
    ax23.set_ylabel('Recall@50 (%)', fontsize=21)
    ax23.grid(True, which='major', linestyle='--', linewidth=0.5, color='black', dashes=(20, 10))
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