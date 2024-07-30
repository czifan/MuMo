from utils.utils import ColorEncoder
from sklearn.metrics import roc_curve, auc
from matplotlib import pyplot as plt
from numpy import interp
import numpy as np
import itertools
from matplotlib import rcParams
import pandas as pd
import seaborn as sn

'''
This file defines functions for plotting ROC curves and confusion matrices
'''

rcParams['font.family'] = 'monospace'
rcParams['font.size'] = 12

font_main_axis = {
    'weight': 'bold',
    'size': 12
}

LINE_WIDTH = 1.5

MICRO_COLOR = 'k'  # (255/255.0, 127/255.0, 0/255.0)
MACRO_COLOR = 'k'  # (255/255.0,255/255.0,51/255.0)
MICRO_LINE_STYLE = 'dashed'
MACRO_LINE_STYLE = 'solid'

CLASS_LINE_WIDTH = 2

GRID_COLOR = (204 / 255.0, 204 / 255.0, 204 / 255.0)
GRID_LINE_WIDTH = 0.25
GRID_LINE_STYLE = ':'


def plot_roc(ground_truth, pred_probs, n_classes, save_loc='./', file_name='dummy',
             class_names=None, dataset_name='bbwsi'):
    class_colors, class_linestyles = ColorEncoder().get_colors(dataset_name=dataset_name)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    # compute ROC curve class-wise
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(ground_truth[:, i], pred_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # COMPUTE MICRO-AVERAGE ROC CURVE AND ROC AREA
    fpr["micro"], tpr["micro"], _ = roc_curve(ground_truth.ravel(), pred_probs.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # COMPUTE MACRO-AVERAGE ROC CURVE AND ROC AREA

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # PLOT the curves
    micro_label = 'Micro avg. (AUC={0:0.2f})'.format(roc_auc["micro"])
    plt.plot(fpr["micro"], tpr["micro"], label=micro_label, color=MICRO_COLOR,
             linestyle=MICRO_LINE_STYLE, linewidth=LINE_WIDTH)

    macro_label = 'Macro avg. (AUC={0:0.2f})'.format(roc_auc["macro"])
    plt.plot(fpr["macro"], tpr["macro"], label=macro_label, color=MACRO_COLOR,
             linestyle=MACRO_LINE_STYLE, linewidth=LINE_WIDTH)

    if class_names is not None:
        assert len(class_names) == n_classes
        for i, c_name in enumerate(class_names):
            label = "{0} (AUC={1:0.2f})".format(c_name, roc_auc[i])
            plt.plot(fpr[i], tpr[i], color=class_colors[i],
                     lw=CLASS_LINE_WIDTH, label=label, linestyle=class_linestyles[i])
    else:
        for i, color in zip(range(n_classes), class_colors):
            label = 'Class {0} (AUC={1:0.2f})'.format(i, roc_auc[i])
            plt.plot(fpr[i], tpr[i], color=color, lw=CLASS_LINE_WIDTH,
                     label=label, linestyle=class_linestyles[i])

    plt.plot([0, 1], [0, 1], 'tab:gray', linestyle='--', linewidth=1)
    plt.grid(color=GRID_COLOR, linestyle=GRID_LINE_STYLE, linewidth=GRID_LINE_WIDTH)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontdict=font_main_axis)
    plt.ylabel('True Positive Rate', fontdict=font_main_axis)
    plt.legend(edgecolor='black', loc="best")
    # plt.tight_layout()
    plt.savefig('{}/{}.pdf'.format(save_loc, file_name), dpi=300, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(cmat_array, class_names=None, save_loc='./', file_name='demo'):
    class_names = range(cmat_array.shape[0]) if class_names is None else class_names
    cmat_array = cmat_array / cmat_array.astype(np.float).sum(axis=1)[:, np.newaxis]

    df_cm = pd.DataFrame(cmat_array, columns=class_names, index=class_names)
    sn.heatmap(df_cm, cmap="Blues",
               xticklabels=class_names,
               annot=True,
               annot_kws=font_main_axis,
               square=True)

    plt.yticks(np.arange(len(class_names)) + 0.5, class_names, va="center")
    plt.ylim([len(class_names), 0])
    plt.ylabel('True label', fontdict=font_main_axis)
    plt.xlabel('Predicted label', fontdict=font_main_axis)
    plt.savefig('{}/{}.pdf'.format(save_loc, file_name), dpi=300, bbox_inches='tight')
    plt.close()

