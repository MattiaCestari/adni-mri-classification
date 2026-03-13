import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
import yaml
import numpy as np
import itertools

from sklearn.metrics import confusion_matrix, classification_report, balanced_accuracy_score, f1_score


def _compute_labels(labels, values, pct=True, n=True):
        def inner(count):
            total = sum(values)
            perc = float(count/total)*100
            if (pct and n):
                return f'{perc:.0f}%\n({count})'
            elif pct:
                return f'{perc:.0f}%'
            else:
                return f'({count})'

        if labels is not None:
            return [ f"{labels[i]}\n{inner(values[i])}" for i in range(len(labels))]
        else:
            return [ f"{inner(values[i])}" for i in range(len(values))] 


def diagnosis_split_graph(splits, samples, ax):

    a = samples.groupby("label")["label"].count()
    b = samples.iloc[ splits["test_idx"], :].groupby("label")["label"].count()
    vals = np.array([a-b, b]).T
        
    size = 0.3

    tab20c = plt.color_sequences["tab20c"]
    inner_colors = [tab20c[i] for i in [1, 5, 9]]
    outer_colors = [tab20c[i] for i in [2, 3, 6, 7, 10, 11]]

    ax.pie( [1], labels=[len(samples)], labeldistance=0, colors=["white"], textprops = dict(rotation_mode = 'anchor', va='center', ha='center'))

    ax.pie(vals.sum(axis=1), startangle=90, radius=1-size, colors=inner_colors,
        wedgeprops=dict(edgecolor='w', width=0.4), labels=_compute_labels(["CN","MCI","AD"], vals.sum(axis=1)), 
        labeldistance=0.7, textprops = dict(rotation_mode = 'anchor', va='center', ha='center'))

    ax.pie(vals.flatten(), startangle=90,radius=1, colors=outer_colors,
        wedgeprops=dict(width=size, edgecolor='w'), 
        labels=_compute_labels(["train/val","test"]*3, vals.flatten(),pct=False), 
        labeldistance=1, textprops = dict(rotation_mode = 'anchor', va='center', ha='center'))

    ax.set(aspect="equal", title='')
    ax.set_title("Diagnosis and splits distribution")
    
def modalities_graph(samples, modalities, ax):

    names = []
    counts = []

    diagnoses = []

    for r in range(len(modalities), 0, -1):
        for combination in itertools.combinations(modalities, r):

            selected = samples[ samples[ list(combination) ].notna().all(axis=1) ].index.to_list()
            count = len(selected)

            if count > 0:
                names.append( ", ".join(list(combination)) )
                counts.append(count)
                diagnoses += list(samples[ samples.index.isin(selected)].groupby("label")["label"].count())
                
                samples = samples[ ~samples.index.isin(selected)]

    tab20c = plt.color_sequences["tab20c"]
    tab20b = plt.color_sequences["tab20b"]

    colors_inner = [ tab20c[i] for i in [1,5,9]]*len(names)
    colors_outer = [ tab20b[i*4+3] for i in range(len(names))]

    ax.pie( diagnoses, radius=0.7, startangle=90,  wedgeprops=dict(width=0.3, edgecolor='w'), colors=colors_inner)
    
    ax.pie( counts, labels=_compute_labels(names, counts), startangle=90,radius=1,
        wedgeprops=dict(width=0.3, edgecolor='w'), labeldistance=1, 
        textprops = dict(rotation_mode = 'anchor', va='center', ha='center'),
        colors=colors_outer)
    
    ax.set_title("Modalities distribution")
    ax.legend(["CN","MCI","AD"])

def average_confusion_matrix(predictions, ax): 

    conf_matrices = []

    for p in predictions: 

        y_pred = p["pred"]
        y_true = p["target"]

        conf_matrices.append( confusion_matrix(y_true, y_pred, normalize="true")*100 )

    conf_matrices = np.stack(conf_matrices)
    conf_matrix = pd.DataFrame(conf_matrices.mean(axis=0))

    lbls = ["CN", "MCI", "AD"]
    conf_matrix.index = lbls
    conf_matrix.columns = lbls

    annot = conf_matrix.map(lambda x: f"{x:.1f}%")
    cmap = sn.light_palette('seagreen', as_cmap=True)
    sn.heatmap(
        conf_matrix,
        annot=annot,
        fmt="",
        cmap=cmap,
        ax=ax,
        cbar_kws={"label": "%"}
    )

    ax.set_title("Confusion Matrix (%)")



def average_classification_report(predictions, ax): 

    class_reports = []
    b_accs = []
    f1_scores = []

    for p in predictions: 
        y_pred = p["pred"]
        y_true = p["target"]

        class_reports.append( pd.DataFrame(classification_report(y_true, y_pred, digits=4, output_dict=True)).T )
        b_accs.append( balanced_accuracy_score(y_true, y_pred) )
        f1_scores.append( f1_score(y_true, y_pred, average="macro"))

    b_accs = np.array(b_accs)
    f1_scores = np.array(f1_scores)

    class_report = sum(class_reports)/len(class_reports)
    class_report = class_report.round(4)

    # draw table in bottom-right
    ax.axis("off")
    ax.set_title("Classification report")

    tbl = ax.table(
        cellText=class_report.values,
        rowLabels=class_report.index,
        colLabels=class_report.columns,
        cellLoc="center",
        loc="center"
    )

    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.2)   # widen a bit, increase row height

    # add balanced accuracy text below
    ax.text(
        0.5, 0.1,
        f"Balanced Accuracy: {b_accs.mean(axis=0):.4f} ± {b_accs.std(axis=0):.4f}",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        transform=ax.transAxes
    )

    # add f1 score text below
    ax.text(
        0.5, 0.2,
        f"F1-score: {f1_scores.mean(axis=0):.4f} ± {f1_scores.std(axis=0):.4f}",
        ha="center",
        fontsize=12,
        fontweight="bold",
        transform=ax.transAxes
    )

    n_folds = len(predictions)

    ax.text(
        0.5, 0.03,
        f"{n_folds}-fold cross-validation",
        ha="center",
        fontsize=11,
        transform=ax.transAxes
    )

def experiment_report(path, folds=None, title=None): 

    # Read config.yaml
    with open( os.path.join(path, "config.yaml"), "r") as f:
        config = yaml.full_load(f)

    # Read splits 
    with open( os.path.join(path, "indices.json"), "r") as f:
        splits = json.load(f)

    # Read samples
    samples = pd.read_csv( os.path.join(path, "samples.csv"))
    samples.reset_index()

    # Get modalities
    modalities = config["datamodule"]["args"]["data"]["modalities"]
    modalities = list(modalities.keys())

    # Get predictions 
    predictions = []
    for fold in range(5):

        try:
            predictions.append(  pd.read_csv( os.path.join(path, f"fold_{fold}", "predictions.csv")) )
        except Exception as e:
            print(e)
            continue 

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    if title is not None: 
        fig.suptitle(title)

    diagnosis_split_graph(splits, samples, axes[0,0])
    modalities_graph(samples, modalities, axes[0,1])
    average_classification_report( predictions, axes[1,1])
    average_confusion_matrix( predictions, axes[1,0])

    fig.tight_layout()
    fig.show()
    