import itertools
import os
import pickle
from collections import OrderedDict

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import sem, stats

from models.utils import experiment_file

def load_best_model_exp(name, dir=None):
    model_file = experiment_file(name) if dir is None else os.path.join(dir, experiment_file(name))
    with open(model_file, 'rb') as f:
        loaded = pickle.load(f)
    loaded["name"] = name
    return loaded


def load_best_model_exps(models, dir=None):
    exps = OrderedDict()
    for name in models.split(","):
        try:
            exps[name]=load_best_model_exp(name,dir)
        except FileNotFoundError:
            print("Experiment: %s in dir: %s not found"%(name,dir))

    return exps

def load_best_model_exps_all_data_sets(data_sets):
    experiments = OrderedDict()

    for data_set in data_sets.split(","):
        experiments[data_set]=load_best_model_exps(data_set)

    return experiments

def load_true_metrics(data_set=None):
    loaded = None
    try:
        metrics_file = os.path.join(data_set, 'real_metrics.pkl') if data_set is not None else 'real_metrics.pkl'
        with open(metrics_file, 'rb') as f:
            loaded = pickle.load(f)
    except IOError:
        pass
    return loaded

def load_true_metrics_all_data_sets(data_sets):
    true_metrics = OrderedDict()

    for data_set in data_sets.split(","):
        true_metrics[data_set] = load_true_metrics(data_set)

    return true_metrics


def compare_stat(eval_name, experiments, true_metrics):
    data = pd.DataFrame(index=range(len(experiments[list(experiments.keys())[0]][eval_name])))
    figsize = (15, 6)

    for name, experiment in experiments.items():
        exp_data = pd.DataFrame(
            experiment[eval_name],
            columns=[experiment["name"] + "_" + eval_name])
        data = pd.merge(data, exp_data, left_index=True, right_index=True)

    if true_metrics is not None:
        if eval_name == "best_model_train_ll":
            true_metric_name = 'train_ll'
        elif eval_name == "best_model_valid_ll":
            true_metric_name = 'validation_ll'
        elif eval_name == "best_model_test_ll":
            true_metric_name = 'test_ll'
        else:
            raise ValueError

        data["true"]=true_metrics[true_metric_name].flatten()

    data_diff = pd.DataFrame(
        {col1 + "-" + col2: data[col1] - data[col2] for col1, col2 in itertools.combinations(data.columns.values, 2)})

    data.plot(figsize=figsize)

    ax=data_diff.plot(figsize=figsize)
    ax.axhline(color="black", linewidth=0.5)

    plt.figure(figsize=figsize)
    for name in data.columns.values:
        plt.hist(data[name], alpha=0.5, label=name)
    plt.legend(loc='upper right')
    plt.show()

    plt.figure(figsize=figsize)
    for name in data_diff.columns.values:
        plt.hist(data_diff[name], alpha=0.5, label=name)
    plt.legend(loc='upper right')
    plt.show()

    plt.figure(figsize=figsize)
    data.boxplot()
    plt.figure(figsize=figsize)
    data_diff.boxplot()

def scatter_models_pp_evals(data_loader, experiments, name, true_metrics):

    if true_metrics is not None:
        if name == "best_model_train_ll":
            true_metrics_ll = data_loader.ll(data_loader.train_x, data_loader.train_y).flatten()
        elif name == "best_model_valid_ll":
            true_metrics_ll = data_loader.ll(data_loader.validation_x, data_loader.validation_y).flatten()
        elif name == "best_model_test_ll":
            true_metrics_ll = data_loader.ll(data_loader.test_x, data_loader.test_y).flatten()
        else:
            raise ValueError

        experiments["true"]={'name': 'true', name : true_metrics_ll}

    plt.figure(figsize=(16,8))
    for model1, model2 in itertools.combinations(experiments.values(), 2):
        plt.plot(model1[name], model2[name], 'o', label="{model1} vs {model2}".
                 format(model1=model1["name"], model2=model2["name"]), alpha=0.3)
    plt.xlabel('model_1')
    plt.ylabel('model_2')
    plt.legend()
    plt.title(name)

def show_exp_res(prop, experiments):
    for name, experiment in experiments.items():
        print(
            "Experiment {model_name} with results {prop}".format(model_name=name, prop=experiment[prop]))

def show_all_experiments_comparison(models, true_metrics, experiments, eval_name):
    columns = [(model_name, col) for model_name in models.split(",")  + ['true'] for col in [eval_name, 'SE']]
    column_index = pd.MultiIndex.from_tuples(columns)

    results = pd.DataFrame(columns=column_index)
    for ds_name, ds in experiments.items():
        row = []
        true_metric = true_metrics[ds_name]
        for model_name in models.split(","):
            if model_name in ds and len(ds[model_name][eval_name])>0:
                model = ds[model_name]
                row.append(np.mean(model[eval_name]))
                row.append(sem(model[eval_name].flatten()))
            else:
                row.append(np.nan)
                row.append(np.nan)

        if true_metric is not None:
            true_metric_name = None
            if eval_name == "best_model_train_ll":
                true_metric_name='train_ll'
            elif eval_name == "best_model_valid_ll":
                true_metric_name = 'validation_ll'
            elif eval_name == "best_model_test_ll":
                true_metric_name = 'test_ll'
            else:
                ValueError("not recognized eval_name: %s"%eval_name)

            row.append(np.mean(true_metric[true_metric_name].flatten()))
            row.append(sem(true_metric[true_metric_name].flatten()))

        else :
            row.append(np.nan)
            row.append(np.nan)

        results.loc[len(results)] = row

    results = results.set_index(pd.Index(list(experiments.keys()), name="data_set", inplace=True))

    def highlight_min(s):
        measures = s[:, eval_name]
        best_measure = measures[:-1].max()
        return ['background-color: yellow' if s[i] == best_measure else '' for i in range(len(s))]

    return results.style.apply(highlight_min, axis=1)

def results_all_experiments_comparison_t_paired(models, true_metrics, experiments, eval_name):
    columns = [model_name for model_name in models.split(",") + ['true']]

    results = pd.DataFrame(columns=columns)
    for ds_name, ds in experiments.items():
        row = []
        row_se = []
        true_metric = true_metrics[ds_name]
        for model_name in models.split(","):
            # print("processing {ds_name}, {model_name}".format(ds_name=ds_name,model_name=model_name))
            if model_name in ds and len(ds[model_name][eval_name]) > 0:
                model = ds[model_name]
                row.append(np.mean(model[eval_name]))
                row_se.append(np.std(model[eval_name])/np.sqrt(len(model[eval_name])))
            else:
                row.append(np.nan)
                row_se.append(np.nan)

        if true_metric is not None:
            true_metric_name = None
            if eval_name == "best_model_train_ll":
                true_metric_name = 'train_ll'
            elif eval_name == "best_model_valid_ll":
                true_metric_name = 'validation_ll'
            elif eval_name == "best_model_test_ll":
                true_metric_name = 'test_ll'
            else:
                ValueError("not recognized eval_name: %s" % eval_name)

            row.append(np.mean(true_metric[true_metric_name].flatten()))
            row_se.append(np.std(true_metric[true_metric_name].flatten())/np.sqrt(len(true_metric[true_metric_name].flatten())))

        else:
            row.append(np.nan)
            row_se.append(np.nan)

        results.loc[len(results)] = row
        results.loc[len(results)] = row_se

    row_names = [el for tuple in zip(list(experiments.keys()), [''] * len(experiments.keys())) for el in tuple]
    results = results.set_index(pd.Index(row_names, name="data_set", inplace=True))
    return results


def show_all_experiments_comparison_t_paired(models, true_metrics, experiments, eval_name):
    results = results_all_experiments_comparison_t_paired(models, true_metrics, experiments, eval_name)

    def highlight_min(s):
        if s.name == '':
            return ['']*s.size

        best_measure_idx = s[:-1].idxmax()

        best_mask = [stats.ttest_rel(
            experiments[s.name][best_measure_idx][eval_name].flatten(),
            experiments[s.name][name][eval_name].flatten()).pvalue > 0.05 if name in experiments[
            s.name] else False for name, val in s.iteritems()]
        best_mask = np.logical_or(best_mask, s.keys() == best_measure_idx)

        return ['font-weight: bold' if val else '' for val in best_mask]


    return results.style.apply(highlight_min, axis=1)


def export_to_latext(experiments, df, col_names, row_names, file):
    with open(file, 'w') as file:
        file.write('\\begin{tabular}{l%s}\n' % "".join((['r'] * len(df.columns))))
        file.write('\\toprule\n')
        file.write('{}')
        for col in df.columns.values:
            file.write(' & \makecell{%s}' % col_names[col])
        file.write('\\\\\n')

        file.write('%s ' % df.index.name.replace('_', ' '))
        file.write("".join(['&    '] * len(df.columns)))
        file.write("\\\\\n")

        for i in range(len(df)):
            row = df.iloc[i]

            best_measure_idx = row[:-1].idxmax()
            best_mask = [stats.ttest_rel(
                experiments[row.name][best_measure_idx]['best_model_test_ll'].flatten(),
                experiments[row.name][name]['best_model_test_ll'].flatten()).pvalue > 0.05
                         if name in experiments[row.name] else False for name, val in row.iteritems()]
            best_mask = np.logical_or(best_mask, row.keys() == best_measure_idx)

            file.write('\makecell[l]{%s}' % row_names[row.name])
            for col_i, col in enumerate(df.columns.values):
                if np.isnan(row[col]):
                    val = '&  '
                else:
                    val = '%.3f ' % row[col]
                    if best_mask[col_i]:
                        val = "\\textbf{%s}" % val
                    val = "& %s" % val

                file.write(val)
            file.write('\\\\\n')

        file.write('\\bottomrule\n')
        file.write('\\end{tabular}\n')
