import os

from bokeh.palettes import gray

from conf import conf
from data.registry import create_data_loader
from experiment.utils import load_best_model_exp
from models.utils import retrieve_vars, experiment_file
from skopt.plots import plot_convergence, plot_evaluations, plot_objective
from collections import OrderedDict

from bokeh.layouts import gridplot
from bokeh.models import LogColorMapper
from bokeh.models import Range1d, LinearAxis
from bokeh.plotting import figure, show
from asynch import Callable, invoke_in_process_pool
from matplotlib import pyplot as plt
import numpy as np
import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor
import pandas as pd


class ExperimentVis:
    def __init__(self):
        self.loaded_exp = load_best_model_exp(conf.model)
        self.__dict__.update(self.loaded_exp)
        self.best_model = self.loaded_exp.train_eval_model_factory.load_estimator(self.best_model_dir, self.best_model_params)
        self.data_loader = create_data_loader()
        self.data_loader.load_data()

    def plot_data(self, show=True):
        return self.data_loader.plot_data(show)

    def plot(self):
        plot_convergence(self.res);
        plot_evaluations(self.res);
        _ = plot_objective(self.res);

    def eval_best_model(self, data=["train", "validation", "test"]):
        callables = {'train': Callable(self.loaded_exp.train_eval_model_factory.eval_estimator, self.best_model,
                                                                  self.data_loader.train_x,
                                                                  self.data_loader.train_y),
                     "validation": Callable(self.loaded_exp.train_eval_model_factory.eval_estimator, self.best_model,
                                                               self.data_loader.validation_x,
                                                               self.data_loader.validation_y),
                     "test": Callable(self.loaded_exp.train_eval_model_factory.eval_estimator, self.best_model,
                                                               self.data_loader.test_x,
                                                               self.data_loader.test_y)}
        selected = [callables[name] for name in data]

        return OrderedDict(zip(data,invoke_in_process_pool(len(data),*selected)))

    def prob_data(self):
        return invoke_in_process_pool("prob_data", 3, Callable(self.loaded_exp.train_eval_model_factory.predict_estimator, self.best_model,
                                                self.data_loader.train_x,
                                                self.data_loader.train_y),
                                    Callable(self.loaded_exp.train_eval_model_factory.predict_estimator, self.best_model,
                                             self.data_loader.validation_x,
                                             self.data_loader.validation_y),
                                    Callable(self.loaded_exp.train_eval_model_factory.predict_estimator, self.best_model,
                                             self.data_loader.test_x,
                                             self.data_loader.test_y)
                                      )

    def show_prob_data(self):
        res = invoke_in_process_pool("show_prob_data", 3, Callable(self.loaded_exp.train_eval_model_factory.predict_estimator, self.best_model,
                                                 self.data_loader.train_x,
                                                 self.data_loader.train_y),
                                     Callable(self.loaded_exp.train_eval_model_factory.predict_estimator, self.best_model,
                                              self.data_loader.validation_x,
                                              self.data_loader.validation_y),
                                     Callable(self.loaded_exp.train_eval_model_factory.predict_estimator, self.best_model,
                                              self.data_loader.test_x,
                                              self.data_loader.test_y)
                                     )

        for i in range(self.data_loader.test_y.shape[1]):
            pdf_name='pdf%d' % i
            if pdf_name in res[0]:
                fig, axes = plt.subplots(3, 1, figsize=(12,8))
                axes[0].plot(res[0][pdf_name], alpha=0.3)
                axes[0].set_title("%s - train data" % pdf_name)

                axes[1].plot(res[1][pdf_name], alpha=0.3)
                axes[1].set_title("%s - validation data" % pdf_name)

                axes[2].plot(res[2][pdf_name], alpha=0.3)
                axes[2].set_title("%s - test data" % pdf_name)
                plt.tight_layout()

        fig, axes = plt.subplots(3, 1, figsize=(12,8))

        axes[0].plot(res[0]['log_likelihood'], label="model", alpha=0.3)
        if self.data_loader.can_compute_ll():
            axes[0].plot(self.data_loader.ll(self.data_loader.train_x,self.data_loader.train_y), label="True", alpha=0.3)
        axes[0].set_title("LL - train data")
        axes[0].legend()

        axes[1].plot(res[1]['log_likelihood'], label="model", alpha=0.3)
        if self.data_loader.can_compute_ll():
            axes[1].plot(self.data_loader.ll(self.data_loader.validation_x, self.data_loader.validation_y), label="True", alpha=0.3)
        axes[1].set_title("LL - validation data")
        axes[1].legend()

        axes[2].plot(res[2]['log_likelihood'], label="model", alpha=0.3)
        if self.data_loader.can_compute_ll():
            axes[2].plot(self.data_loader.ll(self.data_loader.test_x, self.data_loader.test_y), label="True", alpha=0.3)
        axes[2].set_title("LL - test data")
        axes[2].legend()
        plt.tight_layout()

        plt.show()

    def show_pdf_heatmap_model(self,paper=False, x_fixed=None, y_fixed=None,x_lim=None, y_lim=None,pdf_percentile_cut_off=None):
        return self.show_pdf_heatmap_compute_pdf_fun(
            lambda x,y: np.exp(invoke_in_process_pool("show_pdf_heatmap_model", 1, Callable(self.loaded_exp.train_eval_model_factory.predict_estimator, self.best_model, x, y))[0]['log_likelihood']), "",
        paper=paper,x_fixed=x_fixed, y_fixed=y_fixed,x_lim=x_lim, y_lim=y_lim,pdf_percentile_cut_off=pdf_percentile_cut_off);

    def show_pdf_heatmap_real(self,paper=False, x_fixed=None, y_fixed=None,x_lim=None, y_lim=None, pdf_percentile_cut_off=None):
        return self.show_pdf_heatmap_compute_pdf_fun(lambda x,y: np.exp(self.data_loader.ll(x, y)), "real pdf heatmap",
                                                     paper=paper,x_fixed=x_fixed, y_fixed=y_fixed,x_lim=x_lim, y_lim=y_lim,
                                                     pdf_percentile_cut_off=pdf_percentile_cut_off);

    def show_pdf_heatmap(self, x_fixed=None, y_fixed=None, x_lim=None, y_lim=None, pdf_percentile_cut_off=None):
        self.show_pdf_heatmap_model(x_fixed=x_fixed, y_fixed=y_fixed,x_lim=x_lim, y_lim=y_lim,pdf_percentile_cut_off=pdf_percentile_cut_off);
        if self.data_loader.can_compute_ll():
            self.show_pdf_heatmap_real(x_fixed=x_fixed, y_fixed=y_fixed,x_lim=x_lim, y_lim=y_lim,pdf_percentile_cut_off=pdf_percentile_cut_off);

    def show_pdf_heatmap_compute_pdf_fun(self, compute_pdf_fun, title, paper = False,x_fixed=None, y_fixed=None,
                                         x_lim=None, y_lim=None,pdf_percentile_cut_off=None):
        dl = self.data_loader

        grid_size=100
        grids = []
        for i, val in enumerate(x_fixed):
            if val is None:
                grids.append(np.linspace(dl.min_x[i], dl.max_x[i], grid_size))

        for i, val in enumerate(y_fixed):
            if val is None:
                grids.append(np.linspace(dl.min_y[i], dl.max_y[i], grid_size))



        mesh = np.meshgrid(*(grids))
        x_min = np.min(mesh[0])
        x_max = np.max(mesh[0])
        y_min = np.min(mesh[1])
        y_max = np.max(mesh[1])

        xs = []
        ys = []
        mesh_id=0
        for i, val in enumerate(x_fixed):
            if val is None:
                xs.append(mesh[mesh_id].reshape(-1,1))
                mesh_id+=1
            else:
                xs.append(np.ones((grid_size*grid_size, 1)) * val)

        for i, val in enumerate(y_fixed):
            if val is None:
                ys.append(mesh[mesh_id].reshape(-1,1))
                mesh_id+=1
            else:
                ys.append(np.ones((grid_size*grid_size, 1)) * val)

        pdf = compute_pdf_fun(np.concatenate(xs, axis=1), np.concatenate(ys, axis=1))

        print("integral: %f"% ( np.sum(pdf)/(grid_size*grid_size)))
        if pdf_percentile_cut_off is None:
            high=np.max(pdf)
        else:
            high=np.percentile(pdf, 95)

        color_mapper = LogColorMapper(palette=gray(256),low=np.min(pdf), high=high)
        # color_mapper = LinearColorMapper(palette=gray(256),low=np.min(pdf), high=high)

        if paper:
            p = figure(x_range=(x_min, x_max),
                       y_range=(y_min, y_max),
                       tooltips=[("x", "$x"), ("y", "$y"), ("value", "@image")],toolbar_location = None)
        else:
            p = figure(x_range=(x_min, x_max),
                       y_range=(y_min, y_max),
                       title=title,
                       tooltips=[("x", "$x"), ("y", "$y"), ("value", "@image")])

        # p.xaxis.axis_label = 'x%d' % x_i
        # p.yaxis.axis_label = 'y%d' % y_i

        p.xaxis.axis_label_text_font_size = "20pt"
        p.yaxis.axis_label_text_font_size = "20pt"

        p.xaxis.major_label_text_font_size="20pt"
        p.yaxis.major_label_text_font_size = "20pt"

        p.image(image=[pdf.reshape(grid_size, grid_size)],
                x=x_min, y=y_min,dw=x_max - x_min,dh=y_max - y_min,color_mapper=color_mapper)

        # color_bar = ColorBar(color_mapper=color_mapper, ticker=LogTicker(),
        #                      label_standoff=12, border_line_color=None, location=(0, 0))

        # p.add_layout(color_bar, 'right')

        show(p);
        return p

    def show_samples(self):
        if self.data_loader.train_x.shape[1] == 1 & self.data_loader.train_y.shape[1] == 1:
            x_grid = np.linspace(self.data_loader.min_x, self.data_loader.max_x,
                                 1000).reshape(-1, 1)

            try:
                samples_res = invoke_in_process_pool("show_samples", 1, Callable(self.loaded_exp.train_eval_model_factory.predict_estimator, self.best_model, x_grid, None))[0]
            except NotImplementedError:
                print("not supported")
                return

            plt.figure(figsize=(16, 8))
            plt.plot(self.data_loader.train_x, self.data_loader.train_y, 'r.',
                     alpha=0.3,
                     label="train");
            plt.plot(x_grid, samples_res["samples"], 'g.', alpha=0.3, label="predicted");
            plt.legend()
            plt.show();
        elif self.data_loader.train_x.shape[1]==0:
            samples_res = invoke_in_process_pool("show_samples", 1, Callable(self.loaded_exp.train_eval_model_factory.predict_estimator, self.best_model, None, None))[0]
            data = samples_res["samples"][:,:]
            df = pd.DataFrame(data)
            fig, ax = plt.subplots(1, 1, figsize=(16, 8));
            axes = pd.plotting.scatter_matrix(df, alpha=0.2, ax=ax, color="black", label="train", diagonal='kde',
                                              density_kwds={'color': 'black'})

            plt.tight_layout()
            plt.savefig('{ROOT}/samples.png')
            # if show:
            #     plt.show();
            return axes
        else:
            print("not supported")

    def show_weights(self):
        vars = invoke_in_process_pool("show_weights", 1, Callable(retrieve_vars, self.best_model_dir))[0]

        for var in vars:
            if len(var[1].shape) <= 1:
                var[1] = var[1].reshape(1, -1)

            p = figure(x_range=(0, var[1].shape[1]), y_range=(0, var[1].shape[0]), title="weights of %s" % var[0],
                       tooltips=[("x", "$x"), ("y", "$y"), ("value", "@image")]);

            p.image(image=[var[1]], x=0, y=0, dw=var[1].shape[1], dh=var[1].shape[0],
                    palette="Spectral11");
            show(p);
            # else:
            #     print("Variable: %s with value %s" % (var[0], var[1]))

    def compute_true_pdf(self, x, y):
        return np.exp(self.data_loader.ll(x,y))

    def show_distribution_slices(self, x_vals=None):
        if self.data_loader.train_x.shape[1] == 1 & self.data_loader.train_y.shape[1] == 1:
            y_data = self.data_loader.train_y
            x_data = self.data_loader.train_x
            y_vals = np.linspace(np.min(y_data), np.max(y_data), 100)
            if x_vals is None:
                x_vals = np.linspace(np.min(x_data), np.max(x_data), 10)
            plots = []

            res = invoke_in_process_pool("show_distribution_slices", 4, *[Callable(self.loaded_exp.train_eval_model_factory.predict_estimator, self.best_model,
                                                                 np.ones((len(y_vals), 1)) * x_val,
                                                                 y_vals.reshape(-1, 1)) for x_val in x_vals])

            if self.data_loader.can_compute_ll():
                true_pdf = invoke_in_process_pool("show_distribution_slices", 8, *[Callable(self.compute_true_pdf, np.ones((len(y_vals), 1)) * x_val,
                                                     y_vals.reshape(-1, 1)) for x_val in x_vals])



            for i, x_val in enumerate(x_vals):
                cdf_val = res[i]['cdf0'] if 'cdf0' in res[i] else None
                pdf_val = np.exp(res[i]['log_likelihood'])

                p = figure(title="pdf(x=%f)" % x_val, x_axis_label='y', y_axis_label='pdf0')
                if cdf_val is not None:
                    p.extra_y_ranges = {"cdf_range": Range1d(start=min(cdf_val.flatten()), end=max(cdf_val.flatten()))}
                    p.add_layout(LinearAxis(y_range_name="cdf_range"), 'right')

                # add a line renderer with legend and line thickness
                p.line(y_vals, pdf_val.flatten(), legend="pdf0", alpha=0.5, color="black")
                if self.data_loader.can_compute_ll():
                    p.line(y_vals, true_pdf[i].flatten(), legend="true", alpha=0.5, color="green")
                if cdf_val is not None:
                    p.line(y_vals, cdf_val.flatten(), legend="cdf0", alpha=0.5, y_range_name="cdf_range", color="blue")
                plots.append(p)

            grid = gridplot(plots, ncols=2)
            show(grid);
        else:
            print("not supported")


def create_compare_all_experiments_notebook(data_sets, models):
    nb = nbf.v4.new_notebook()

    imports = """\
%load_ext autoreload
%load_ext ipycache
%autoreload 2
from experiment.utils import *
from bokeh.plotting import output_notebook
import pandas as pd"""

    flags = """\
data_sets = '{data_sets}'
models = '{models}'
plot=True""".format(data_sets=data_sets, models=models)

    nb['cells'] = [nbf.v4.new_code_cell(imports),
                   nbf.v4.new_code_cell(flags),
                   nbf.v4.new_code_cell("""experiments = load_best_model_exps_all_data_sets()
true_metrics=load_true_metrics_all_data_sets()"""),
                   nbf.v4.new_code_cell("""show_all_experiments_comparison_t_paired(true_metrics, experiments, "best_model_train_ll")"""),
                   nbf.v4.new_code_cell("""show_all_experiments_comparison_t_paired(true_metrics, experiments, "best_model_valid_ll")"""),
                   nbf.v4.new_code_cell("""show_all_experiments_comparison_t_paired(true_metrics, experiments, "best_model_test_ll")"""),
                   ]

    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
    try:
        ep.preprocess(nb, {'metadata': {'path': '.'}})
    except Exception as e:
        print(e)

    nbf.write(nb, 'compare_models.ipynb')

def create_vis_notebook(model, data_set):
    model_file = os.path.join(conf.dir, data_set, experiment_file(model))
    if not os.path.exists(model_file):
        return

    print("create_vis_notebook started: %s/%s" %(model,data_set))
    nb = nbf.v4.new_notebook()

    os.chdir(os.path.join(conf.dir ,data_set))

    imports="""\
%load_ext autoreload
%autoreload 2
from experiment.experiment_vis import ExperimentVis
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.WARN)
from bokeh.plotting import output_notebook
%matplotlib inline"""


    plot_data= """\
exp = ExperimentVis()
exp.plot_data()"""

    nb['cells'] = [nbf.v4.new_code_cell(imports),
                   nbf.v4.new_code_cell("output_notebook()"),
                   nbf.v4.new_code_cell(plot_data),
                   nbf.v4.new_code_cell("exp.eval_best_model()"),
                   nbf.v4.new_code_cell("exp.show_prob_data()"),
                   nbf.v4.new_code_cell("exp.show_pdf_heatmap()"),
                   nbf.v4.new_code_cell("exp.show_distribution_slices()"),
                   nbf.v4.new_code_cell("exp.show_samples()"),
                   nbf.v4.new_code_cell("exp.show_weights()"),
                   ]

    # ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
    # try:
    #     ep.preprocess(nb, {'metadata': {'path': '.'}})
    # except Exception as e: print(e)


    nbf.write(nb, 'vis_{model}.ipynb'.format(model=model))

    print("create_vis_notebook completed: %s/%s" % (model, data_set))

def create_vis_notebooks(models, data_sets):
    invoke_in_process_pool("create_vis_notebooks", 3, *[Callable(create_vis_notebook, model, data_set) for model in models.split(",") for data_set in data_sets.split(",")])

def create_compare_notebook(data_set):
    print("create_compare_notebook started: %s"%data_set)
    nb = nbf.v4.new_notebook()
    try:
        os.chdir(os.path.join(conf.dir,data_set))
    except FileNotFoundError:
        print("create_compare_notebook data set missing: %s" % data_set)
        return

    imports = """\
%load_ext autoreload
%load_ext ipycache
%autoreload 2
from experiment.utils import *
from bokeh.plotting import output_notebook
from data.registry import create_data_loader"""


    nb['cells'] = [nbf.v4.new_code_cell(imports),
                   nbf.v4.new_code_cell("output_notebook()"),
                   nbf.v4.new_code_cell("""experiments = load_best_model_exps()
true_metrics = load_true_metrics()
data_loader = create_data_loader()"""),
                   nbf.v4.new_code_cell("""show_exp_res("best_model_params", experiments)"""),
                   nbf.v4.new_code_cell("""compare_stat("best_model_train_ll",  experiments, true_metrics)"""),
                   nbf.v4.new_code_cell("""compare_stat("best_model_valid_ll",  experiments, true_metrics)"""),
                   nbf.v4.new_code_cell("""compare_stat("best_model_test_ll",  experiments, true_metrics)"""),
                   nbf.v4.new_code_cell("""scatter_models_pp_evals(data_loader, experiments, "best_model_train_ll",true_metrics)"""),
                   nbf.v4.new_code_cell("""scatter_models_pp_evals(data_loader, experiments, "best_model_valid_ll",true_metrics)"""),
                   nbf.v4.new_code_cell("""scatter_models_pp_evals(data_loader, experiments, "best_model_test_ll",true_metrics)"""),
                   ]

    # ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
    # try:
    #     ep.preprocess(nb, {'metadata': {'path': '.'}})
    # except Exception as e:
    #     print(e)

    nbf.write(nb, 'compare_models.ipynb')
    print("create_compare_notebook completed: %s" % data_set)


def create_compare_notebooks(data_sets):
    invoke_in_process_pool("create_compare_notebooks" ,4, *[Callable(create_compare_notebook, data_set) for data_set in data_sets.split(",")])






