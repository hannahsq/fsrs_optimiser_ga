import fsrs_optimizer as optimizer
import pandas as pd
import numpy as np
from statistics import median
import torch
import subprocess
from platform import python_version
from tqdm import tqdm
import pathlib
import time
import matplotlib.pyplot as plt
import matplotlib
import pickle
import argparse
import torch
import pygad
import math
from functools import reduce
from operator import mul
import os
from copy import deepcopy
from collections.abc import Callable

from sklearn.metrics import root_mean_squared_error, log_loss

filename = "kanji.apkg"
pickle_path = '/home/hannah/.ramdisk/anki_opts'

pid = 0

affinity = os.sched_getaffinity(pid)
print("Process is eligible to run on:", affinity)

affinity_mask = {x - 1 for x in {4, 8}}
# affinity_mask = {x - 1 for x in {2, 6}}

os.sched_setaffinity(pid, affinity_mask)

affinity = os.sched_getaffinity(pid)
print("Process restricted to run on:", affinity)


print("Num Intra-op Threads: {}".format(torch.get_num_threads()))
torch.set_num_threads(2)
print("Num Intra-op Threads: {}".format(torch.get_num_interop_threads()))
torch.set_num_interop_threads(2)
global DEFAULT_WEIGHTS_SET
global test_dataset

global error_df, error_count, error_count_thresh
error_df = None
error_count = -1
error_count_thresh = 2
default_epsilon = 1e-8

DEFAULT_WEIGHTS_SET = [
    [0.5614, 1.2546, 3.5878, 7.9731, 5.1043, 1.1303, 0.823, 0.0465, 1.629, 0.135, 1.0045, 2.132, 0.0839, 0.3204, 1.3547, 0.219, 2.7849],
    #[0.3676, 1.7156, 2.9123, 4.1443, 5.4068, 1.4330, 0.8886, 0.0000, 1.3300, 0.1661, 0.7019, 2.3070, 0.0100, 0.4587, 1.5227, 0.0000, 2.5339],
    #[0.9758, 1.4317, 5.2286, 15.7375, 4.8569, 0.9738, 0.6300, 0.2580, 1.9151, 0.1000, 1.2883, 2.4680, 0.0157, 0.6638, 1.6931, 0.0212, 2.6561],
    # [0.4835, 1.7156, 3.7731, 9.4012, 5.3527, 1.3660, 0.8719, 0.1280, 1.4610, 0.3113, 0.8492, 2.3732, 0.0991, 0.5039, 1.5914, 0.0000, 2.7861],
    # [0.4895, 1.7156, 4.1832, 11.2486, 5.3171, 1.3468, 0.5337, 0.1021, 1.4544, 0.4066, 0.8493, 2.4057, 0.1056, 0.4936, 1.6096, 0.0, 3.0504],
    # [0.4927, 1.7156, 4.4525, 12.7303, 5.3173, 1.3454, 0.5342, 0.1028, 1.453, 0.4076, 0.8478, 2.4054, 0.1049, 0.4925, 1.6091, 0.0, 3.0502],
    #[0.5148, 1.4851, 4.0073, 10.0624, 5.3682, 1.3074, 0.7208, 0.1849, 1.4155, 0.4867, 0.8133, 2.4105, 0.0477, 0.4942, 1.6088, 0.0000, 3.0044],
    # [1.5000, 3.0000, 6.0000, 20.0000, 10.000, 5.0000, 5.0000, 0.5000, 3.0000, 0.8000, 2.5000, 5.0000, 0.2000, 0.9000, 2.0000, 1.0000, 4,0000],
    # [0.1000, 0.7500, 1.2500, 2.0000,  1.0000, 0.1000, 0.1000, 0.0000, 0.0000, 0.1000, 0.0100, 0.5000, 0.0100, 0.0100, 0.0100, 0.0000, 1.0000],
    [0.5004, 1.6703, 4.5355, 12.4605, 5.5217, 1.5509, 1.4384, 0.3599, 1.2380, 0.6754, 1.1363, 3.0350, 0.0185, 0.3688, 1.8420, 0.0000, 3.5423]
]

previous_best_soln = [0.5426, 2.1499, 4.6043, 14.1837, 9.1421, 3.8922, 1.1032, 0.3348, 2.1741, 0.6154, 1.0298, 2.6512, 0.0774, 0.4156, 1.9984, 0.2772, 3.5381]

default_set_weights = [1.0/(2**(i+3)) for i in range(4)]
#default_set_weights = [1.0/(1.2**(i+2)) for i in range(20)]
#default_set_weights = [1.0/(2**(i+1)) for i in range(4)]
#default_set_weights = [0.0725]

n_reps = 3
rep_count = len(default_set_weights)*len(DEFAULT_WEIGHTS_SET)*n_reps

opts = {}

def clamp(x: float, range: tuple[float, 2]):
    return max(range[0], min(x, range[1]))

class Sampler:
    base_dataset: pd.DataFrame
    def __init__(self, base_dataset: pd.DataFrame):
        self.base_dataset = base_dataset.copy()
    
    def get_random_sample(self, frac: float):
        return self.base_dataset.sample(frac=frac, replace=False)
    
class AnkiSampler(Sampler):

    def __init__(self, revlog_path: pathlib.Path):
        super.__init__(self.create_dataset(revlog_path))

    def create_dataset(self, path: pathlib.Path) -> pd.DataFrame:
        def lineToTensor(line: str) -> torch.Tensor:
            ivl = line[0].split(",")
            response = line[1].split(",")
            tensor = torch.zeros(len(response), 2)
            for li, response in enumerate(response):
                tensor[li][0] = int(ivl[li])
                tensor[li][1] = int(response)
            return tensor

        dataset = pd.read_csv(
            path,
            sep="\t",
            index_col=None,
            dtype={"r_history": str, "t_history": str},
        )
        dataset = dataset[
            (dataset["i"] > 1)
            & (dataset["delta_t"] > 0)
            & (dataset["t_history"].str.count(",0") == 0)
        ]
        dataset["tensor"] = dataset.apply(
            lambda x: lineToTensor(list(zip([x["t_history"]], [x["r_history"]]))[0]),
            axis=1,
        )

        dataset["group"] = dataset["r_history"] + dataset["t_history"]
        return dataset

class WHistogram:
    fig: matplotlib.figure.Figure
    axs: list[matplotlib.axes.Axes]
    bins: list[np.ndarray]
    containers: list[matplotlib.container.BarContainer] = []
    bounds: list[tuple[float, 2]]

    def __init__(self, n_bins: int, bounds: list[tuple[float, 2]], data:list[list[float]], pop_size: int):
        plt.ion()
        rows = 3
        cols = 6

        self.bounds = bounds

        self.fig, axs = plt.subplots(rows, cols, sharey=True, tight_layout=True)
        self.axs = [ax for row in axs for ax in row]
        bins = [np.linspace(lb, ub, n_bins) for lb, ub in self.bounds[:-1]]
        self.bins = bins
        data = data
        # bins = [np.linspace(lb, ub, n_bins) for lb, ub in self.bounds[:-1]]
        # self.bins = bins + [[]]
        # data = data + [[]]
        for i, ax in enumerate(self.axs):
            _, _, bar_container = ax.hist(data[i], self.bins[i], lw=1, alpha=0.5)
            self.containers.append(bar_container)
            ax.set_ylim(top=pop_size + 1)  # set safe limit to ensure that all data is visible.

    def update(self, data):
        hist_data = []
        for i in range(len(data)):
            n, _ = np.histogram(data[i], self.bins[i])
            hist_data.append(n)
        for n, bar_container in zip(hist_data, self.containers):
            for count, rect in zip(n, bar_container.patches):
                rect.set_height(count)
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

def create_dataset(path: pathlib.Path) -> pd.DataFrame:
    def lineToTensor(line: str) -> torch.Tensor:
        ivl = line[0].split(",")
        response = line[1].split(",")
        tensor = torch.zeros(len(response), 2)
        for li, response in enumerate(response):
            tensor[li][0] = int(ivl[li])
            tensor[li][1] = int(response)
        return tensor

    dataset = pd.read_csv(
        path,
        sep="\t",
        index_col=None,
        dtype={"r_history": str, "t_history": str},
    )
    dataset = dataset[
        (dataset["i"] > 1)
        & (dataset["delta_t"] > 0)
        & (dataset["t_history"].str.count(",0") == 0)
    ]
    dataset["tensor"] = dataset.apply(
        lambda x: lineToTensor(list(zip([x["t_history"]], [x["r_history"]]))[0]),
        axis=1,
    )

    dataset["group"] = dataset["r_history"] + dataset["t_history"]
    return dataset


def get_new_dataset(df: pd.DataFrame, fraction: float):
    return df.groupby(by=["r_history", "t_history"]).sample(frac=fraction, replace=False)

def maybe_get_new_dataset(dataset: pd.DataFrame, error_count_thresh: int, frac: float):
    global error_df, error_count
    if error_count >= error_count_thresh or error_count < 0:
        error_count = 1
        error_df = get_new_dataset(dataset, frac)
        print("Train set created with {:.2f}% of full dataset ({} entries), next update in {} generation{}".format(round(frac, 4)*100, len(error_df), error_count_thresh, '' if error_count_thresh == 1 else 's'))
    else:
        print("Update in {} generations".format(error_count_thresh - error_count))
        error_count += 1
    return error_df

def rounded_float_list_repr(float_list: list[float], decimals: int = 3):
    return"[{}]".format(", ".join(map( lambda x: "{{:.{}f}}".format(decimals).format(x), float_list)))

def loss_from_error(e: list[float]):
    return e
    return reduce(mul, e)**(1/len(e))

def fitness_from_error(e: list[float], eps: float = default_epsilon) -> list[float]:
    return 1/(loss_from_error(e) + eps)
    return [1/(x + eps) for x in e]

def error_from_fitness(e: list[float], eps: float = default_epsilon) -> list[float]:
    return 1/e - eps
    return [1/x - eps for x in e]

def loss_from_fitness(f: list[float], eps: float = default_epsilon) -> float:
    return loss_from_error(error_from_fitness(f, eps))

def calc_error_from_dataset(dataset: pd.DataFrame, weights: list[float], rmse_weight: float = 1.) -> list[float]:
    def get_rmse_bins(df: pd.DataFrame, bins: int = 20):
        df = df.groupby('bin').agg({"y": ["mean"], "r": ["mean", "count"]})
        return root_mean_squared_error(y_true=df["y", "mean"], y_pred=df["r", "mean"], sample_weight=df["r", "count"])

    def get_bin(x, bins: int = 20):
        return np.round(np.log(np.floor(np.exp(np.log(bins + 1) * x)) + 1) / np.log(bins), 3)

    rmse_weight = min(max(rmse_weight, 0), 0.7)

    col = optimizer.Collection(weights)
    if rmse_weight <= 0.00000000001:
        rmse_bins = 0.01
    else:
        stabilities, difficulties = col.batch_predict(dataset)
        dataset['stability'] = stabilities
        dataset['difficulty'] = difficulties
        dataset['r'] = optimizer.power_forgetting_curve(dataset['delta_t'], dataset['stability'])
        dataset["bin"] = dataset["r"].map(lambda x: get_bin(x, bins = 20))

        rmse_bins = get_rmse_bins(dataset, bins=20)
        # dataset.drop(dataset[(dataset["i"] <= 2)].index)


    # rmse_bins_ratings = []
    # counts = []
    # for last_rating in ('1', "2", '3', "4"):
    #     temp = dataset[dataset["r_history"].str.endswith(last_rating)]
    #     counts.append(len(temp))
    #     rmse_bins_ratings.append(get_rmse_bins(temp, bins=20))
    # rmse_bins_ratings_wavg = sum([rating * count for rating, count in zip(rmse_bins_ratings, counts)])/sum(counts)

    # print("{}, {}".format(rmse_bins, rmse_bins_2))
    if 1 - rmse_weight <= 0.00000000001:
        overall_loss = 0.4
    else:
        overall_loss = float(optimizer.calculate_loss_from_revlog_dataset(optimizer.RevlogDataset(dataset), optimizer.FSRS(weights)))
        if rmse_weight > 0.00000000001:
            overall_loss *= 0.02

    # return [overall_loss, rmse_bins, rmse_bins_ratings_wavg]
    return rmse_bins * rmse_weight + (1 - rmse_weight) * overall_loss
    return [overall_loss, rmse_bins]

def calc_loss_from_dataset(dataset: pd.DataFrame, weights: list[float], eps: float = default_epsilon, rmse_weight: float = 1.) -> float:
    return loss_from_error(calc_error_from_dataset(dataset, weights, rmse_weight=rmse_weight))

def calc_fitness_from_dataset(dataset: pd.DataFrame, weights: list[float], eps: float = default_epsilon, rmse_weight: float = 1.) -> list[float]:
    return fitness_from_error(calc_error_from_dataset(dataset, weights, rmse_weight=rmse_weight), eps)

def calc_loss_from_fn(dataset_fn: Callable[[], float], weights: list[float], rmse_weight: float = 1.):
    return calc_loss_from_dataset(dataset_fn(), weights, rmse_weight=rmse_weight)

def plot_graphs(w: list[float], dataset: pd.DataFrame):
    collection = optimizer.Collection(w)
    stabilities, difficulties = collection.batch_predict(dataset)
    dataset['stability'] = stabilities
    dataset['difficulty'] = difficulties
    dataset['p'] = optimizer.power_forgetting_curve(dataset['delta_t'], dataset['stability'])
    fig1 = plt.figure(figsize=(16, 12))
    optimizer.plot_brier(dataset['p'], dataset['y'], bins=40, ax=fig1.gca())
    fig2 = plt.figure(figsize=(16, 12))
    for last_rating in ("1","2","3","4"):
        subplt = plt.subplot(2, 2, int(last_rating))
        print(f"\nLast rating: {last_rating}")
        optimizer.plot_brier(dataset[dataset['r_history'].str.endswith(last_rating)]['p'], dataset[dataset['r_history'].str.endswith(last_rating)]['y'], ax =subplt, bins=40)
    plt.show()

def main(update: bool, alg: str = 'ga'):
    if alg == 'gd':
        opts = {}
        if update or not pathlib.Path(pickle_path + 'gd').with_suffix('.pickle').exists():
            start = time.time()
            optimizer.Optimizer().anki_extract(filename)

            print('using gradient descent')

            with tqdm(total = rep_count) as pbar:

                def process(optim):
                    opt, weights = optim
                    opt.create_time_series('Australia/Brisbane', '2023-11-01', 4, True)
                    opt.define_model(default_weights = weights[0])
                    opt.pretrain(default_weights=weights[0], def_set_wght=weights[1], verbose=False)
                    opt.train(n_splits=10, n_epoch=30, batch_size=256, lr=0.1, n_reps=n_reps, verbose=False, split_by_time = False)
                    pbar.update(n_reps)

                #default_set_weights = [1.0/(1.2**(i+1)) for i in range(25)]c
                opts = {optimizer.Optimizer(): [weights, dsw, i] for i, weights in enumerate(DEFAULT_WEIGHTS_SET) for dsw in default_set_weights}
                
                for opt in opts.items():
                    process(opt)

                pbar.close()
            
            with pathlib.Path(pickle_path + 'gd').with_suffix('.pickle').open('wb') as pickle_file:
                pickle.dump(opts, pickle_file, pickle.HIGHEST_PROTOCOL)

            print('time taken: {}s'.format(time.time() - start))
        else:
            with pathlib.Path(pickle_path + 'gd').with_suffix('.pickle').open('rb') as pickle_file:
                opts = pickle.load(pickle_file)
        
        median_weights = []
        for opt, weights in opts.items():
            temp = []
            for w in opt.w_unique_sorted:
                temp += [w[0]]
            median_weights.append(list((np.median(np.array(temp[:len(temp)//3]), axis=0))))
        
        rmse_median_weights = []
        dataset = create_dataset(pathlib.Path("./revlog_history.tsv"))

        for weight in median_weights:
            error = calc_error_from_dataset(dataset, weight)

            rmse_median_weights.append((weight, round(float(error), 7)))

        rmse_median_weights.sort(key = lambda x: x[-1])
        [print(', '.join([str(w_n) for w_n in weight[0]])) for weight in rmse_median_weights[:min(10, len(rmse_median_weights))]]

    elif alg == 'ga':
        global fitness_func
        global callback_generation
        global callback_start
        global crossover_func
        global callback_fitness
        global best_eval_loss
        global best_eval_soln
        global ga_instance
        global best_n
        global test_dataset
        global hist_updater

        temp = optimizer.Optimizer()
        temp.create_time_series('Australia/Brisbane', '2023-11-01', 4, True)
        df = create_dataset(pathlib.Path("./revlog_history.tsv"))
        dataset = optimizer.RevlogDataset(df)
        test_dataset = df
        best_eval_loss = 100.
        best_eval_soln = []
        best_n = 10
        
        def get_generation_frac(current: int, total: int) -> float:
            return (current - 10) / (total//20)

        def fitness_func(ga_instance: pygad.GA, solution: list[float], sol_idx: int):
                global error_df, DEFAULT_WEIGHTS_SET
                scaling = DEFAULT_WEIGHTS_SET[0]
                generation_frac: float = get_generation_frac(ga_instance.generations_completed, ga_instance.num_generations)
                weights = [weight*scale for weight,scale in zip(solution, scaling)]
                fitness = calc_fitness_from_dataset(error_df, weights, rmse_weight=generation_frac)
                return fitness

        def update_dataset(range: tuple[float, 2]):
            global error_df, error_count, error_count_thresh
            if error_count >= error_count_thresh or error_count < 0:
                error_count = 1
                rng = np.random.default_rng()
                dataset_fraction = rng.uniform(range[0], range[1])
                error_df = get_new_dataset(test_dataset, dataset_fraction)
                error_count_thresh = int(round(rng.uniform(1, 5),0))
                print("Train set created with {:.2f}% of full dataset ({} entries), next update in {} generation{}".format(round(dataset_fraction, 4)*100, len(error_df), error_count_thresh, '' if error_count_thresh == 1 else 's'))
                return True
            else:
                print("Update in {} generations".format(error_count_thresh - error_count))
                error_count += 1
                return False

        def update_dataset_from_loss_spread(spread: float):
            global error_df
            def get_linear_coeffs(range: tuple[float, 2]):
                return (range[1] - range[0], range[1])
            
            def linear(x: float, coeffs: tuple[float, 2]):
                return coeffs[0] * x + coeffs[1]
            
            def scale_input(x: float, range: tuple[float, 2]):
                range = (1/range[1], range[0])
                return clamp(linear(x, get_linear_coeffs(range)), (0, 1))
            
            def get_clamp_range(range: tuple[float, 2]):
                return (min(range), max(range))
            
            def clipped_linear_from_range(x: float, range: tuple[float, 2]):
                return clamp(linear(x, get_linear_coeffs(range)), get_clamp_range(range))

            def clipped_neg_linear_from_range(x: float, range: tuple[float, 2]):
                return clipped_linear_from_range(x, tuple(reversed(range)))
            
            scaled_spread = scale_input(spread, (0, 0.25))
            low = clipped_neg_linear_from_range(scaled_spread, (0.25, 0.75))
            high = clipped_neg_linear_from_range(scaled_spread, (0.5, 0.85))

            print("Generated new dataset")
            error_df = get_new_dataset(test_dataset, 0.33)
            #update_dataset((low, high))
                
        def callback_generation(ga_instance: pygad.GA):
            global DEFAULT_WEIGHTS_SET
            scaling = DEFAULT_WEIGHTS_SET[0]
            global best_eval_loss
            global best_eval_soln
            global error_df
            global best_n
            global hist_updater
            
            def get_pop_data(population: list[list[float]]):
                return [[population[j][i] for j in range(len(population))] for i in range(len(population[0]))]

            population = [[scale * weight for scale, weight in zip(scaling, pop)] for pop in ga_instance.population]
            solns_and_fitnesses = sorted([(soln, loss_from_fitness(fitness))for soln, fitness in zip(population, ga_instance.last_generation_fitness)], key=lambda x: x[1])
            best_n_sol_fit = sorted(solns_and_fitnesses[:best_n], key=lambda x: x[1], reverse=True)
            worst_n_sol_fit = solns_and_fitnesses[-1 * best_n:]
            
            best_n_train_losses = list(map(lambda y: y[1], best_n_sol_fit))
            worst_n_train_losses = list(map(lambda y: y[1], worst_n_sol_fit))
            gaps_n_best_worst = [worst - best for best, worst in zip(best_n_train_losses, worst_n_train_losses)]
            
            
            generation = ga_instance.generations_completed
            print("\nGeneration = {generation}".format(generation=generation))
            generation_frac: float = get_generation_frac(generation, ga_instance.num_generations)
            print("Generation frac: {}".format(generation_frac))
            
            print("Best train losses this generation: {}".format(rounded_float_list_repr(best_n_train_losses)))
            print("Worst train losses this generation: {}".format(rounded_float_list_repr(worst_n_train_losses)))
            
            print("Best (train) parameters this generation: {}".format(rounded_float_list_repr(best_n_sol_fit[-1][0], 4)))
            print("Worst (train) parameters this generation: {}".format(rounded_float_list_repr(worst_n_sol_fit[-1][0], 4)))
                
            if(generation % 10 == 0):
                solns_and_fitnesses = sorted([(soln, loss, calc_loss_from_dataset(test_dataset, soln, rmse_weight=0.6))for soln, loss in solns_and_fitnesses], key=lambda x: x[2])
                best_n_sol_losses = sorted(solns_and_fitnesses[:best_n], key=lambda x: x[1], reverse=True)
                worst_n_sol_losses = solns_and_fitnesses[-1 * best_n:]
                best_n_eval_losses = list(map(lambda y: y[2], best_n_sol_losses))
                worst_n_eval_losses = list(map(lambda y: y[2], worst_n_sol_losses))
                best_eval_solns = list(map(lambda y: y[0], best_n_sol_losses))
                worst_eval_solns = list(map(lambda y: y[0], worst_n_sol_losses))
                gaps_n_best_worst = [worst - best for best, worst in zip(best_n_eval_losses, worst_n_eval_losses)]
                best_gen_eval_loss = best_n_eval_losses[-1]
                if(best_gen_eval_loss < best_eval_loss):
                    best_eval_loss = best_gen_eval_loss
                    best_eval_soln = best_eval_solns[-1]
                print("Best eval loss ever: {}".format(best_eval_loss))
                print("Best eval losses this generation: {}".format(rounded_float_list_repr(best_n_eval_losses)))
                print("Worst eval losses this generation: {}".format(rounded_float_list_repr(worst_n_eval_losses)))
                print("Gaps: {}".format(rounded_float_list_repr(gaps_n_best_worst)))
                print("Best eval parameters ever: {}".format(rounded_float_list_repr(best_eval_soln, 4)))
                print("Best (eval) parameters this generation: {}".format(rounded_float_list_repr(best_eval_solns[-1], 4)))
                print("Worst (eval) parameters this generation: {}".format(rounded_float_list_repr(worst_eval_solns[-1], 4)))
            else:
                print("Gaps: {}".format(rounded_float_list_repr(gaps_n_best_worst)))

            hist_updater.update(get_pop_data(population) + [[x[1] for x in solns_and_fitnesses]])
            update_dataset_from_loss_spread(1)

        def callback_start(ga_instance: pygad.GA):
            global error_df, test_dataset, error_count, error_count_thresh
            update_dataset_from_loss_spread(1)
            # error_count_thresh = 15
            # error_count = 0
            # error_df = test_dataset
            
            
        def crossover_func(parents, offspring_size, ga_instance: pygad.GA):
            offspring = np.empty(offspring_size, dtype=type(parents[0,0]))
            rng = np.random.default_rng()
            for k in range(offspring_size[0]):
                this_parents = []
                this_parents.append(parents[k % parents.shape[0], :])
                this_parents.append(parents[(k + 1) % parents.shape[0], :])

                rand = rng.uniform(0, 1)
                if rand <= 0.01:
                    parent_to_select_gene_from = [rng.choice([0,1]) for i in range(offspring_size[1])]
                    offspring[k] = [this_parents[parent][i] for i, parent in enumerate(parent_to_select_gene_from)]
                else:
                    gene_weights = [rng.uniform(0,1) for i in range(offspring_size[1])]

                    offspring[k] = [this_parents[0][i]*weight + this_parents[1][i]*(1-weight) for i, weight in enumerate(gene_weights)]

            return offspring
        
        start = time.time()
        optimizer.Optimizer().anki_extract(filename)

        print('using genetic algorithm')

        
        scaling = DEFAULT_WEIGHTS_SET[0]

        num_generations =2000
        sol_per_pop = 100
        best_n = max(np.sqrt(sol_per_pop)//3, 5)
        num_parents_mating = int(round(sol_per_pop * 0.33, 0))
        num_genes = len(DEFAULT_WEIGHTS_SET[0])
        lb = [0.25, 1.8, 4., 10., 1., 0.1, 0.1, 0.01, 0.01, 0.1, 0.01, 0.5, 0.01, 0.01, 0.01, 0.01, 1., 0.]
        ub = [2., 8., 16., 32., 10., 5., 5., 0.5, 3., 0.8, 2.5, 5., 0.2, 0.9, 2., 1., 4., 0.5]
        gene_space_absolute = [
            {'low':0.2, 'high':0.5},
            {'low':1.75, 'high':2.5},
            {'low':4, 'high':5.5},
            {'low':17, 'high':22},

            {'low':5, 'high':8},
            {'low':2.5, 'high':3.5},
            {'low':1.5, 'high':2.5},
            {'low':0.27, 'high':0.33},

            {'low':0.2, 'high':0.8},
            {'low':0.2, 'high':0.6},
            {'low':0.01, 'high':1},
            {'low':0.9, 'high':1.6},

            {'low':0.075, 'high':0.175},
            {'low':0.01, 'high':0.2},
            {'low':0.5, 'high':1},
            {'low':0.2, 'high':0.8},

            {'low':1.5, 'high':3.5},
        ]

        gene_space = [{'low': gene['low']/scale, 'high': gene['high']/scale} for gene, scale in zip(gene_space_absolute, scaling)]

        init_range_low = 0
        init_range_high = 5

        # parent_selection_type = "nsga2"
        parent_selection_type = "rank"
        keep_elitism = 0
        keep_parents = 0
        save_solutions = False
        save_best_solutions = False

        crossover_type = crossover_func

        mutation_type = "adaptive"
        mutation_percent_genes = 6
        mutation_probability = (0.1, 0.01)
        gene_type = np.float32

        bounds = list(zip(lb, ub)) + [(0, 0)]
        hist_updater = WHistogram(100, bounds, [[0 for j in range(18)]for i in range(sol_per_pop)], sol_per_pop)
        
        ga_instance = pygad.GA(
            num_generations=num_generations, 
            num_parents_mating=num_parents_mating,
            sol_per_pop = sol_per_pop,
            num_genes = num_genes,
            init_range_low = init_range_low,
            init_range_high = init_range_high,
            parent_selection_type = parent_selection_type,
            keep_elitism = keep_elitism,
            keep_parents = keep_parents,
            crossover_type = crossover_type,
            mutation_type = mutation_type,
            mutation_percent_genes = mutation_percent_genes,
            mutation_probability = mutation_probability,
            random_mutation_min_val = 0,
            random_mutation_max_val = 0.1,
            gene_space = gene_space,
            save_solutions=save_solutions,
            save_best_solutions=save_best_solutions,
            fitness_func=fitness_func,
            #parallel_processing=['thread', 2],
            on_generation=callback_generation,
            on_start = callback_start,
            # on_fitness = callback_fitness,
            # on_mutation = callback_mutation,
            gene_type=gene_type
        )

        # rng = np.random.default_rng()
        # ga_instance.population[0] = [0.5303, 2.5585, 5.2142, 23.1365, 6.8702, 3.0615, 1.3671, 0.0869, 1.8545, 0.7371, 1.3257, 3.0619, 0.1095, 0.4612, 1.2460, 0.3620, 3.1619]
        # for i in range(sol_per_pop//50):
        #     ga_instance.population[i] =[i + rng.normal(j/40.,j/40.) for j in previous_best_soln]

        if update or not pathlib.Path(pickle_path + 'ga').with_suffix('.pickle').exists():

            ga_instance.run()
            print('time taken: {}s'.format(time.time() - start))

            with pathlib.Path(pickle_path + 'ga').with_suffix('.pickle').open('wb') as pickle_file:

                pickle.dump((ga_instance.population, ga_instance.last_generation_fitness), pickle_file, pickle.HIGHEST_PROTOCOL)

            print('time taken: {}s'.format(time.time() - start))
        else:
            with pathlib.Path(pickle_path + 'ga').with_suffix('.pickle').open('rb') as pickle_file:
                ga_instance.population, ga_instance.last_generation_fitness = pickle.load(pickle_file)

        plot_graphs(ga_instance.best_solution()[0], df)
        ga_instance.plot_fitness()
    
    elif alg == 'ps':
        
        temp = optimizer.Optimizer()
        temp.anki_extract(filename)
        temp.create_time_series('Australia/Brisbane', '2023-11-01', 4, True)
        df = create_dataset(pathlib.Path("./revlog_history.tsv"))

        n_dim = len(previous_best_soln)
        size_pop = 50
        max_iter = 2000
        lb = [0, 1.8, 4, 10, 1, 0.1, 0.1, 0, 0, 0.1, 0.01, 0.5, 0.01, 0.01, 0.01, 0, 1]
        ub = [2, 8, 16, 32, 10, 5, 5, 0.5, 3, 0.8, 2.5, 5, 0.2, 0.9, 2, 1, 4]

        # lb = [0.45, 2.0, 4.5, 11.5, 1., 0.3, 0.2, 0.0, 0, 0.0, 0.7, 1.0, 0.01, 0.1, 0.01, 0.0, 1]
        # ub = [0.65, 3.0, 7.0, 16.0, 10, 5.0, 5.0, 0.5, 3, 0.8, 2.5, 3.5, 0.20, 0.9, 2.00, 1.0, 4]

        global best_ps_params, best_ps_loss, worst_train_eval_diff

        best_ps_loss = 100.
        best_ps_params = None
        n = 4

        def obj_func(swarm: np.ndarray, size_pop: int):
            global best_ps_params, best_ps_loss, worst_train_eval_diff
            dataset = get_new_dataset(df, 0.5)
            losses = [(point, calc_loss_from_dataset(dataset, point)) for point in swarm]
            losses_srt = sorted(losses, key = lambda x: x[1])[:max(size_pop//2, 5)]
            eval_losses = sorted([(point, train_loss, calc_loss_from_dataset(df, point)) for point, train_loss in losses_srt], key=lambda x: x[2])
            gen_best_eval_loss = eval_losses[0][2]
            gen_best_point = eval_losses[0][0]
            if gen_best_eval_loss < best_ps_loss:
                    best_ps_loss = gen_best_eval_loss
                    best_ps_params = gen_best_point
            print('Best eval loss this generation: {}\nWith params:\n{}'.format(gen_best_eval_loss, rounded_float_list_repr(gen_best_point, 4)))
            print('Best loss so far: {}\nWith params:\n{}'.format(best_ps_loss, rounded_float_list_repr(best_ps_params, 4)))
            return np.array([loss for _, loss in losses])

        import pyswarms as ps
        from pyswarms.utils.functions import single_obj as fx

        # Set-up hyperparameters
        options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}

        # Call instance of GlobalBestPSO
        opt = ps.single.GlobalBestPSO(
            n_particles=size_pop,
            dimensions=n_dim,
            bounds = (np.array(lb), np.array(ub)),
            bh_strategy='reflective',
            vh_strategy='invert',
            options=options)

        # Perform optimization
        stats = opt.optimize(lambda x: obj_func(x, size_pop), iters=max_iter)
        
        # pso = PSO(func=lambda x: obj_func(x, size_pop), n_dim=n_dim, pop=size_pop, max_iter=max_iter, lb=lb, ub=ub)
        # pso.run()

        # afsa = AFSA(lambda x: obj_func(x, size_pop), n_dim=n_dim, size_pop=size_pop, max_iter=max_iter, max_try_num=1000, step=5, visual=2.5, delta=0.25)
        # best_x, best_y = afsa.run()
        # print('best_x is ', best_x, 'best_y is', best_y)
        # print('best_x is ', pso.gbest_x, 'best_y is', pso.gbest_y)
        # plt.plot(pso.gbest_y_hist)
        # plt.show()

    #[print(opt.w_unique_sorted[:len(opt.w_unique_sorted)//3][0][0]) for opt, weights in opts.items()]
    #temp = [[np.median(opt.w_unique_sorted[:len(opt.w_unique_sorted)//3][0][0])] for opt, weights in opts.items()]
    
   

        
    # weights_list = sorted([w[0] + weights[-2:] + [w[1]] + [opt] for opt, weights in opts.items() for w in opt.w_unique_sorted], key=lambda x:x[-2])
    # [print(weights[:-3]) for weights in weights_list[-20:]]

    # best_opt = weights_list[-1][-1]
    # dd = best_opt.predict_memory_states()
    # print(dd)
    # figures = best_opt.find_optimal_retention(
    #     deck_size=3000,
    #     learn_span=1000,
    #     max_cost_perday=5400,
    #     max_ivl=36500,
    #     loss_aversion=2.5
    # )
    # print(figures)
    # for i, f in enumerate(figures):
    #     f.savefig(f"find_optimal_retention_{i}.png")
    #     plt.close(f)
    # best_opt.preview(best_opt.optimal_retention)


    # notify = subprocess.Popen(['notify-send', '-a', 'python {}: {}'.format(python_version(),pathlib.Path(sys.argv[0]).resolve().name), 'Script Finished', 'Error: {}\nBest Parameters: {}'.format(rmse_median_weights[1], rmse_median_weights[0])])

    # #notify = subprocess.Popen(['notify-send', '-a', 'python {}: {}'.format(python_version(),pathlib.Path(sys.argv[0]).resolve().name), 'Script Finished'])

    # try:
    #     outs, errs = notify.communicate(timeout=15)
    # except subprocess.TimeoutExpired:
    #     notify.kill()
    #     outs, errs = notify.communicate()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog = 'optimise.py',
        description = 'Optimises FSRS parameters'
    )
    parser.add_argument('--update', '-u', dest='update', action="store_true", required=False, help='Rerun the optimiser instead of using pickled data')
    parser.add_argument('--algorithm', '-a', dest='alg', action="store", required=False, help='Algorithm to use (one of "gd", "ga", "ps")')
    args = parser.parse_args()
    print(args)
    main(args.update, args.alg)