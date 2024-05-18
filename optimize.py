#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 2.1 of
#    the License, or (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with this program. If not, see <http://www.gnu.org/licenses/>.

import copy
import random
import numpy as np
import numpy.typing as npTyping
import numba
import pandas as pd
from pandas import DataFrame, Series
from fsrs_optimizer import (
    Optimizer,
    BatchDataset,
    FSRS,
    power_forgetting_curve,
    lineToTensor as line_to_tensor,
    rmse_matrix
)
import torch
from torch import Tensor
from torch.nn import BCELoss
from pathlib import Path
from collections import deque
from collections.abc import Callable
from multiprocessing import Pipe, Process
from multiprocessing.connection import Connection
from typing import NamedTuple, Optional, TextIO
from deap.base import Fitness
from deap import tools
from pymoo.util.ref_dirs import get_reference_directions
from functools import partial, cache
from statsmodels.nonparametric.smoothers_lowess import lowess
from sklearn.metrics import root_mean_squared_error
import pickle
from operator import attrgetter
from io import StringIO
from datetime import datetime
import lzma
# from typing import TYPE_CHECKING
# if TYPE_CHECKING:
#     from _typeshed import SupportsAdd

torch.autograd.set_detect_anomaly(mode=False, check_nan=False)
torch.autograd.profiler.profile(enabled=False)

@cache
def clamp(x: int|float, range: tuple[int|float,int|float]) -> int|float:
    return min(max(x, range[0]), range[1])

def ser_to_tensor(s: Series) -> Tensor:
    return torch.tensor(data=s.tolist(), dtype=torch.float)

class GAFitness(Fitness):
    def __init__(self, weights) -> None:
        self.weights: list[float] = weights

    def __deepcopy__(self, memo):
        """Replace the basic deepcopy function with a faster one.

        It assumes that the elements in the :attr:`values` tuple are
        immutable and the fitness does not contain any other object
        than :attr:`values` and :attr:`weights`.
        """
        copy_ = self.__class__(self.weights)
        copy_.wvalues = self.wvalues
        return copy_

class GAIndividual:

    lb: list[float] = [0., 1.8,  4., 10.,  1., 0.1, 0.1, 0.00, 0., 0.0, 0.01, 0.5, 0.01, 0.01, 0.01, 0., 1.]
    ub: list[float] = [2., 8.0, 16., 60., 10., 5.0, 5.0, 0.75, 4., 0.8, 3.00, 5.0, 0.20, 0.90, 3.00, 1., 6.]

    # obj_weights: list[float] = [-1.0, -0.5, -0.125, -0.5, -0.125] * 3 + [-1.0]
    obj_weights: list[float] = [-0.5, -1.0, -1.0]

    def __init__(self, lb: list[float]=lb, ub: list[float] = ub) -> None:
        self.weights = [random.uniform(a=a, b=b) for a, b in zip(lb, ub)]
        self._fitness: GAFitness = GAFitness(weights=GAIndividual.obj_weights)

    def __len__(self) -> int:
        return len(self.weights)
    
    def __getitem__(self, index) -> float:
        return self.weights[index]
    
    def __setitem__(self, index, value) -> None:
        self.weights[index] = value

    @property
    def weights(self) -> list[float]:
        return self._weights
    
    @weights.setter
    def weights(self, weights: list[float]) -> None:
        self._weights: list[float] = [clamp(x=w, range=b) for w, b in zip(weights, zip(GAIndividual.lb,GAIndividual.ub))]
        self.n_dim = len(self.weights)

    @property
    def n_dim(self) -> int:
        return self._n_dim
    
    @n_dim.setter
    def n_dim(self, n_dim) -> None:
        self._n_dim: int = n_dim

    @property
    def fitness(self) -> Fitness:
        return self._fitness
    
    @fitness.setter
    def fitness(self, fitness: list[float]) -> None:
        self._fitness.values = fitness
    
    # @fitness.deleter
    # def fitness(self) -> None:
    #     del self._fitness.values

class AnkiSample:
    _bceloss = BCELoss(reduction='none')
    DECAY = -0.5
    FACTOR = 0.9 ** (1 / DECAY) - 1

    def __init__(self, df: DataFrame) -> None:
        self.df = df
        self._dataset = BatchDataset(dataframe=self.df)
        self._metric_fns: list[Callable[[DataFrame], float]] = [
            # self.calc_logloss,
            self.calc_rmse_matrix,
            # self.calc_rmse_bins,
            self.calc_ici,
        ]
    
    @property
    def df(self) -> DataFrame:
        return self._df
    
    @df.setter
    def df(self, df: DataFrame) -> None:
        self._df: DataFrame = df

    @property
    def ind(self) -> GAIndividual:
        return self._ind
    
    @ind.setter
    def ind(self, ind: GAIndividual)-> None:
        self._ind: GAIndividual = ind
        self.calc_DSR()
        self.bin_data()
        self.calculate_metrics()

    @property
    def errors(self) -> list[float]:
        return self._errors

    def calc_DSR(self) -> None:
        def fast_calc_p(t, s):
            @cache
            def calc_p(t_s):
                return (1 + AnkiSample.FACTOR * t_s[0] / t_s[1]) ** AnkiSample.DECAY

            return np.array(map(calc_p, zip(t, s)))
    
        model = FSRS(w=self.ind.weights)
        model.eval()
        # _model: torch.jit.ScriptModule = torch.jit.optimize_for_inference(torch.jit.script(model))
        # _model.eval()
        _model = model
        with torch.no_grad():
            outputs: Tensor
            stabilities: Tensor
            difficulties: Tensor
            outputs, _ = _model(self._dataset.x_train.transpose(0, 1))
            # outputs = _model(self._dataset.x_train.transpose(0, 1))
            stabilities, difficulties = outputs[
                self._dataset.seq_len - 1, torch.arange(end=len(self._dataset))
            ].transpose(0, 1)
        self.df['stability'] = stabilities.tolist()
        self.df['difficulty'] = difficulties.tolist()
        self.df['p'] = fast_calc_p(
            t=self.df['delta_t'].values,
            s=self.df['stability'].values
        )
    
    @classmethod
    def calc_logloss(cls, df: DataFrame) -> float:
        review_losses: list[float] = cls._bceloss.forward(
            input=ser_to_tensor(df['p']), 
            target=ser_to_tensor(df['y'])
        ).tolist()
        # print(review_losses)
        return float(np.mean(review_losses))

    @classmethod
    def calc_ici(cls, df: DataFrame) -> float:
        retention = df['p'].values
        retention_calibrated: npTyping.ArrayLike = lowess(
            endog=df['y'].values,
            exog=retention,
            it=0,
            delta=0.01 * (max(retention) - min(retention)),
            return_sorted=False
        )
        abs_err = np.abs(retention_calibrated - retention)
        # percentiles = np.percentile(abs_err, [0.5, 0.7, 0.9, 0.99], method="median_unbiased")


        # return float(np.mean(percentiles + [np.mean(abs_err)]))
        return float(np.mean(abs_err))

    def bin_data(self):
        def get_bin(x: float) -> int:
            return int(x*100)
        
        @cache
        def get_bin_2(x: float):
            return (
                np.log(np.minimum(np.floor(np.exp(np.log(40 + 1) * x) - 1), 40 - 1) + 1)
                / np.log(40)
            ).round(3)
        
        def get_bin_cached(x: float):
            return get_bin_2(round(x, 3))

        @cache
        def get_count(x):
            return x.count("1")

        @cache
        def get_expensive_bin_nonzero(x, logn, scale, dig):
            x = round(x, 3)
            return round(scale * np.power(logn, np.floor(np.log(x) / np.log(logn))), dig)

        @cache
        def get_expensive_bin(x, logn, scale, dig):
            return get_expensive_bin_nonzero(x, logn, scale, dig) if x != 0 else 0

        self.df["bin"] = self.df['p'].map(get_bin_cached)
        self.df["lapse"] = self.df["r_history"].map(get_count)
        self.df["dt_bin"] = self.df["delta_t"].map(partial(get_expensive_bin_nonzero, logn=3.62, scale=2.48, dig=2))
        self.df["i_bin"] = self.df["i"].map(partial(get_expensive_bin_nonzero, logn=1.89, scale=1.99, dig=0))
        self.df["lapse_bin"] = self.df["lapse"].map(partial(get_expensive_bin, logn=1.73, scale=1.65, dig=0))
    
    @classmethod
    def calc_rmse_bins(cls, df: DataFrame) -> float:
        df = df.groupby(by='bin').agg({"y": ["mean"], 'p': ["mean", "count"]})
        return root_mean_squared_error(
            y_true=df[("y", "mean")],
            y_pred=df[('p', "mean")],
            sample_weight=df[('p', "count")]
        )

    @classmethod
    def calc_rmse_matrix(cls, df: DataFrame) -> float:
        tmp = (
            df.groupby(["dt_bin", "i_bin", "lapse_bin", "bin"])
            .agg({"y": "mean", "p": "mean", "card_id": "count"})
            .reset_index()
        )
        # tmp = (df.pivot_table(columns=["dt_bin", "i_bin", "lapse_bin"], aggfunc={"y": "mean", "p": "mean", "card_id": "count"}).transpose().reset_index())
        # print(df, "\n", tmp_old, "\n", tmp)
        return root_mean_squared_error(tmp["y"], tmp["p"], sample_weight=tmp["card_id"])

    def calculate_metrics(self) -> None:
        pop_error: list[float] = [metric_fn(self.df) for metric_fn in self._metric_fns]
        # pop_error = [(pop_error[0] - 0.38)*10] + pop_error[1:-1] + [pop_error[-1] * 30]
        # temp = self.df.drop(self.df[(self.df["i"] <= 2)].index)
        # groups = temp.groupby(by=temp["r_history"].str[-1])
        # ratings_errors: list[list[float]] = [[metric_fn(group) for metric_fn in self._metric_fns] for _, group in groups]
        # errors: list[float] = [elem for row in zip(*([pop_error] + ratings_errors)) for elem in row]
        # temp = list(np.subtract(errors[:5], 0.21)) + list(np.add(errors[5:10], 0.194)) + list(np.add(errors[10:], 0.2))
        # self._errors: list[float] = errors + [float(np.prod(temp)**(1/(len(temp)))-0.2)]
        # self._errors: list[float] = pop_error + [sum(pop_error)]
        self._errors: list[float] = pop_error + [sum(pop_error)]


class AnkiSampler:
    def __init__(self, df: DataFrame):
        self._df: DataFrame = df

    def resample(self, frac: float = 1.) -> None:
        # self._sample: AnkiSample = AnkiSample(self._df.sample(frac=frac))
        self._sample: AnkiSample = AnkiSample(self._df.groupby(by="group").sample(frac=frac))
    
    @property
    def sample(self):
        return self._sample

class AnkiRevlogConfig(NamedTuple):
    timezone: str
    start_date: str
    day_start_hour: int

class AnkiExtractor:
    def __init__(self, path: Path, config: AnkiRevlogConfig) -> None:
        self.opt = Optimizer()
        self._path: Path = path
        self._config: AnkiRevlogConfig = config

    def extract(self) -> DataFrame:
        self.opt.anki_extract(filename=str(object=self._path))
        self.opt.create_time_series(
            timezone=self._config.timezone,
            revlog_start_date=self._config.start_date,
            next_day_starts_at=self._config.day_start_hour,
            analysis=False
        )

        dataset: DataFrame = pd.read_csv(
            filepath_or_buffer="./revlog_history.tsv",
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
            lambda x: line_to_tensor(list(zip([x["t_history"]], [x["r_history"]]))[0]), # type: ignore
            axis=1,
        )

        dataset["group"] = dataset["r_history"] + dataset["t_history"]

        return dataset
    
class GADemeConfig(NamedTuple):
    worker_id: int
    n_pop: int
    n_gen: int
    n_mig: int
    cx_prob: float
    mx_prob: float
    mig_freq: int
    # select_fn: Callable[[GAPopulation], GAPopulation]
    # mutate_fn: Callable[[GAPopulation], GAPopulation]
    # eval_fn: Callable[[GAPopulation], GAPopulation]
    # crossover_fn: Callable[[GAPopulation], GAPopulation]

class GADeme(list):
    _n_obj: int = len(GAIndividual.obj_weights)
    _pickle_path: Path = Path(f'selector_{_n_obj}.pickle')
    _pickle_archive_path: Path = Path(f'selector_{_n_obj}.pickle.xz')
    _cx_fn = partial(
        tools.cxSimulatedBinaryBounded,
        low=GAIndividual.lb,
        up=GAIndividual.ub,
        eta=30.0
    )
    _mut_fn = partial(
        tools.mutPolynomialBounded,
        low=GAIndividual.lb,
        up=GAIndividual.ub,
        eta=20.0
    )
    
    def __init__(
            self,
            *,
            config: GADemeConfig,
            pipein: Connection,
            pipeout: Connection,
            sampler: AnkiSampler,
            print_fn: Callable,
            dump_file: TextIO
        ) -> None:
        self.config: GADemeConfig = config
        self.population: list[GAIndividual] = self._gen_seed_population(n_pop=self.config.n_pop)
        self._sampler: AnkiSampler = sampler
        self.pipe_in = pipein
        self.pipe_out = pipeout
        self._last_migrated = 0
        self._print_fn = print_fn
        self._dump_file=dump_file
        if not hasattr(GADeme, '_selector'):
            if GADeme._pickle_archive_path.exists():
                with lzma.open(GADeme._pickle_archive_path, mode='rb') as pickle_file:
                    GADeme._selector: tools.selNSGA3WithMemory = pickle.load(file=pickle_file)
            elif GADeme._pickle_path.exists():
                with GADeme._pickle_path.open(mode='rb') as pickle_file:
                    GADeme._selector: tools.selNSGA3WithMemory = pickle.load(file=pickle_file)
                if not GADeme._pickle_archive_path.exists():
                    with lzma.open(GADeme._pickle_archive_path, mode='wb') as pickle_file:
                        pickle.dump(obj=GADeme._selector, file=pickle_file, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                GADeme._selector = tools.selNSGA3WithMemory(
                    ref_points=get_reference_directions('layer-energy', GADeme._n_obj, [3,2,1,1,0])
                )
                with lzma.open(GADeme._pickle_archive_path, mode='wb', preset= 9 | lzma.PRESET_EXTREME) as pickle_file:
                    pickle.dump(obj=GADeme._selector, file=pickle_file, protocol=pickle.HIGHEST_PROTOCOL)

    def _print_gen_stats(self, generation: int) -> None:
        def format_stats_list(stats: list[tuple[int, float]], idx: int) -> str:
            idx_to_stat_name = {0: " 0%:", 1: "16%:", 2: "50%:", 3: "84%:", 5: "100%:"}
            ret: str = str("")
            stats_strs: list[str] = ["{:7.3g}".format(stat[1]) for stat in stats]
            step = 5
            temp = idx_to_stat_name[idx] + " |".join([" ".join(stats_strs[i:i + step]) for i in range(0, len(stats_strs), step)])
            return f"{temp: <144}"

        def get_gen_stats(data: list[list[float]]) -> list[list[tuple[int, float]]]:

            def get_pctiles(data:list[float], pctiles: list[float]) -> list[tuple[int, float]]:

                p_idxs:list[int] = [round((len(data) - 1) * val) for val in pctiles]
                prt = np.partition(data, p_idxs)

                return [(idx, prt[idx]) for idx in p_idxs]
            
            data_t = list(zip(*data))
            return list(zip(*[get_pctiles(data=row, pctiles=[0, 0.159, 0.50, 0.841]) for row in data_t]))
        
        def format_gen_stats(attr_name: str) -> tuple[list[list[tuple[int, float]]], list[str]]:
            stats_attr_getter = attrgetter(attr_name)
            stats_list: list[list[tuple[int, float]]] = get_gen_stats(data=[stats_attr_getter(pop) for pop in self.population])
            stats_list_strs = [format_stats_list(stats=stats, idx=i) for i, stats in enumerate(stats_list)]
            return (stats_list, ["    ".join(stats_list_strs[i:i + 2]) for i in range(0, len(stats_list_strs), 2)])


        fit_stats, fit_stats_strs = format_gen_stats('fitness.values')
        pop_stats, pop_stats_strs = format_gen_stats('weights')

        best_idxs = [self.population[idx].weights for idx in next(zip(*fit_stats[0]))]

        pop_list = [" ".join(["{:9.5g}".format(weight) for weight in list(pop.weights) + list(pop.fitness.values)]) + '\n' for pop in sorted(self.population, key=lambda x:x.fitness.values[-1])]
        self._dump_file.writelines(pop_list)

        stats_str = "\n".join([
            f"Worker {self.config.worker_id}: Generation: {generation}, Timestamp: {datetime.now().astimezone().isoformat()}",
            "Fitness Stats:",
            *fit_stats_strs,
            "Population Stats:",
            *pop_stats_strs
        ])
        self._print_fn(stats_str)

        if generation >= self.config.n_gen - 1:
            with Path(f"dumps/deme_{self.config.worker_id}.pickle").open('wb') as pickle_file:
                pickle.dump(self.population, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)

    def evaluate(self) -> None:
        for ind in self.population:
            if not ind.fitness.valid:
                self._sampler.sample.ind = ind
                ind.fitness = self._sampler.sample.errors
        for ind in self.offspring:
            if not ind.fitness.valid:
                self._sampler.sample.ind = ind
                ind.fitness = self._sampler.sample.errors
    
    def select(self) -> None:
        post_cxmx_pop: list[GAIndividual] = self.population+self.offspring
        self.population: list[GAIndividual] = self._selector(post_cxmx_pop, len(self.population))
        pass

    def mutate(self) -> None:
        for i in range(len(self.offspring)):
            if random.random() < self.config.mx_prob:
                self.offspring[i], = GADeme._mut_fn(self.offspring[i], indpb = 1/len(self.offspring[i]))
                del self.offspring[i].fitness.values

    def crossover(self) -> None:
        offspring: list[GAIndividual] = []
        for i in range(4):
            offspring += copy.deepcopy(self.population)
        
        # Apply crossover and mutation on the offspring
        for i in range(1, len(offspring), 2):
            if random.random() < self.config.cx_prob:
                offspring[i - 1], offspring[i] = GADeme._cx_fn(
                    ind1=offspring[i - 1],
                    ind2=offspring[i],
                )
                del offspring[i - 1].fitness.values, offspring[i].fitness.values

        self.offspring: list[GAIndividual] = offspring

    def migrate(self) -> None:
        emigrants: list[GAIndividual] = sorted(self.population, key=lambda x: x.fitness, reverse=True)[:self.config.n_mig]
        replace_idxs: list[int] = [self.population.index(emigrant) for emigrant in emigrants]
        
        self.pipe_out.send(obj=emigrants)
        immigrant_pop: list[GAIndividual] = self.pipe_in.recv()
        
        for idx, immigrant in zip(replace_idxs, immigrant_pop):
            self.population[idx] = immigrant

    def generation(self) -> bool:
        self.crossover()
        self.mutate()
        if self._last_migrated >= self.config.mig_freq:
            self.migrate()
        self.evaluate()
        self.select()
        #TODO implement early stopping
        return False

    def begin_evolution(self) -> None:
        self._sampler.resample(frac=1)
        for i in range(self.config.n_gen):
            stop: bool = self.generation()
            self._print_gen_stats(generation=i)
            if stop == True:
                break
        # TODO: print final stats

    def _gen_seed_population(self, n_pop: int) -> list[GAIndividual]:
        seed_pop: list[GAIndividual] = []
        for _ in range(n_pop):
            seed_pop.append(GAIndividual())
        return seed_pop
    
    @property
    def pipe_in(self) -> Connection:
        return self._pipe_in
    
    @pipe_in.setter
    def pipe_in(self, pipe_in: Connection) -> None:
        self._pipe_in: Connection = pipe_in
    
    @pipe_in.deleter
    def pipe_in(self) -> None:
        del self._pipe_in

    @property
    def pipe_out(self) -> Connection:
        return self._pipe_out
    
    @pipe_out.setter
    def pipe_out(self, pipe_out: Connection) -> None:
        self._pipe_out: Connection = pipe_out
    
    @pipe_out.deleter
    def pipe_out(self) -> None:
        del self._pipe_out

def worker_proc(
        procid: int,
        pipein: Connection,
        pipeout: Connection,
        spipe: Connection,
        sampler: AnkiSampler,
        n_gen: int,
        seed=None
) -> None:
    random.seed(a=seed)
    # import cProfile
    # pr = cProfile.Profile()
    # pr.enable()
    def print_to_stats_proc(*args, **kwargs) -> None:
        with StringIO() as output:
            print(*args, file=output, **kwargs)
            contents = output.getvalue()
        spipe.send(obj=contents)

    deme_config = GADemeConfig(
        worker_id = procid,
        n_pop = 200,
        n_gen = n_gen,
        n_mig = 40,
        cx_prob = 1.0,
        mx_prob = 0.1,
        mig_freq = 5
    )
    with Path(f"dumps/pop_dump_{procid}.txt").open('w') as dump_file:
        deme = GADeme(config=deme_config, pipein=pipein, pipeout=pipeout, sampler=sampler, print_fn=print_to_stats_proc, dump_file=dump_file)
        deme.begin_evolution()
    # pr.disable()

    # pr.dump_stats(f"worker_{procid}.bin")

def stats_proc(pipes_in, n_gen) -> None:
    for gen in range(n_gen):
        for pipe_in in pipes_in:
            stats: str = pipe_in.recv()
            print(stats, end = '')

if __name__ == "__main__":
    random.seed(64)
    
    NUM_DEMES = 5
    NUM_GENERATIONS = 400
    
    w2w_pipes: list[tuple[Connection, Connection]] = [Pipe(duplex=False) for _ in range(NUM_DEMES)]
    w2w_pipes_in = deque(iterable=(p[0] for p in w2w_pipes))
    w2w_pipes_out = deque(iterable=(p[1] for p in w2w_pipes))
    w2w_pipes_in.rotate(1)
    w2w_pipes_out.rotate(-1)
    
    w2s_pipes: list[tuple[Connection, Connection]] = [Pipe(duplex=False) for _ in range(NUM_DEMES)]
    w2s_pipes_in = deque(iterable=(p[0] for p in w2w_pipes))
    w2s_pipes_out = deque(iterable=(p[1] for p in w2w_pipes))

    extractor_config = AnkiRevlogConfig(
        timezone='Australia/Brisbane',
        start_date='2023-11-01',
        day_start_hour=4
    )
    extractor = AnkiExtractor(path=Path('kanji.apkg'), config=extractor_config)
    sampler = AnkiSampler(df=extractor.extract())
    
    processes: list[Process] = [Process(target=stats_proc, args=(w2s_pipes_in, NUM_GENERATIONS))] + [Process(target=worker_proc, args=(i, ipipe, opipe, spipe, copy.deepcopy(sampler), NUM_GENERATIONS, random.random())) for i, (ipipe, opipe, spipe) in enumerate(zip(w2w_pipes_in, w2w_pipes_out, w2s_pipes_out))]

    for proc in processes:
        proc.start()
    
    for proc in processes:
        proc.join()

    # import pstats, io
    # from pstats import SortKey

    # s = io.StringIO()
    # sortby = SortKey.CUMULATIVE
    # bin_files = [f"worker_{i}.bin" for i in range(NUM_DEMES)]
    # ps = pstats.Stats(*bin_files, stream=s).strip_dirs().sort_stats(sortby)
    # # ps.print_stats(0.1)
    # # ps.print_callers(".*evaluate.*")
    # ps.print_callees(".*rmse_.*")
    # print(s.getvalue())