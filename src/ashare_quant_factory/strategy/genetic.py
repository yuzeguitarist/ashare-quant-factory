from __future__ import annotations

import datetime as dt
import json
import math
import multiprocessing as mp
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from rich.console import Console

from .backtester import BacktestMetrics, CostModel, Genome, backtest

console = Console()

# ---- Discrete search space (fast + cache-friendly) ----
FAST_MA = [5, 7, 10, 12, 15, 20, 25, 30]
SLOW_MA = [30, 40, 50, 60, 80, 100, 120, 150, 200]
RSI_PERIOD = [7, 14, 21]
RSI_BUY = [25, 30, 35, 40]
RSI_SELL = [60, 65, 70, 75]
ATR_PERIOD = [10, 14, 21]
STOP_ATR = [1.0, 1.5, 2.0, 2.5, 3.0]
TAKE_ATR = [2.0, 3.0, 4.0, 5.0, 6.0]

# Worker globals for fork-based multiprocessing (Linux sweet spot)
_WORKER_DATASET: dict[str, pd.DataFrame] | None = None
_WORKER_COSTS: CostModel | None = None


def _set_worker_context(dataset: dict[str, pd.DataFrame], costs: CostModel) -> None:
    global _WORKER_DATASET, _WORKER_COSTS
    _WORKER_DATASET = dataset
    _WORKER_COSTS = costs


def random_genome(rng: random.Random) -> Genome:
    fast = rng.choice(FAST_MA)
    slow = rng.choice([x for x in SLOW_MA if x > fast])
    rsi_p = rng.choice(RSI_PERIOD)
    rsi_buy = rng.choice(RSI_BUY)
    rsi_sell = rng.choice([x for x in RSI_SELL if x > rsi_buy])
    atr_p = rng.choice(ATR_PERIOD)
    stop = rng.choice(STOP_ATR)
    take = rng.choice(TAKE_ATR)
    return Genome(
        fast_ma=fast,
        slow_ma=slow,
        rsi_period=rsi_p,
        rsi_buy=rsi_buy,
        rsi_sell=rsi_sell,
        atr_period=atr_p,
        stop_loss_atr=stop,
        take_profit_atr=take,
    )


def crossover(a: Genome, b: Genome, rng: random.Random) -> Genome:
    # Uniform crossover on each field
    child = Genome(
        fast_ma=rng.choice([a.fast_ma, b.fast_ma]),
        slow_ma=rng.choice([a.slow_ma, b.slow_ma]),
        rsi_period=rng.choice([a.rsi_period, b.rsi_period]),
        rsi_buy=rng.choice([a.rsi_buy, b.rsi_buy]),
        rsi_sell=rng.choice([a.rsi_sell, b.rsi_sell]),
        atr_period=rng.choice([a.atr_period, b.atr_period]),
        stop_loss_atr=rng.choice([a.stop_loss_atr, b.stop_loss_atr]),
        take_profit_atr=rng.choice([a.take_profit_atr, b.take_profit_atr]),
    )
    # Fix constraints
    if child.slow_ma <= child.fast_ma:
        child = Genome(**{**child.to_dict(), "slow_ma": min([x for x in SLOW_MA if x > child.fast_ma])})
    if child.rsi_sell <= child.rsi_buy:
        child = Genome(**{**child.to_dict(), "rsi_sell": min([x for x in RSI_SELL if x > child.rsi_buy])})
    return child


def mutate(g: Genome, rng: random.Random, rate: float) -> Genome:
    d = g.to_dict()
    if rng.random() < rate:
        d["fast_ma"] = rng.choice(FAST_MA)
    if rng.random() < rate:
        d["slow_ma"] = rng.choice(SLOW_MA)
    if rng.random() < rate:
        d["rsi_period"] = rng.choice(RSI_PERIOD)
    if rng.random() < rate:
        d["rsi_buy"] = rng.choice(RSI_BUY)
    if rng.random() < rate:
        d["rsi_sell"] = rng.choice(RSI_SELL)
    if rng.random() < rate:
        d["atr_period"] = rng.choice(ATR_PERIOD)
    if rng.random() < rate:
        d["stop_loss_atr"] = rng.choice(STOP_ATR)
    if rng.random() < rate:
        d["take_profit_atr"] = rng.choice(TAKE_ATR)

    # constraints
    if d["slow_ma"] <= d["fast_ma"]:
        d["slow_ma"] = min([x for x in SLOW_MA if x > d["fast_ma"]])
    if d["rsi_sell"] <= d["rsi_buy"]:
        d["rsi_sell"] = min([x for x in RSI_SELL if x > d["rsi_buy"]])
    return Genome.from_dict(d)


def _genome_key(g: Genome) -> tuple:
    return (
        g.fast_ma,
        g.slow_ma,
        g.rsi_period,
        g.rsi_buy,
        g.rsi_sell,
        g.atr_period,
        g.stop_loss_atr,
        g.take_profit_atr,
    )


@dataclass(frozen=True)
class EvalSummary:
    fitness: float
    metrics: dict[str, Any]


def _aggregate_metrics(metrics: list[BacktestMetrics]) -> dict[str, Any]:
    if not metrics:
        return {
            "valid_symbols": 0,
            "mean_sharpe": 0.0,
            "mean_annual_return": 0.0,
            "mean_max_drawdown": 0.0,
            "mean_trades": 0.0,
        }
    sharpe = np.array([m.sharpe for m in metrics], dtype=float)
    ann = np.array([m.annual_return for m in metrics], dtype=float)
    dd = np.array([m.max_drawdown for m in metrics], dtype=float)
    trades = np.array([m.n_trades for m in metrics], dtype=float)
    return {
        "valid_symbols": int(len(metrics)),
        "mean_sharpe": float(np.mean(sharpe)),
        "mean_annual_return": float(np.mean(ann)),
        "mean_max_drawdown": float(np.mean(dd)),  # negative
        "mean_trades": float(np.mean(trades)),
    }


def fitness_from_agg(agg: dict[str, Any]) -> float:
    """Composite fitness (higher is better)."""
    if agg["valid_symbols"] <= 0:
        return -1e9

    mean_sharpe = float(agg["mean_sharpe"])
    mean_ann = float(agg["mean_annual_return"])
    mean_dd = float(agg["mean_max_drawdown"])  # negative
    mean_trades = float(agg["mean_trades"])

    # Core: Sharpe + return - drawdown
    f = 2.0 * mean_sharpe + 1.0 * mean_ann - 1.5 * abs(mean_dd)

    # Penalize too few trades (avoid "never trade" strategies)
    if mean_trades < 3:
        f -= (3 - mean_trades) * 0.3

    # Soft cap absurdly high annual return (usually overfit)
    if mean_ann > 1.5:
        f -= (mean_ann - 1.5) * 0.5

    return float(f)


def evaluate_genome(dataset: dict[str, pd.DataFrame], g: Genome, costs: CostModel) -> EvalSummary:
    metrics: list[BacktestMetrics] = []
    for _code, df in dataset.items():
        r = backtest(df, g, costs)
        if r.metrics.n_days <= 0:
            continue
        metrics.append(r.metrics)

    agg = _aggregate_metrics(metrics)
    fit = fitness_from_agg(agg)
    return EvalSummary(fitness=fit, metrics=agg | {"n_symbols": len(dataset)})


def _evaluate_worker(genome_dict: dict[str, Any]) -> tuple[tuple, EvalSummary]:
    """Worker function: evaluates a genome using fork-inherited dataset."""
    if _WORKER_DATASET is None or _WORKER_COSTS is None:
        raise RuntimeError("Worker context not initialized. Use fork start method on Linux.")
    g = Genome.from_dict(genome_dict)
    s = evaluate_genome(_WORKER_DATASET, g, _WORKER_COSTS)
    return _genome_key(g), s


def tournament_select(pop: list[Genome], fitnesses: list[float], rng: random.Random, k: int = 3) -> Genome:
    idx = [rng.randrange(len(pop)) for _ in range(k)]
    best_i = max(idx, key=lambda i: fitnesses[i])
    return pop[best_i]


@dataclass
class GAConfig:
    population_size: int = 64
    elite_size: int = 8
    crossover_rate: float = 0.65
    mutation_rate: float = 0.25
    workers: int = 3
    seed: int | None = 42
    checkpoint_path: str | None = None


class GeneticEngine:
    def __init__(self, dataset: dict[str, pd.DataFrame], costs: CostModel, cfg: GAConfig):
        self.dataset = dataset
        self.costs = costs
        self.cfg = cfg
        self.rng = random.Random(cfg.seed if cfg.seed is not None else random.randrange(1_000_000_000))
        self._cache: dict[tuple, EvalSummary] = {}

    def evolve_until(self, stop_at: dt.datetime) -> tuple[Genome, EvalSummary]:
        pop = [random_genome(self.rng) for _ in range(self.cfg.population_size)]
        best_g = pop[0]
        best_s = EvalSummary(fitness=-1e9, metrics={})

        # Prefer fork-based multiprocessing on Linux for speed.
        ctx = None
        try:
            ctx = mp.get_context("fork")
        except ValueError:
            ctx = None

        pool = None
        if ctx is not None and self.cfg.workers > 1:
            _set_worker_context(self.dataset, self.costs)
            pool = ctx.Pool(processes=self.cfg.workers)

        try:
            gen = 0
            while dt.datetime.now(tz=stop_at.tzinfo) < stop_at:
                gen += 1

                self._eval_population(pop, pool)

                fitnesses = [self._cache[_genome_key(g)].fitness for g in pop]
                summaries = [self._cache[_genome_key(g)] for g in pop]

                gen_best_i = int(np.argmax(np.asarray(fitnesses)))
                if fitnesses[gen_best_i] > best_s.fitness:
                    best_g = pop[gen_best_i]
                    best_s = summaries[gen_best_i]

                if gen == 1 or gen % 5 == 0:
                    console.log(
                        f"[bold cyan]GA[/] gen={gen} best_fitness={best_s.fitness:.4f} "
                        f"mean_sharpe={best_s.metrics.get('mean_sharpe', 0):.3f} "
                        f"mean_ann={best_s.metrics.get('mean_annual_return', 0):.3f} "
                        f"mean_dd={best_s.metrics.get('mean_max_drawdown', 0):.3f}"
                    )
                    self._maybe_checkpoint(gen, best_g, best_s)

                # next generation
                order = np.argsort(np.asarray(fitnesses))[::-1]
                elites = [pop[i] for i in order[: self.cfg.elite_size]]

                new_pop: list[Genome] = list(elites)
                while len(new_pop) < self.cfg.population_size:
                    p1 = tournament_select(pop, fitnesses, self.rng, k=3)
                    p2 = tournament_select(pop, fitnesses, self.rng, k=3)
                    child = p1
                    if self.rng.random() < self.cfg.crossover_rate:
                        child = crossover(p1, p2, self.rng)
                    child = mutate(child, self.rng, self.cfg.mutation_rate)
                    new_pop.append(child)
                pop = new_pop

            self._maybe_checkpoint(gen, best_g, best_s, force=True)
            return best_g, best_s
        finally:
            if pool is not None:
                pool.close()
                pool.join()

    def _eval_population(self, pop: list[Genome], pool) -> None:  # noqa: ANN001
        to_eval: list[Genome] = []
        for g in pop:
            key = _genome_key(g)
            if key not in self._cache:
                to_eval.append(g)

        if not to_eval:
            return

        if pool is None:
            for g in to_eval:
                key = _genome_key(g)
                self._cache[key] = evaluate_genome(self.dataset, g, self.costs)
            return

        # multiprocessing (fork): evaluate only missing genomes
        payloads = [g.to_dict() for g in to_eval]
        for key, summary in pool.imap_unordered(_evaluate_worker, payloads, chunksize=4):
            self._cache[key] = summary

    def _maybe_checkpoint(self, gen: int, g: Genome, s: EvalSummary, force: bool = False) -> None:
        if not self.cfg.checkpoint_path:
            return
        if not force and gen % 10 != 0:
            return
        p = Path(self.cfg.checkpoint_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "generated_at": dt.datetime.utcnow().isoformat(),
            "generation": gen,
            "genome": g.to_dict(),
            "fitness": s.fitness,
            "metrics": s.metrics,
        }
        p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
