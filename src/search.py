"""Differential Evolution (DE) and Particle Swarm Optimisation (PSO) for
hyperparameter search, as committed in the synopsis for the 1D CNN.

Owner: Parth Upadhyay

What is searched and what is not: DE and PSO choose *hyperparameters*
(learning rate, filters, kernel size, depth, dropout, dense width). The
network *weights* are still trained by RMSprop inside every candidate
evaluation. Both searches minimise 1 - validation macro-F1 (N/S/V/F); the
DS2 test split is never seen during the search.

Both optimisers share one interface and one budget so they can be compared
fairly: same search box, same seed, n_agents * (n_iter + 1) evaluations.

References
----------
Storn & Price (1997), "Differential Evolution - A Simple and Efficient
    Heuristic for Global Optimization over Continuous Spaces",
    J. Global Optimization 11:341-359.
Kennedy & Eberhart (1995), "Particle Swarm Optimization",
    Proc. ICNN'95, vol. 4, pp. 1942-1948.
"""
from __future__ import annotations

import numpy as np


class HyperSpace:
    """Maps a point in the unit box [0,1]^d to typed hyperparameters, so DE
    and PSO can both work on continuous vectors.

    >>> space = HyperSpace({'lr': ('log', 1e-4, 1e-2), 'k': ('choice', [3, 5, 7])})
    >>> space.decode(np.array([0.5, 0.9]))
    {'lr': 0.001, 'k': 7}
    """

    def __init__(self, spec: dict):
        self.spec = spec
        self.names = list(spec)
        self.dim = len(self.names)

    def decode(self, u) -> dict:
        u = np.clip(np.asarray(u, dtype=float), 0.0, 1.0)
        out = {}
        for t, name in zip(u, self.names):
            kind, *args = self.spec[name]
            if kind == 'float':
                lo, hi = args
                out[name] = round(float(lo + t * (hi - lo)), 4)
            elif kind == 'log':
                lo, hi = args
                out[name] = float(f'{np.exp(np.log(lo) + t * np.log(hi / lo)):.3g}')
            elif kind == 'int':
                lo, hi = args
                out[name] = int(min(hi, np.floor(lo + t * (hi - lo + 1))))
            elif kind == 'choice':
                opts = args[0]
                out[name] = opts[min(int(t * len(opts)), len(opts) - 1)]
            else:
                raise ValueError(f'unknown kind {kind!r} for {name!r}')
        return out


class _Evaluator:
    """Scores positions, caching decoded configs so a repeated config is not
    retrained, and logs every evaluation for the convergence plot."""

    def __init__(self, objective, space, verbose):
        self.objective, self.space, self.verbose = objective, space, verbose
        self.cache, self.log = {}, []

    def __call__(self, positions):
        scores = np.empty(len(positions))
        for i, p in enumerate(positions):
            cfg = self.space.decode(p)
            key = tuple(sorted(cfg.items()))
            if key not in self.cache:
                self.cache[key] = float(self.objective(cfg))
                self.log.append({'eval': len(self.log) + 1, 'config': cfg,
                                 'objective': self.cache[key]})
                if self.verbose:
                    print(f'  eval {len(self.log):3d}  1-F1={self.cache[key]:.4f}  {cfg}',
                          flush=True)
            scores[i] = self.cache[key]
        return scores


def differential_evolution(objective, space, n_agents=6, n_iter=3, seed=42,
                           F=0.5, CR=0.9, verbose=True):
    """DE/rand/1/bin. For each target vector, a mutant a + F(b - c) is built
    from three other random members, crossed over gene-wise with probability
    CR, and replaces the target only if it scores at least as well."""
    rng = np.random.default_rng(seed)
    d = space.dim
    ev = _Evaluator(objective, space, verbose)
    pop = rng.random((n_agents, d))
    fit = ev(pop)
    history = [float(fit.min())]
    for g in range(n_iter):
        trials = np.empty_like(pop)
        for i in range(n_agents):
            a, b, c = pop[rng.choice([j for j in range(n_agents) if j != i], 3,
                                     replace=False)]
            mutant = np.clip(a + F * (b - c), 0.0, 1.0)
            cross = rng.random(d) < CR
            cross[rng.integers(d)] = True             # at least one gene changes
            trials[i] = np.where(cross, mutant, pop[i])
        tfit = ev(trials)
        better = tfit <= fit
        pop[better], fit[better] = trials[better], tfit[better]
        history.append(float(fit.min()))
        if verbose:
            print(f'DE generation {g+1}/{n_iter}  best 1-F1={fit.min():.4f}', flush=True)
    best = int(np.argmin(fit))
    return space.decode(pop[best]), float(fit[best]), history, ev.log


def particle_swarm(objective, space, n_agents=6, n_iter=3, seed=42,
                   w=0.7, c1=1.5, c2=1.5, vmax=0.3, verbose=True):
    """Global-best PSO. Each particle's velocity mixes its own momentum (w),
    a pull towards its personal best (c1) and towards the swarm best (c2)."""
    rng = np.random.default_rng(seed)
    d = space.dim
    ev = _Evaluator(objective, space, verbose)
    x = rng.random((n_agents, d))
    v = rng.uniform(-vmax, vmax, (n_agents, d))
    fit = ev(x)
    pbest, pfit = x.copy(), fit.copy()
    g = int(np.argmin(pfit))
    history = [float(pfit[g])]
    for t in range(n_iter):
        r1, r2 = rng.random((n_agents, d)), rng.random((n_agents, d))
        v = w * v + c1 * r1 * (pbest - x) + c2 * r2 * (pbest[g] - x)
        v = np.clip(v, -vmax, vmax)
        x = np.clip(x + v, 0.0, 1.0)
        fit = ev(x)
        improved = fit < pfit
        pbest[improved], pfit[improved] = x[improved], fit[improved]
        g = int(np.argmin(pfit))
        history.append(float(pfit[g]))
        if verbose:
            print(f'PSO iteration {t+1}/{n_iter}  best 1-F1={pfit[g]:.4f}', flush=True)
    return space.decode(pbest[g]), float(pfit[g]), history, ev.log


OPTIMIZERS = {'de': differential_evolution, 'pso': particle_swarm}
