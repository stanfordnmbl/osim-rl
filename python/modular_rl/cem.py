from .core import *
from . import parallel_utils
from tabulate import tabulate
import os

# ================================================================
# Cross-entropy method 
# ================================================================

def cem(f,th_mean,batch_size,n_iter,elite_frac, initial_std=1.0, extra_std=0.0, std_decay_time=1.0, pool=None):
    r"""
    Noisy cross-entropy method
    http://dx.doi.org/10.1162/neco.2006.18.12.2936
    http://ie.technion.ac.il/CE/files/papers/Learning%20Tetris%20Using%20the%20Noisy%20Cross-Entropy%20Method.pdf
    Incorporating schedule described on page 4 (also see equation below.)

    Inputs
    ------

    f : function of one argument--the parameter vector
    th_mean : initial distribution is theta ~ Normal(th_mean, initial_std)
    batch_size : how many samples of theta per iteration
    n_iter : how many iterations
    elite_frac : how many samples to select at the end of the iteration, and use for fitting new distribution
    initial_std : standard deviation of initial distribution
    extra_std : "noise" component added to increase standard deviation.
    std_decay_time : how many timesteps it takes for noise to decay

    \sigma_{t+1}^2 =  \sigma_{t,elite}^2 + extra_std * Z_t^2
    where Zt = max(1 - t / std_decay_time, 10 , 0) * extra_std.
    """
    n_elite = int(np.round(batch_size*elite_frac))

    th_std = np.ones(th_mean.size)*initial_std

    for iteration in xrange(n_iter):

        extra_var_multiplier = max((1.0-iteration/float(std_decay_time)),0) # Multiply "extra variance" by this factor
        print "extra var", extra_var_multiplier
        sample_std = np.sqrt(th_std + np.square(extra_std) * extra_var_multiplier)

        ths = np.array([th_mean + dth for dth in  sample_std[None,:]*np.random.randn(batch_size, th_mean.size)])
        if pool is None: ys = np.array(map(f, ths))
        else: ys = np.array(pool.map(f, ths))
        assert ys.ndim==1
        elite_inds = ys.argsort()[-n_elite:]
        elite_ths = ths[elite_inds]
        th_mean = elite_ths.mean(axis=0)
        th_std = elite_ths.var(axis=0)
        yield {"ys":ys,"th":th_mean,"ymean":ys.mean(), "std" : sample_std}


CEM_OPTIONS = [
    ("batch_size", int, 200, "Number of episodes per batch"),
    ("n_iter", int, 200, "Number of iterations"),
    ("elite_frac", float, 0.2, "fraction of parameter settings used to fit pop"),
    ("initial_std", float, 1.0, "initial standard deviation for parameters"),
    ("extra_std", float, 0.0, "extra stdev added"),
    ("std_decay_time", float, -1.0, "number of timesteps that extra stdev decays over. negative => n_iter/2"),
    ("timestep_limit", int, 0, "maximum length of trajectories"),
    ("parallel", int, 0, "collect trajectories in parallel"),
]

def _cem_objective(th):
    G = parallel_utils.G
    G.agent.set_from_flat(th)
    path = rollout(G.env, G.agent, G.timestep_limit)
    return path["reward"].sum()

def _seed_with_pid():
    np.random.seed(os.getpid())

def run_cem_algorithm(env, agent, usercfg=None, callback=None):
    cfg = update_default_config(CEM_OPTIONS, usercfg)
    if cfg["std_decay_time"] < 0: cfg["std_decay_time"] = cfg["n_iter"] / 2 
    cfg.update(usercfg)
    print "cem config", cfg

    G = parallel_utils.G
    G.env = env
    G.agent = agent
    G.timestep_limit = cfg["timestep_limit"]
    if cfg["parallel"]:
        parallel_utils.init_pool()
        pool = G.pool
    else:
        pool = None

    th_mean = agent.get_flat()

    for info in cem(_cem_objective, th_mean, cfg["batch_size"], cfg["n_iter"], cfg["elite_frac"], 
        cfg["initial_std"], cfg["extra_std"], cfg["std_decay_time"], pool=pool):
        callback(info)
        ps = np.linspace(0,100,5)
        print tabulate([ps, np.percentile(info["ys"].ravel(),ps), np.percentile(info["std"].ravel(),ps)])

        agent.set_from_flat(info["th"])

