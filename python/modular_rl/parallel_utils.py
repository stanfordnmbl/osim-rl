import multiprocessing
import sys

class G: #pylint: disable=W0232
    pass

G.pool = None
G.worker_queue = None
G.queue = None

def init_pool():
    """
    Initialize a pool of workers. The number of workers is decided by the
    number of physical cores on the current machine.

    This should be called at the beginning of the script before Theano is
    initialized. Otherwise, if the main process is using Theano with GPU,
    all the other worker processes will be messed up.
    """
    nproc = multiprocessing.cpu_count()
    if sys.platform == "darwin":
        nproc /= 2 
    # hyperthreading makes num cpu look twice as high
    # but there's no speedup
    # store data in global variables so it's accessible from forked processes
    # (which start when multiprocessing.Pool is created)
    G.worker_queue = multiprocessing.Queue()
    G.queue = multiprocessing.Queue()
    G.pool = multiprocessing.Pool(nproc, initializer=worker_init_theano)
    G.n_parallel = nproc

def worker_init_theano():
    pass
    # import os
    # os.environ['THEANO_FLAGS'] = 'device=cpu'

def worker_run_task_blocked(all_args):
    f, args, kwargs = all_args
    # signals to the master that this task is up and running
    G.worker_queue.put(None)
    # wait for the master to signal continuation
    G.queue.get()
    return f(*args, **kwargs)

def worker_run_task(all_args):
    f, args, kwargs = all_args
    return f(*args, **kwargs)

def apply_each(f, *args, **kwargs):
    """
    Apply the method f to each worker process. The method will be actually
    invoked like f(G, *args)

    This is actually a little bit tricky to implement because the Pool
    abstraction tries to hide away which worker is executing the task.
    """
    assert G.pool, "G.pool not initialized. Make sure to call init_pool() first"
    results = G.pool.map_async(worker_run_task_blocked,
                               [(f, args, kwargs)] * G.n_parallel)
    for _ in range(G.n_parallel):
        G.worker_queue.get()
    for _ in range(G.n_parallel):
        G.queue.put(None)
    results.get()

def apply_async(f, *args, **kwargs):
    assert G.pool, "G.pool not initialized. Make sure to call init_pool() first"
    return G.pool.apply_async(worker_run_task, ((f, args, kwargs),))

def close_pool():
    if G.pool:
        G.pool.close()
