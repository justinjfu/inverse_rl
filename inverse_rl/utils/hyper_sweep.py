"""
Usage

args = {
'param1': [1e-3, 1e-2, 1e-2],
'param2': [1,5,10,20],
}

run_sweep_parallel(func, args)

or

run_sweep_serial(func, args)

"""
import itertools
import multiprocessing
import random
from datetime import datetime

class Sweeper(object):
    def __init__(self, hyper_config, repeat):
        self.hyper_config = hyper_config
        self.repeat = repeat

    def __iter__(self):
        count = 0
        for _ in range(self.repeat):
            for config in itertools.product(*[val for val in self.hyper_config.values()]):
                kwargs = {key:config[i] for i, key in enumerate(self.hyper_config.keys())}
                timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
                kwargs['exp_name'] = "%s_%d" % (timestamp, count)
                count += 1
                yield kwargs



def run_sweep_serial(run_method, params, repeat=1):
    sweeper = Sweeper(params, repeat)
    for config in sweeper:
        import tensorflow as tf; tf.reset_default_graph()
        run_method(**config)

def kwargs_wrapper(args_method):
    import tensorflow as tf; tf.reset_default_graph()
    args, method = args_method
    return method(**args)


def run_sweep_parallel(run_method, params, repeat=1, num_cpu=multiprocessing.cpu_count()):
    sweeper = Sweeper(params, repeat)
    pool = multiprocessing.Pool(num_cpu)
    exp_args = []
    for config in sweeper:
        exp_args.append((config, run_method))
    random.shuffle(exp_args)
    pool.map(kwargs_wrapper, exp_args)


def example_run_method(exp_name, param1, param2='a', param3=3, param4=4):
    import time
    time.sleep(1.0)
    print(exp_name, param1, param2, param3, param4)


if __name__ == "__main__":
    sweep_op = {
        'param1': [1e-3, 1e-2, 1e-1],
        'param2': [1,5,10,20],
        'param3': [True, False]
    }
    run_sweep_parallel(example_run_method, sweep_op, repeat=2)