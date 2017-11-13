import numpy as np
import time
import scipy as sp
import scipy.stats
import contextlib


def flat_to_one_hot(val, ndim):
    """

    >>> flat_to_one_hot(2, ndim=4)
    array([ 0.,  0.,  1.,  0.])
    >>> flat_to_one_hot(4, ndim=5)
    array([ 0.,  0.,  0.,  0.,  1.])
    >>> flat_to_one_hot(np.array([2, 4, 3]), ndim=5)
    array([[ 0.,  0.,  1.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  1.],
           [ 0.,  0.,  0.,  1.,  0.]])
    """
    shape =np.array(val).shape
    v = np.zeros(shape + (ndim,))
    if len(shape) == 1:
        v[np.arange(shape[0]), val] = 1.0
    else:
        v[val] = 1.0
    return v

def one_hot_to_flat(val):
    """
    >>> one_hot_to_flat(np.array([0,0,0,0,1]))
    4
    >>> one_hot_to_flat(np.array([0,0,1,0]))
    2
    >>> one_hot_to_flat(np.array([[0,0,1,0], [1,0,0,0], [0,1,0,0]]))
    array([2, 0, 1])
    """
    idxs = np.array(np.where(val == 1.0))[-1]
    if len(val.shape) == 1:
        return int(idxs)
    return idxs


def flatten_list(lol):
    return [ a for b in lol for a in b ]

class TrainingIterator(object):
    def __init__(self, itrs, heartbeat=float('inf')):
        self.itrs = itrs
        self.heartbeat_time = heartbeat
        self.__vals = {}

    def random_idx(self, N, size):
        return np.random.randint(0, N, size=size)

    @property
    def itr(self):
        return self.__itr

    @property
    def heartbeat(self):
        return self.__heartbeat

    @property
    def elapsed(self):
        assert self.heartbeat, 'elapsed is only valid when heartbeat=True'
        return self.__elapsed

    def itr_message(self):
        return '==> Itr %d/%d (elapsed:%.2f)' % (self.itr+1, self.itrs, self.elapsed)

    def record(self, key, value):
        if key in self.__vals:
            self.__vals[key].append(value)
        else:
            self.__vals[key] = [value]

    def pop(self, key):
        vals = self.__vals.get(key, [])
        del self.__vals[key]
        return vals

    def pop_mean(self, key):
        return np.mean(self.pop(key))

    def __iter__(self):
        prev_time = time.time()
        self.__heartbeat = False
        for i in range(self.itrs):
            self.__itr = i
            cur_time = time.time()
            if (cur_time-prev_time) > self.heartbeat_time or i==(self.itrs-1):
                self.__heartbeat = True
                self.__elapsed = cur_time-prev_time
                prev_time = cur_time
            yield self
            self.__heartbeat = False


def gd_optimizer(lr, lr_sched=None):
    if lr_sched is None:
        lr_sched = {}

    itr = 0
    def update(x, grad):
        nonlocal itr, lr
        if itr in lr_sched:
            lr *= lr_sched[itr]
        new_x = x - lr * grad
        itr += 1
        return new_x
    return update


def gd_momentum_optimizer(lr, momentum=0.9, lr_sched=None):
    if lr_sched is None:
        lr_sched = {}

    itr = 0
    prev_grad = None
    def update(x, grad):
        nonlocal itr, lr, prev_grad
        if itr in lr_sched:
            lr *= lr_sched[itr]

        if prev_grad is None:
            grad = grad
        else:
            grad = grad + momentum * prev_grad
        new_x = x - lr * grad
        prev_grad = grad
        itr += 1
        return new_x
    return update


def adam_optimizer(lr, beta1=0.9, beta2=0.999, eps=1e-8):
    itr = 0
    pm = None
    pv = None
    def update(x, grad):
        nonlocal itr, lr, pm, pv
        if pm is None:
            pm = np.zeros_like(grad)
            pv = np.zeros_like(grad)

        pm = beta1 * pm + (1-beta1)*grad
        pv = beta2 * pv + (1-beta2)*(grad*grad)
        mhat = pm/(1-beta1**(itr+1))
        vhat = pv/(1-beta2**(itr+1))
        update_vec = mhat / (np.sqrt(vhat)+eps)
        new_x = x - lr * update_vec
        itr += 1
        return new_x
    return update


@contextlib.contextmanager
def np_seed(seed):
    """ A context for np random seeds """
    if seed is None:
        yield
    else:
        old_state = np.random.get_state()
        np.random.seed(seed)
        yield
        np.random.set_state(old_state)

