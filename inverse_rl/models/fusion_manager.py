import os
import joblib
import re

import numpy as np
import tensorflow as tf
from rllab.misc.logger import get_snapshot_dir


class FusionDistrManager(object):
    def add_paths(self, paths):
        raise NotImplementedError()

    def sample_paths(self, n):
        raise NotImplementedError()


class PathsReader(object):
    ITR_REG = re.compile(r"itr_(?P<itr_count>[0-9]+)\.pkl")

    def __init__(self, path_dir):
        self.path_dir = path_dir

    def get_path_files(self):
        itr_files = []
        for i, filename in enumerate(os.listdir(self.path_dir)):
            m = PathsReader.ITR_REG.match(filename)
            if m:
                itr_count = m.group('itr_count')
                itr_files.append((itr_count, filename))

        itr_files = sorted(itr_files, key=lambda x: int(x[0]), reverse=True)
        for itr_file_and_count in itr_files:
            fname = os.path.join(self.path_dir, itr_file_and_count[1])
            yield fname

    def __len__(self):
        return len(list(self.get_path_files()))


class DiskFusionDistr(FusionDistrManager):
    def __init__(self, path_dir=None):
        if path_dir is None:
            path_dir = get_snapshot_dir()
        self.path_dir = path_dir
        self.paths_reader = PathsReader(path_dir)

    def add_paths(self, paths):
        raise NotImplementedError()

    def sample_paths(self, n):
        # load from disk!
        fnames = list(self.paths_reader.get_path_files())
        N = len(fnames)
        sample_files = np.random.randint(0, N, size=(n))
        #sample_hist = np.histogram(sample_files, range=(0, N))
        #print(sample_hist)
        unique, counts = np.unique(sample_files, return_counts=True)
        unique_dict = dict(zip(unique, counts))

        all_paths = []
        for fidx in unique_dict:
            fname = fnames[fidx]
            n_samp = unique_dict[fidx]
            print(fname, n_samp)

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            with tf.Graph().as_default():
                with tf.Session(config=config).as_default():
                    snapshot_dict = joblib.load(fname)
            paths = snapshot_dict['paths']
            pidxs = np.random.randint(0, len(paths), size=(n_samp))
            all_paths.extend([paths[pidx] for pidx in pidxs])
        return all_paths


class RamFusionDistr(FusionDistrManager):
    def __init__(self, buf_size, subsample_ratio=0.5):
        self.buf_size = buf_size
        self.buffer = []
        self.subsample_ratio = subsample_ratio

    def add_paths(self, paths, subsample=True):
        if subsample:
            paths = paths[:int(len(paths)*self.subsample_ratio)]
        self.buffer.extend(paths)
        overflow = len(self.buffer)-self.buf_size
        while overflow > 0:
            #self.buffer = self.buffer[overflow:]
            N = len(self.buffer)
            probs = np.arange(N)+1
            probs = probs/float(np.sum(probs))
            pidx = np.random.choice(np.arange(N), p=probs)
            self.buffer.pop(pidx)
            overflow -= 1

    def sample_paths(self, n):
        if len(self.buffer) == 0:
            return []
        else:
            pidxs = np.random.randint(0, len(self.buffer), size=(n))
            return [self.buffer[pidx] for pidx in pidxs]


if __name__ == "__main__":
    #fm = DiskFusionDistr(path_dir='data_nobs/gridworld_random/gru1')
    #paths = fm.sample_paths(10)
    fm = RamFusionDistr(10)
    fm.add_paths([1,2,3,4,5,6,7,8,9,10,11,12,13])
    print(fm.buffer)
    print(fm.sample_paths(5))
