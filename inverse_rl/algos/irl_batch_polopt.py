import time

from rllab.algos.base import RLAlgorithm
import rllab.misc.logger as logger
import rllab.plotter as plotter
from sandbox.rocky.tf.policies.base import Policy
import tensorflow as tf
from sandbox.rocky.tf.samplers.batch_sampler import BatchSampler
from sandbox.rocky.tf.samplers.vectorized_sampler import VectorizedSampler
import numpy as np
from collections import deque

from inverse_rl.utils.hyperparametrized import Hyperparametrized


class IRLBatchPolopt(RLAlgorithm, metaclass=Hyperparametrized):
    """
    Base class for batch sampling-based policy optimization methods.
    This includes various policy gradient methods like vpg, npg, ppo, trpo, etc.
    """

    def __init__(
            self,
            env,
            policy,
            baseline,
            scope=None,
            n_itr=500,
            start_itr=0,
            batch_size=5000,
            max_path_length=500,
            discount=0.99,
            gae_lambda=1,
            plot=False,
            pause_for_plot=False,
            center_adv=True,
            positive_adv=False,
            store_paths=True,
            whole_paths=True,
            fixed_horizon=False,
            sampler_cls=None,
            sampler_args=None,
            force_batch_sampler=False,
            init_pol_params = None,
            irl_model=None,
            irl_model_wt=1.0,
            discrim_train_itrs=10,
            zero_environment_reward=False,
            init_irl_params=None,
            train_irl=True,
            key='',
            **kwargs
    ):
        """
        :param env: Environment
        :param policy: Policy
        :type policy: Policy
        :param baseline: Baseline
        :param scope: Scope for identifying the algorithm. Must be specified if running multiple algorithms
        simultaneously, each using different environments and policies
        :param n_itr: Number of iterations.
        :param start_itr: Starting iteration.
        :param batch_size: Number of samples per iteration.
        :param max_path_length: Maximum length of a single rollout.
        :param discount: Discount.
        :param gae_lambda: Lambda used for generalized advantage estimation.
        :param plot: Plot evaluation run after each iteration.
        :param pause_for_plot: Whether to pause before contiuing when plotting.
        :param center_adv: Whether to rescale the advantages so that they have mean 0 and standard deviation 1.
        :param positive_adv: Whether to shift the advantages so that they are always positive. When used in
        conjunction with center_adv the advantages will be standardized before shifting.
        :param store_paths: Whether to save all paths data to the snapshot.
        :return:
        """
        self.env = env
        self.policy = policy
        self.baseline = baseline
        self.scope = scope
        self.n_itr = n_itr
        self.start_itr = start_itr
        self.batch_size = batch_size
        self.max_path_length = max_path_length
        self.discount = discount
        self.gae_lambda = gae_lambda
        self.plot = plot
        self.pause_for_plot = pause_for_plot
        self.center_adv = center_adv
        self.positive_adv = positive_adv
        self.store_paths = store_paths
        self.whole_paths = whole_paths
        self.fixed_horizon = fixed_horizon
        self.init_pol_params = init_pol_params
        self.init_irl_params = init_irl_params
        self.irl_model = irl_model
        self.irl_model_wt = irl_model_wt
        self.no_reward = zero_environment_reward
        self.discrim_train_itrs = discrim_train_itrs
        self.train_irl = train_irl
        self.__irl_params = None

        if self.irl_model_wt > 0:
            assert self.irl_model is not None, "Need to specify a IRL model"

        if sampler_cls is None:
            if self.policy.vectorized and not force_batch_sampler:
                print('using vec sampler')
                sampler_cls = VectorizedSampler
            else:
                print('using batch sampler')
                sampler_cls = BatchSampler
        if sampler_args is None:
            sampler_args = dict()
        self.sampler = sampler_cls(self, **sampler_args)
        self.init_opt()

    def start_worker(self):
        self.sampler.start_worker()
        if self.plot:
            plotter.init_plot(self.env, self.policy)

    def shutdown_worker(self):
        self.sampler.shutdown_worker()

    def obtain_samples(self, itr):
        return self.sampler.obtain_samples(itr)

    def process_samples(self, itr, paths):
        #processed = self.sampler.process_samples(itr, paths)
        return self.sampler.process_samples(itr, paths)

    def log_avg_returns(self, paths):
        undiscounted_returns = [sum(path["rewards"]) for path in paths]
        avg_return = np.mean(undiscounted_returns)
        return avg_return

    def get_irl_params(self):
        return self.__irl_params

    def compute_irl(self, paths, itr=0):
        if self.no_reward:
            tot_rew = 0
            for path in paths:
                tot_rew += np.sum(path['rewards'])
                path['rewards'] *= 0
            logger.record_tabular('OriginalTaskAverageReturn', tot_rew/float(len(paths)))

        if self.irl_model_wt <=0:
            return paths

        if self.train_irl:
            max_itrs = self.discrim_train_itrs
            lr=1e-3
            mean_loss = self.irl_model.fit(paths, policy=self.policy, itr=itr, max_itrs=max_itrs, lr=lr,
                                           logger=logger)

            logger.record_tabular('IRLLoss', mean_loss)
            self.__irl_params = self.irl_model.get_params()

        probs = self.irl_model.eval(paths, gamma=self.discount, itr=itr)

        logger.record_tabular('IRLRewardMean', np.mean(probs))
        logger.record_tabular('IRLRewardMax', np.max(probs))
        logger.record_tabular('IRLRewardMin', np.min(probs))


        if self.irl_model.score_trajectories:
            # TODO: should I add to reward here or after advantage computation?
            for i, path in enumerate(paths):
                path['rewards'][-1] += self.irl_model_wt * probs[i]
        else:
            for i, path in enumerate(paths):
                path['rewards'] += self.irl_model_wt * probs[i]
        return paths

    def train(self):
        sess = tf.get_default_session()
        sess.run(tf.global_variables_initializer())
        if self.init_pol_params is not None:
            self.policy.set_param_values(self.init_pol_params)
        if self.init_irl_params is not None:
            self.irl_model.set_params(self.init_irl_params)
        self.start_worker()
        start_time = time.time()

        returns = []
        for itr in range(self.start_itr, self.n_itr):
            itr_start_time = time.time()
            with logger.prefix('itr #%d | ' % itr):
                logger.log("Obtaining samples...")
                paths = self.obtain_samples(itr)

                logger.log("Processing samples...")
                paths = self.compute_irl(paths, itr=itr)
                returns.append(self.log_avg_returns(paths))
                samples_data = self.process_samples(itr, paths)

                logger.log("Logging diagnostics...")
                self.log_diagnostics(paths)
                logger.log("Optimizing policy...")
                self.optimize_policy(itr, samples_data)
                logger.log("Saving snapshot...")
                params = self.get_itr_snapshot(itr, samples_data)  # , **kwargs)
                if self.store_paths:
                    params["paths"] = samples_data["paths"]
                logger.save_itr_params(itr, params)
                logger.log("Saved")
                logger.record_tabular('Time', time.time() - start_time)
                logger.record_tabular('ItrTime', time.time() - itr_start_time)
                logger.dump_tabular(with_prefix=False)
                if self.plot:
                    self.update_plot()
                    if self.pause_for_plot:
                        input("Plotting evaluation run: Press Enter to "
                              "continue...")
        self.shutdown_worker()
        return 

    def log_diagnostics(self, paths):
        self.env.log_diagnostics(paths)
        self.policy.log_diagnostics(paths)
        self.baseline.log_diagnostics(paths)

    def init_opt(self):
        """
        Initialize the optimization procedure. If using tensorflow, this may
        include declaring all the variables and compiling functions
        """
        raise NotImplementedError

    def get_itr_snapshot(self, itr, samples_data):
        """
        Returns all the data that should be saved in the snapshot for this
        iteration.
        """
        raise NotImplementedError

    def optimize_policy(self, itr, samples_data):
        raise NotImplementedError

    def update_plot(self):
        if self.plot:
            plotter.update_plot(self.policy, self.max_path_length)
