import tensorflow as tf

from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.gym_env import GymEnv

from inverse_rl.envs.env_utils import CustomGymEnv
from inverse_rl.algos.irl_trpo import IRLTRPO
from inverse_rl.models.airl_state import *
from inverse_rl.utils.log_utils import rllab_logdir, load_latest_experts, load_latest_experts_multiple_runs
from inverse_rl.utils.hyper_sweep import run_sweep_parallel, run_sweep_serial

def main(exp_name=None, fusion=False):
    env = TfEnv(CustomGymEnv('CustomAnt-v0', record_video=False, record_log=False))

    # load ~2 iterations worth of data from each forward RL experiment as demos
    experts = load_latest_experts_multiple_runs('data/ant_data_collect', n=2)

    irl_model = AIRL(env=env, expert_trajs=experts, state_only=True, fusion=fusion, max_itrs=10)

    policy = GaussianMLPPolicy(name='policy', env_spec=env.spec, hidden_sizes=(32, 32))
    algo = IRLTRPO(
        env=env,
        policy=policy,
        irl_model=irl_model,
        n_itr=1000,
        batch_size=10000,
        max_path_length=500,
        discount=0.99,
        store_paths=True,
        irl_model_wt=1.0,
        entropy_weight=0.1,
        zero_environment_reward=True,
        baseline=LinearFeatureBaseline(env_spec=env.spec),
    )
    with rllab_logdir(algo=algo, dirname='data/ant_state_irl/%s' % exp_name):
        with tf.Session():
            algo.train()

if __name__ == "__main__":
    params_dict = {
        'fusion': [True]
    }
    run_sweep_parallel(main, params_dict, repeat=3)

