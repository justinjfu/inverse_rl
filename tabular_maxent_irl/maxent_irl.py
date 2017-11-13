"""
This implements Maximum Entropy IRL using dynamic programming. This

Simply call tabular_maxent_irl(env, expert_visitations)
    The expert visitations can be generated via the compute_visitations function on an expert q_function (exact),
    or using compute_visitations_demo on demos (approximate)

"""
import numpy as np
from utils import one_hot_to_flat, flat_to_one_hot
from q_iteration import q_iteration, logsumexp, get_policy
from utils import TrainingIterator
from utils import gd_momentum_optimizer, adam_optimizer


def compute_visitation(env, q_fn, ent_wt=1.0, T=50, discount=1.0):
    pol_probs = get_policy(q_fn, ent_wt=ent_wt)

    dim_obs = env.observation_space.flat_dim
    dim_act = env.action_space.flat_dim
    state_visitation = np.expand_dims(env.initial_state_distribution, axis=1)
    t_matrix = env.transition_matrix  # S x A x S
    sa_visit_t = np.zeros((dim_obs, dim_act, T))

    for i in range(T):
        sa_visit = state_visitation * pol_probs
        sa_visit_t[:,:,i] = sa_visit #(discount**i) * sa_visit
        # sum-out (SA)S
        new_state_visitation = np.einsum('ij,ijk->k', sa_visit, t_matrix)
        state_visitation = np.expand_dims(new_state_visitation, axis=1)
    return np.sum(sa_visit_t, axis=2) / float(T)


def compute_vistation_demos(env, demos):
    dim_obs = env.observation_space.flat_dim
    dim_act = env.action_space.flat_dim
    counts = np.zeros((dim_obs, dim_act))

    for demo in demos:
        obs = demo['observations']
        act = demo['actions']
        state_ids = one_hot_to_flat(obs)
        T = len(state_ids)
        for t in range(T):
            counts[state_ids[t], act[t]] += 1
    return counts / float(np.sum(counts))


def sample_states(env, q_fn, visitation_probs, n_sample, ent_wt):
    dS, dA = visitation_probs.shape
    samples = np.random.choice(np.arange(dS*dA), size=n_sample, p=visitation_probs.reshape(dS*dA))
    policy = get_policy(q_fn, ent_wt=ent_wt)
    observations = samples // dA
    actions = samples % dA
    a_logprobs = np.log(policy[observations, actions])

    observations_next = []
    for i in range(n_sample):
        t_distr = env.tabular_trans_distr(observations[i], actions[i])
        next_state = flat_to_one_hot(np.random.choice(np.arange(len(t_distr)), p=t_distr), ndim=dS)
        observations_next.append(next_state)
    observations_next = np.array(observations_next)

    return {'observations': flat_to_one_hot(observations, ndim=dS),
            'actions': flat_to_one_hot(actions, ndim=dA),
            'a_logprobs': a_logprobs,
            'observations_next': observations_next}


def tabular_maxent_irl(env, demo_visitations, num_itrs=50, ent_wt=1.0, lr=1e-3, state_only=False,
                       discount=0.99, T=5):
    dim_obs = env.observation_space.flat_dim
    dim_act = env.action_space.flat_dim

    # Initialize policy and reward function
    reward_fn = np.zeros((dim_obs, dim_act))
    q_rew = np.zeros((dim_obs, dim_act))

    update = adam_optimizer(lr)

    for it in TrainingIterator(num_itrs, heartbeat=1.0):
        q_itrs = 20 if it.itr>5 else 100
        ### compute policy in closed form
        q_rew = q_iteration(env, reward_matrix=reward_fn, ent_wt=ent_wt, warmstart_q=q_rew, K=q_itrs, gamma=discount)

        ### update reward
        # need to count how often the policy will visit a particular (s, a) pair
        pol_visitations = compute_visitation(env, q_rew, ent_wt=ent_wt, T=T, discount=discount)

        grad = -(demo_visitations - pol_visitations)
        it.record('VisitationInfNormError', np.max(np.abs(grad)))
        if state_only:
            grad = np.sum(grad, axis=1, keepdims=True)
        reward_fn = update(reward_fn, grad)

        if it.heartbeat:
            print(it.itr_message())
            print('\tVisitationError:',it.pop_mean('VisitationInfNormError'))
    return reward_fn, q_rew


if __name__ == "__main__":
    # test IRL
    from q_iteration import q_iteration
    from simple_env import random_env
    np.set_printoptions(suppress=True)

    # Environment parameters
    env = random_env(16, 4, seed=1, terminate=False, t_sparsity=0.8)
    dS = env.spec.observation_space.flat_dim
    dU = env.spec.action_space.flat_dim
    dO = 8
    ent_wt = 1.0
    discount = 0.9
    obs_matrix = np.random.randn(dS, dO)

    # Compute optimal policy for double checking
    true_q = q_iteration(env, K=150, ent_wt=ent_wt, gamma=discount)
    true_sa_visits = compute_visitation(env, true_q, ent_wt=ent_wt, T=5, discount=discount)
    expert_pol = get_policy(true_q, ent_wt=ent_wt)

    # Run MaxEnt IRL State-only
    learned_rew, learned_q = tabular_maxent_irl(env, true_sa_visits, lr=0.01, num_itrs=1000,
                                                ent_wt=ent_wt, state_only=True,
                                                discount=discount)
    learned_pol = get_policy(learned_q, ent_wt=ent_wt)

    
    # Normalize reward (if state_only=True, reward is accurate up to a constant)
    adjusted_rew = learned_rew - np.mean(learned_rew) + np.mean(env.rew_matrix)

    diff_rew = np.abs(env.rew_matrix - adjusted_rew)
    diff_pol = np.abs(expert_pol - learned_pol)
    print('----- Results State Only -----')
    print('InfNormRewError', np.max(diff_rew))
    print('InfNormPolicyError', np.max(diff_pol))
    print('AvdDiffRew', np.mean(diff_rew))
    print('AvgDiffPol', np.mean(diff_pol))
    print('True Reward', env.rew_matrix)
    print('Learned Reward', adjusted_rew)


    # Run MaxEnt IRL State-Action
    learned_rew, learned_q = tabular_maxent_irl(env, true_sa_visits, lr=0.01, num_itrs=1000,
                                                ent_wt=ent_wt, state_only=False,
                                                discount=discount)
    learned_pol = get_policy(learned_q, ent_wt=ent_wt)
    
    # Normalize reward (if state_only=True, reward is accurate up to a constant)
    adjusted_rew = learned_rew - np.mean(learned_rew) + np.mean(env.rew_matrix)

    diff_rew = np.abs(env.rew_matrix - adjusted_rew)
    diff_pol = np.abs(expert_pol - learned_pol)
    print('----- Results State-Action -----')
    print('InfNormRewError', np.max(diff_rew))
    print('InfNormPolicyError', np.max(diff_pol))
    print('AvdDiffRew', np.mean(diff_rew))
    print('AvgDiffPol', np.mean(diff_pol))
    print('True Reward', env.rew_matrix)
    print('Learned Reward', adjusted_rew)
