"""
Use q-iteration to solve for an optimal policy

Usage: q_iteration(env, gamma=discount factor, ent_wt= entropy bonus)
"""
import numpy as np
from scipy.misc import logsumexp as sp_lse

def softmax(q, alpha=1.0):
    q = (1.0/alpha)*q
    q = q-np.max(q)
    probs = np.exp(q)
    probs = probs/np.sum(probs)
    return probs

def logsumexp(q, alpha=1.0, axis=1):
    return alpha*sp_lse((1.0/alpha)*q, axis=axis)

def get_policy(q_fn, ent_wt=1.0):
    """
    Return a policy by normalizing a Q-function
    """
    v_rew = logsumexp(q_fn, alpha=ent_wt)
    adv_rew = q_fn - np.expand_dims(v_rew, axis=1)
    pol_probs = np.exp((1.0/ent_wt)*adv_rew)
    assert np.all(np.isclose(np.sum(pol_probs, axis=1), 1.0)), str(pol_probs)
    return pol_probs

def q_iteration(env, reward_matrix=None, K=50, gamma=0.99, ent_wt=0.1, warmstart_q=None, policy=None):
    """
    Perform tabular soft Q-iteration

    If policy is given, this computes Q_pi rather than Q_star
    """
    dim_obs = env.observation_space.flat_dim
    dim_act = env.action_space.flat_dim
    if reward_matrix is None:
        reward_matrix = env.rew_matrix
    if warmstart_q is None:
        q_fn = np.zeros((dim_obs, dim_act))
    else:
        q_fn = warmstart_q

    t_matrix = env.transition_matrix
    for k in range(K):
        if policy is None:
            v_fn = logsumexp(q_fn, alpha=ent_wt)
        else:
            v_fn = np.sum((q_fn - np.log(policy))*policy, axis=1)
        new_q = reward_matrix + gamma*t_matrix.dot(v_fn)
        q_fn = new_q
    return q_fn

