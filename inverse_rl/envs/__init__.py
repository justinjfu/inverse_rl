import logging

from gym.envs import register

LOGGER = logging.getLogger(__name__)

_REGISTERED = False
def register_custom_envs():
    global _REGISTERED
    if _REGISTERED:
        return
    _REGISTERED = True

    LOGGER.info("Registering custom gym environments")
    register(id='ObjPusher-v0', entry_point='inverse_rl.envs.pusher_env:PusherEnv', kwargs={'sparse_reward': False})
    register(id='TwoDMaze-v0', entry_point='inverse_rl.envs.twod_maze:TwoDMaze')
    register(id='PointMazeRight-v0', entry_point='inverse_rl.envs.point_maze_env:PointMazeEnv',
             kwargs={'sparse_reward': False, 'direction': 1})
    register(id='PointMazeLeft-v0', entry_point='inverse_rl.envs.point_maze_env:PointMazeEnv',
             kwargs={'sparse_reward': False, 'direction': 0})

    # A modified ant which flips over less and learns faster via TRPO
    register(id='CustomAnt-v0', entry_point='inverse_rl.envs.ant_env:CustomAntEnv',
             kwargs={'gear': 30, 'disabled': False})
    register(id='DisabledAnt-v0', entry_point='inverse_rl.envs.ant_env:CustomAntEnv',
             kwargs={'gear': 30, 'disabled': True})

    register(id='VisualPointMaze-v0', entry_point='inverse_rl.envs.visual_pointmass:VisualPointMazeEnv',
             kwargs={'sparse_reward': False, 'direction': 1})
