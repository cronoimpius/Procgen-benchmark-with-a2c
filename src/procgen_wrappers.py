"""
Procgen Wrappers for its environments
"""

import random
import numpy as np
import torch 
import gym 

from abc import ABC, abstractmethod 
from gym import spaces 
from procgen import ProcgenEnv 

def make_env(n_env, name, start_level, num_levels,use_backgrounds=False):
    set_global_seed(0)
    set_global_levels(40)

    env = ProcgenEnv(
        num_envs=n_env, 
        env_name=name, 
        start_level=start_level,
        num_levels=num_levels,
        use_backgrounds=use_backgrounds,
        distribution_mode="easy",
        restrict_themes=not use_backgrounds,
        render_mode="rgb_array",
        rand_seed=0
    )
    env = VecExtractDictObs(env, "rgb")
    env = VecNormalize(env)
    env = TransposeFrame(env)
    env = ScaledFloatFrame(env)
    env = TensorEnv(env)

    return env

def set_global_seed(seed):
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False 
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def set_global_levels(lev):
    gym.logger.set_level(lev)

# The following part is done by watching the gym/wrappers in openai gym repository

class VecEnv(ABC):

    closed = False
    viewer = None

    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, num_envs, observation_space, action_space):
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.action_space = action_space

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step_async(self, actions):
        pass

    @abstractmethod
    def step_wait(self):
        pass

    def close_extras(self):
        pass

    def close(self):
        if self.closed:
            return
        if self.viewer is not None:
            self.viewer.close()
        self.close_extras()
        self.closed = True

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def render(self, mode="human"):
        if mode == "human":
            self.get_viewer().imshow("human")
            return self.get_viewer().isopen
        elif mode == "rgb_array":
            return "rgb_array"
        else:
            raise NotImplementedError

    def get_images(self):
        raise NotImplementedError

    @property
    def unwrapped(self):
        if isinstance(self, VecEnvWrapper):
            return self.venv.unwrapped
        else:
            return self

    def get_viewer(self):
        if self.viewer is None:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.SimpleImageViewer()
        return self.viewer


class VecEnvWrapper(VecEnv):
    """
    An environment wrapper that applies to an entire batch
    of environments at once.
    """

    def __init__(self, venv, observation_space=None, action_space=None):
        self.venv = venv
        super().__init__(
            num_envs=venv.num_envs,
            observation_space=observation_space or venv.observation_space,
            action_space=action_space or venv.action_space,
        )

    def step_async(self, actions):
        self.venv.step_async(actions)

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step_wait(self):
        pass

    def close(self):
        return self.venv.close()

    def render(self, mode="human"):
        return self.venv.render(mode=mode)

    def get_images(self):
        return self.venv.get_images()

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError("attempted to get missing private attribute '{}'".format(name))
        return getattr(self.venv, name)


class VecEnvObservationWrapper(VecEnvWrapper):
    @abstractmethod
    def process(self, obs):
        pass

    def reset(self):
        obs = self.venv.reset()
        return self.process(obs)

    def step_wait(self):
        obs, rews, dones, infos = self.venv.step_wait()
        return self.process(obs), rews, dones, infos


class CloudpickleWrapper(object):

    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle

        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle

        self.x = pickle.loads(ob)


class VecFrameStack(VecEnvWrapper):
    def __init__(self, venv, nstack):
        self.venv = venv
        self.nstack = nstack
        wos = venv.observation_space  # wrapped ob space
        low = np.repeat(wos.low, self.nstack, axis=-1)
        high = np.repeat(wos.high, self.nstack, axis=-1)
        self.stackedobs = np.zeros((venv.num_envs,) + low.shape, low.dtype)
        observation_space = spaces.Box(low=low, high=high, dtype=venv.observation_space.dtype)
        VecEnvWrapper.__init__(self, venv, observation_space=observation_space)

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        self.stackedobs = np.roll(self.stackedobs, shift=-1, axis=-1)
        for (i, new) in enumerate(news):
            if new:
                self.stackedobs[i] = 0
        self.stackedobs[..., -obs.shape[-1] :] = obs
        return self.stackedobs, rews, news, infos

    def reset(self):
        obs = self.venv.reset()
        self.stackedobs[...] = 0
        self.stackedobs[..., -obs.shape[-1] :] = obs
        return self.stackedobs


class VecExtractDictObs(VecEnvObservationWrapper):
    def __init__(self, venv, key):
        self.key = key
        super().__init__(venv=venv, observation_space=venv.observation_space.spaces[self.key])

    def process(self, obs):
        return obs[self.key]


class RunningMeanStd(object):
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, "float64")
        self.var = np.ones(shape, "float64")
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count
        )


def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count


class VecNormalize(VecEnvWrapper):
    """
    A vectorized wrapper that normalizes the observations
    and returns from an environment.
    """

    def __init__(
        self, venv, ob=True, ret=True, clipob=10.0, cliprew=10.0, gamma=0.99, epsilon=1e-8
    ):
        VecEnvWrapper.__init__(self, venv)

        self.ob_rms = RunningMeanStd(shape=self.observation_space.shape) if ob else None
        self.ret_rms = RunningMeanStd(shape=()) if ret else None

        self.clipob = clipob
        self.cliprew = cliprew
        self.ret = np.zeros(self.num_envs)
        self.gamma = gamma
        self.epsilon = epsilon

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        for i in range(len(infos)):
            infos[i]["reward"] = rews[i]
        self.ret = self.ret * self.gamma + rews
        obs = self.obfilt(obs)
        if self.ret_rms:
            self.ret_rms.update(self.ret)
            rews = np.clip(
                rews / np.sqrt(self.ret_rms.var + self.epsilon), -self.cliprew, self.cliprew
            )
        self.ret[news] = 0.0
        return obs, rews, news, infos

    def obfilt(self, obs):
        if self.ob_rms:
            self.ob_rms.update(obs)
            obs = np.clip(
                (obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon),
                -self.clipob,
                self.clipob,
            )
            return obs
        else:
            return obs

    def reset(self):
        self.ret = np.zeros(self.num_envs)
        obs = self.venv.reset()
        return self.obfilt(obs)


class TransposeFrame(VecEnvWrapper):
    def __init__(self, env):
        super().__init__(venv=env)
        obs_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(obs_shape[2], obs_shape[0], obs_shape[1]), dtype=np.float32
        )

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        return obs.transpose(0, 3, 1, 2), reward, done, info

    def reset(self):
        obs = self.venv.reset()
        return obs.transpose(0, 3, 1, 2)


class ScaledFloatFrame(VecEnvWrapper):
    def __init__(self, env):
        super().__init__(venv=env)
        obs_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=obs_shape, dtype=np.float32)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        return obs / 255.0, reward, done, info

    def reset(self):
        obs = self.venv.reset()
        return obs / 255.0


class TensorEnv(VecEnvWrapper):
    def __init__(self, env):
        super().__init__(venv=env)

    def step_async(self, actions):
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        return torch.Tensor(obs), reward, done, info

    def reset(self):
        obs = self.venv.reset()
        return torch.Tensor(obs)

