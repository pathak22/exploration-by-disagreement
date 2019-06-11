import os
import itertools
from collections import deque
from copy import copy

import gym
import numpy as np
from PIL import Image
from utils import save_np_as_mp4


def unwrap(env):
    if hasattr(env, "unwrapped"):
        return env.unwrapped
    elif hasattr(env, "env"):
        return unwrap(env.env)
    elif hasattr(env, "leg_env"):
        return unwrap(env.leg_env)
    else:
        return env


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = deque(maxlen=2)
        self._skip = skip

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        acc_info = {}
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            acc_info.update(info)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)

        return max_frame, total_reward, done, acc_info

    def reset(self):
        """Clear past frame buffer and init. to first obs. from inner env."""
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs


class ProcessFrame84(gym.ObservationWrapper):
    def __init__(self, env, crop=True):
        self.crop = crop
        super(ProcessFrame84, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

    def observation(self, obs):
        return ProcessFrame84.process(obs, crop=self.crop)

    @staticmethod
    def process(frame, crop=True):
        if frame.size == 210 * 160 * 3:
            img = np.reshape(frame, [210, 160, 3]).astype(np.float32)
        elif frame.size == 250 * 160 * 3:
            img = np.reshape(frame, [250, 160, 3]).astype(np.float32)
        elif frame.size == 224 * 240 * 3:  # mario resolution
            img = np.reshape(frame, [224, 240, 3]).astype(np.float32)
        elif frame.size == 84 * 84 * 3:  # unity maze
            img = frame
            img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
            x_t = np.reshape(img, [84, 84, 1])
            return x_t.astype(np.uint8)
        else:
            assert False, "Unknown resolution." + str(frame.size)
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
        size = (84, 110 if crop else 84)
        resized_screen = np.array(Image.fromarray(img).resize(size,
                                                              resample=Image.BILINEAR), dtype=np.uint8)
        x_t = resized_screen[18:102, :] if crop else resized_screen
        x_t = np.reshape(x_t, [84, 84, 1])
        return x_t.astype(np.uint8)


class ExtraTimeLimit(gym.Wrapper):
    def __init__(self, env, max_episode_steps=None):
        gym.Wrapper.__init__(self, env)
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = 0

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps > self._max_episode_steps:
            done = True
        return observation, reward, done, info

    def reset(self):
        self._elapsed_steps = 0
        return self.env.reset()


class AddRandomStateToInfo(gym.Wrapper):
    def __init__(self, env):
        """Adds the random state to the info field on the first step after reset
        """
        gym.Wrapper.__init__(self, env)

    def step(self, action):
        ob, r, d, info = self.env.step(action)
        if self.random_state_copy is not None:
            info['random_state'] = self.random_state_copy
            self.random_state_copy = None
        return ob, r, d, info

    def reset(self, **kwargs):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.random_state_copy = copy(self.unwrapped.np_random)
        return self.env.reset(**kwargs)


class MontezumaInfoWrapper(gym.Wrapper):
    ram_map = {
        "room": dict(
            index=3,
        ),
        "x": dict(
            index=42,
        ),
        "y": dict(
            index=43,
        ),
    }

    def __init__(self, env):
        super(MontezumaInfoWrapper, self).__init__(env)
        self.visited = set()
        self.visited_rooms = set()

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        ram_state = unwrap(self.env).ale.getRAM()
        for name, properties in MontezumaInfoWrapper.ram_map.items():
            info[name] = ram_state[properties['index']]
        pos = (info['x'], info['y'], info['room'])
        self.visited.add(pos)
        self.visited_rooms.add(info["room"])
        if done:
            info['mz_episode'] = dict(pos_count=len(self.visited),
                                      visited_rooms=copy(self.visited_rooms))
            self.visited.clear()
            self.visited_rooms.clear()
        return obs, rew, done, info

    def reset(self):
        return self.env.reset()


class MarioXReward(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.current_level = [0, 0]
        self.visited_levels = set()
        self.visited_levels.add(tuple(self.current_level))
        self.current_max_x = 0.

    def reset(self):
        ob = self.env.reset()
        self.current_level = [0, 0]
        self.visited_levels = set()
        self.visited_levels.add(tuple(self.current_level))
        self.current_max_x = 0.
        return ob

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        levellow, levelhigh, xscrollHi, xscrollLo = \
            info["levelLo"], info["levelHi"], info["xscrollHi"], info["xscrollLo"]
        currentx = xscrollHi * 256 + xscrollLo
        new_level = [levellow, levelhigh]
        if new_level != self.current_level:
            self.current_level = new_level
            self.current_max_x = 0.
            reward = 0.
            self.visited_levels.add(tuple(self.current_level))
        else:
            if currentx > self.current_max_x:
                delta = currentx - self.current_max_x
                self.current_max_x = currentx
                reward = delta
            else:
                reward = 0.
        if done:
            info["levels"] = copy(self.visited_levels)
            info["retro_episode"] = dict(levels=copy(self.visited_levels))
        return ob, reward, done, info


class UnityRoomCounterWrapper(gym.Wrapper):
    def __init__(self, env,use_ext_reward=True):
        gym.Wrapper.__init__(self, env)
        self.current_room = None
        self.visited_rooms = set()
        self.use_ext_reward = use_ext_reward

    def reset(self):
        ob = self.env.reset()
        self.current_room = None
        self.visited_rooms = set()
        return ob


    def step(self, action):
        ob, true_reward, done, info = self.env.step(action)
        reward = 0.0
        current_room = info["curRoom"]

        if self.current_room is None:
            self.current_room = current_room
            reward = 1.0
            self.visited_rooms.add(self.current_room)

        if current_room != self.current_room:
            self.current_room = current_room
            if self.current_room not in self.visited_rooms:
                reward = 1.0
                self.visited_rooms.add(self.current_room)
            else:
                reward = 0.0
        info = {"unity_rooms":copy(self.visited_rooms)}
        return ob, reward if not self.use_ext_reward else true_reward, done, info


class LimitedDiscreteActions(gym.ActionWrapper):
    KNOWN_BUTTONS = {"A", "B"}
    KNOWN_SHOULDERS = {"L", "R"}

    '''
    Reproduces the action space from curiosity paper.
    '''

    def __init__(self, env, all_buttons, whitelist=KNOWN_BUTTONS | KNOWN_SHOULDERS):
        gym.ActionWrapper.__init__(self, env)

        self._num_buttons = len(all_buttons)
        button_keys = {i for i in range(len(all_buttons)) if all_buttons[i] in whitelist & self.KNOWN_BUTTONS}
        buttons = [(), *zip(button_keys), *itertools.combinations(button_keys, 2)]
        shoulder_keys = {i for i in range(len(all_buttons)) if all_buttons[i] in whitelist & self.KNOWN_SHOULDERS}
        shoulders = [(), *zip(shoulder_keys), *itertools.permutations(shoulder_keys, 2)]
        arrows = [(), (4,), (5,), (6,), (7,)]  # (), up, down, left, right
        acts = []
        acts += arrows
        acts += buttons[1:]
        acts += [a + b for a in arrows[-2:] for b in buttons[1:]]
        self._actions = acts
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self, a):
        mask = np.zeros(self._num_buttons)
        for i in self._actions[a]:
            mask[i] = 1
        return mask


class FrameSkip(gym.Wrapper):
    def __init__(self, env, n):
        gym.Wrapper.__init__(self, env)
        self.n = n

    def step(self, action):
        done = False
        totrew = 0
        for _ in range(self.n):
            ob, rew, done, info = self.env.step(action)
            totrew += rew
            if done: break
        return ob, totrew, done, info


def make_mario_env(crop=True, frame_stack=True, clip_rewards=False):
    assert clip_rewards is False
    import gym
    import retro
    from baselines.common.atari_wrappers import FrameStack

    #gym.undo_logger_setup()
    env = retro.make('SuperMarioBros-Nes', 'Level1-1')
    buttons = env.buttons
    env = MarioXReward(env)
    env = FrameSkip(env, 4)
    env = ProcessFrame84(env, crop=crop)
    if frame_stack:
        env = FrameStack(env, 4)
    env = LimitedDiscreteActions(env, buttons)
    return env


class OneChannel(gym.ObservationWrapper):
    def __init__(self, env, crop=True):
        self.crop = crop
        super(OneChannel, self).__init__(env)
        assert env.observation_space.dtype == np.uint8
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

    def observation(self, obs):
        return obs[:, :, 2:3]


class RetroALEActions(gym.ActionWrapper):
    def __init__(self, env, all_buttons, n_players=1):
        gym.ActionWrapper.__init__(self, env)
        self.n_players = n_players
        self._num_buttons = len(all_buttons)
        bs = [-1, 0, 4, 5, 6, 7]
        actions = []

        def update_actions(old_actions, offset=0):
            actions = []
            for b in old_actions:
                for button in bs:
                    action = []
                    action.extend(b)
                    if button != -1:
                        action.append(button + offset)
                    actions.append(action)
            return actions

        current_actions = [[]]
        for i in range(self.n_players):
            current_actions = update_actions(current_actions, i * self._num_buttons)
        self._actions = current_actions
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self, a):
        mask = np.zeros(self._num_buttons * self.n_players)
        for i in self._actions[a]:
            mask[i] = 1
        return mask


class NoReward(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)

    def step(self, action):
        ob, rew, done, info = self.env.step(action)
        return ob, 0.0, done, info


def make_multi_pong(frame_stack=True):
    import gym
    import retro
    from baselines.common.atari_wrappers import FrameStack
    gym.undo_logger_setup()
    game_env = env = retro.make('Pong-Atari2600', players=2)
    env = RetroALEActions(env, game_env.BUTTONS, n_players=2)
    env = NoReward(env)
    env = FrameSkip(env, 4)
    env = ProcessFrame84(env, crop=False)
    if frame_stack:
        env = FrameStack(env, 4)

    return env


def make_unity_maze(env_id, seed=0, rank=0, expID=0, frame_stack=True,
        logdir=None, ext_coeff=1.0, recordUnityVid=False, **kwargs):
    import os
    import sys
    import time
    try:
        sys.path.insert(0, os.path.abspath("ml-agents/python/"))
        from unityagents import UnityEnvironment
        from unity_wrapper import GymWrapper
    except ImportError:
        print("Import error in unity environment. Ignore if not using unity.")
        pass
    from baselines.common.atari_wrappers import FrameStack
    # gym.undo_logger_setup()  # deprecated in new version of gym

    # max 20 workers per expID, max 30 experiments per machine
    if rank>=0 and rank<=200:
        time.sleep(rank * 2)
    env = UnityEnvironment(file_name='envs/' + env_id,
        worker_id=(expID % 60) * 200 + rank)
    maxsteps = 3000 if 'big' in env_id else 500
    env = GymWrapper(env, seed=seed, rank=rank, expID=expID, maxsteps=maxsteps,
        **kwargs)
    if "big" in env_id:
        env = UnityRoomCounterWrapper(env, use_ext_reward=(ext_coeff != 0.0))
    if rank == 1 and recordUnityVid:
        env = RecordBestScores(env, directory=logdir, freq=1)
    print('Loaded environment %s with rank %d\n\n' % (env_id, rank))

    # env = NoReward(env)
    # env = FrameSkip(env, 4)
    env = ProcessFrame84(env, crop=False)
    if frame_stack:
        env = FrameStack(env, 4)
    return env


class RecordBestScores(gym.Wrapper):
    def __init__(self, env, directory, freq=100):
        super(RecordBestScores, self).__init__(env)
        self.freq = freq
        self.frames = []
        self.highest_reward = None
        self.episodic_reward = 0.
        self.longest_length = 0
        self.directory = directory
        self.episode_number = 0
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

    def _step(self, action):
        state, reward, done, info = self.env.step(action)
        self.frames.append(self.env.render(mode='rgb_array'))
        self.episodic_reward += reward
        if done:
            if self.highest_reward == None:
                self.highest_reward = self.episodic_reward
                self._record_last_episode("high_score_")
            elif self.highest_reward < self.episodic_reward:
                self.highest_reward = self.episodic_reward
                self._record_last_episode("high_score_")
            elif self.episode_number % self.freq == 0:
                self._record_last_episode("random_")

            self.frames = []
            self.episodic_reward = 0
            self.episode_number += 1
        return state, reward, done, info

    def _record_last_episode(self, prefix=""):
        save_np_as_mp4(self.frames, os.path.join(self.directory, prefix+'replay{}.mp4'.format(self.episode_number)))


class StickyActionEnv(gym.Wrapper):
    def __init__(self, env, p=0.25):
        super(StickyActionEnv, self).__init__(env)
        self.p = p
        self.last_action = 0

    def reset(self):
        self.last_action = 0
        return self.env.reset()

    def step(self, action):
        if self.unwrapped.np_random.uniform() < self.p:
            action = self.last_action
        self.last_action = action
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, info
