import json
import os
import argparse
import warnings
import numpy as np
from PRIMAL2_Observer_H import PRIMAL2_Observer
from Observer_Builder import DummyObserver
import tensorflow as tf
from ACNet_H import ACNet
from Map_Generator import *
from Env_Builder import *
from scipy.ndimage import zoom
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore', category=Warning)

def get_max_steps(map_name):
    name = map_name.lower()

    if "20" in name or "40" in name:
        return 128
    elif "80" in name:
        return 192
    elif "160" in name:
        return 256

    # fallback (important)
    print(f"[WARN] Could not infer steps from name: {map_name}, defaulting to 256")
    return 256

def _is_movingai_only_maps(maps):
    """
    Detect MovingAI-only converter output.

    Supported shapes after np.load(..., allow_pickle=True):
      - (1, 2, H, W)
      - or a nested array where maps[0][0] and maps[0][1] are 2D arrays
    """
    if not isinstance(maps, np.ndarray):
        return False

    try:
        if len(maps) == 1:
            first = np.array(maps[0])
            if first.ndim >= 3 and first.shape[0] == 2:
                return np.array(first[0]).ndim == 2 and np.array(first[1]).ndim == 2

        if len(maps) == 2:
            return np.array(maps[0]).ndim == 2 and np.array(maps[1]).ndim == 2

    except Exception:
        return False

    return False


class RL_Planner(MAPFEnv):
    """
    result saved for NN Continuous Planner:
        target_reached      [int ]: num_target that is reached during the episode.
                                    Affected by timeout or non-solution
        computing_time_list [list]: a computing time record of each run of M*
        num_crash           [int ]: number of crash during the episode
        episode_status      [str ]: whether the episode is 'succeed', 'timeout' or 'no-solution'
        succeed_episode     [bool]: whether the episode is successful (i.e. no timeout, no non-solution) or not
        step_count          [int ]: num_step taken during the episode. The 64 timeout step is included
        frames              [list]: list of GIP frames
    """

    def __init__(self, observer, model_path, IsDiagonal=False, isOneShot=True, frozen_steps=0,
                 gpu_fraction=0.04):
        super().__init__(observer=observer, map_generator=DummyGenerator(), num_agents=1,
                         IsDiagonal=IsDiagonal, frozen_steps=frozen_steps, isOneShot=isOneShot)

        self._set_testType()
        self._set_tensorflow(model_path, gpu_fraction)

    def _set_testType(self):
        self.ACTION_COST, self.GOAL_REWARD, self.COLLISION_REWARD = 0, 0.5, 1
        self.test_type = 'oneShot' if self.isOneShot else 'continuous'
        self.method = '_' + self.test_type + 'RL'

    def _set_tensorflow(self, model_path, gpu_fraction):
        config = tf.ConfigProto(allow_soft_placement=True, device_count={"GPU": 0})
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = gpu_fraction
        self.sess = tf.Session(config=config)

        # MUST match trained model
        self.num_channels = 11

        self.network = ACNet(
            "global",
            a_size=5,
            trainer=None,
            TRAINING=False,
            NUM_CHANNEL=self.num_channels,
            OBS_SIZE=self.observer.observation_size,
            MAP_H = 43,
            MAP_W = 43,
            GLOBAL_NET_SCOPE="global"
        )

        saver = tf.train.Saver()

        checkpoint_file = os.path.join(model_path, "model-32800.cptk")
        print("Loading model from:", checkpoint_file)

        saver.restore(self.sess, checkpoint_file)

        self.agent_states = []
        for _ in range(self.num_agents):
            self.agent_states.append(self.network.state_init)

    def set_world(self):
        return

    def give_moving_reward(self, agentID):
        collision_status = self.world.agents[agentID].status
        if collision_status == 0:
            reward = self.ACTION_COST
            self.isStandingOnGoal[agentID] = False
        elif collision_status == 1:
            reward = self.ACTION_COST + self.GOAL_REWARD
            self.isStandingOnGoal[agentID] = True
            self.world.agents[agentID].dones += 1
        else:
            reward = self.ACTION_COST + self.COLLISION_REWARD
            self.isStandingOnGoal[agentID] = False
        self.individual_rewards[agentID] = reward

    def listValidActions(self, agent_ID, agent_obs):
        return

    def _reset(self, map_generator=None, worldInfo=None):
        self.map_generator = map_generator
        if worldInfo is not None:
            self.world = TestWorld(self.map_generator, world_info=worldInfo, isDiagonal=self.IsDiagonal,
                                   isConventional=False)
        else:
            self.world = World(self.map_generator, num_agents=self.num_agents, isDiagonal=self.IsDiagonal)
            raise UserWarning('you are using re-computing env mode')
        self.num_agents = self.world.num_agents
        self.observer.set_env(self.world)
        self.fresh = True
        if self.viewer is not None:
            self.viewer = None
        self.agent_states = []
        for i in range(self.num_agents):
            rnn_state = self.network.state_init
            self.agent_states.append(rnn_state)

    def step_greedily(self, o):
        def run_network(o):
            inputs, goal_pos = [], []

            for agentID in range(1, self.num_agents + 1):
                agent_obs = o[agentID]
                inputs.append(agent_obs[0])
                goal_pos.append(agent_obs[1])

            # compute heatmap for all agents
            heatmap = self.observer.get_global_heatmap()  # (H, W)
            heatmap_feed = np.stack([heatmap] * len(inputs))[:, :, :, np.newaxis]  # (N, H, W, 1)
            fixed_heatmaps = []
            # scale the map to match the neural network input
            for h in heatmap_feed:
                scale_x = 21 / h.shape[0]
                scale_y = 21 / h.shape[1]

                small = zoom(h[:, :, 0], (scale_x, scale_y), order=1)

                fixed_heatmaps.append(small[:, :, np.newaxis])

            heatmap_feed = np.array(fixed_heatmaps, dtype=np.float32)
            h3_vec = self.sess.run(
                [self.network.h3],
                feed_dict={
                    self.network.inputs      : inputs,
                    self.network.goal_pos    : goal_pos,
                    self.network.heatmap_input: heatmap_feed
                })
            h3_vec = h3_vec[0]
            rnn_out = []
            for a in range(0, self.num_agents):
                rnn_state = self.agent_states[a]
                lstm_output, state = self.sess.run(
                    [self.network.rnn_out, self.network.state_out],
                    feed_dict={
                        self.network.inputs      : [inputs[a]],
                        self.network.h3          : [h3_vec[a]],
                        self.network.heatmap_input: heatmap_feed[a:a+1],
                        self.network.state_in[0] : rnn_state[0],
                        self.network.state_in[1] : rnn_state[1]
                    })
                rnn_out.append(lstm_output[0])
                self.agent_states[a] = state

            policy_vec = self.sess.run(
                [self.network.policy],
                feed_dict={
                    self.network.rnn_out: rnn_out
                })
            policy_vec = policy_vec[0]
            action_dict = {agentID: np.argmax(policy_vec[agentID - 1]) 
                        for agentID in range(1, self.num_agents + 1)}
            return action_dict

        numCrashedAgents, computing_time = 0, 0

        start_time = time.time()
        action_dict = run_network(o)
        computing_time = time.time() - start_time

        next_o, reward = self.step_all(action_dict)

        for agentID in reward.keys():
            if reward[agentID] // 1 != 0:
                numCrashedAgents += 1
        assert numCrashedAgents <= self.num_agents

        return numCrashedAgents, computing_time, next_o

    def find_path(self, max_length, saveImage, time_limit=np.Inf):
        assert max_length > 0
        step_count, num_crash, computing_time_list, frames = 0, 0, [], []
        episode_status = 'no early stop'

        obs = self._observe()
        for step in range(1, max_length + 1):
            if saveImage:
                frames.append(self._render(mode='rgb_array'))
            numCrash_AStep, computing_time, obs = self.step_greedily(obs)

            computing_time_list.append(computing_time)
            num_crash += numCrash_AStep
            step_count = step

            if time_limit < computing_time:
                episode_status = "timeout"
                break

        if saveImage:
            frames.append(self._render(mode='rgb_array'))

        target_reached = 0
        for agentID in range(1, self.num_agents + 1):
            target_reached += self.world.getDone(agentID)
        return [target_reached,
                computing_time_list,
                num_crash,
                episode_status,
                episode_status == 'no early stop',
                step_count,
                frames]


class MstarContinuousPlanner(MAPFEnv):
    def __init__(self, IsDiagonal=False, frozen_steps=0):
        super().__init__(observer=DummyObserver(), map_generator=DummyGenerator(), num_agents=1,
                         IsDiagonal=IsDiagonal, frozen_steps=frozen_steps, isOneShot=False)
        self._set_testType()

    def set_world(self):
        return

    def give_moving_reward(self, agentID):
        collision_status = self.world.agents[agentID].status
        if collision_status == 0:
            reward = self.ACTION_COST
            self.isStandingOnGoal[agentID] = False
        elif collision_status == 1:
            reward = self.ACTION_COST + self.GOAL_REWARD
            self.isStandingOnGoal[agentID] = True
            self.world.agents[agentID].dones += 1
        else:
            reward = self.ACTION_COST + self.COLLISION_REWARD
            self.isStandingOnGoal[agentID] = False
        self.individual_rewards[agentID] = reward

    def listValidActions(self, agent_ID, agent_obs):
        return

    def _set_testType(self):
        self.ACTION_COST, self.GOAL_REWARD, self.COLLISION_REWARD = 0, 0.5, 1
        self.test_type = 'continuous'
        self.method = '_' + self.test_type + 'mstar'

    def _reset(self, map_generator=None, worldInfo=None):
        self.map_generator = map_generator
        if worldInfo is not None:
            self.world = TestWorld(self.map_generator, world_info=worldInfo, isDiagonal=self.IsDiagonal,
                                   isConventional=True)
        else:
            self.world = World(self.map_generator, num_agents=self.num_agents, isDiagonal=self.IsDiagonal)
        self.num_agents = self.world.num_agents
        self.observer.set_env(self.world)
        self.fresh = True
        if self.viewer is not None:
            self.viewer = None

    def find_path(self, max_length, saveImage, time_limit=300):
        def parse_path(path, step_count):
            on_goal = False
            path_step = 0
            while step_count < max_length and not on_goal:
                actions = {}
                for i in range(self.num_agents):
                    agent_id = i + 1
                    next_pos = path[path_step][i]
                    diff = tuple_minus(next_pos, self.world.getPos(agent_id))
                    actions[agent_id] = dir2action(diff)

                    if self.world.agents[agent_id].goal_pos == next_pos and not on_goal:
                        on_goal = True

                self.step_all(actions, check_col=False)
                if saveImage:
                    frames.append(self._render(mode='rgb_array'))

                step_count += 1
                path_step += 1
            return step_count if step_count <= max_length else max_length

        def compute_path_piece(time_limit):
            succeed = True
            start_time = time.time()
            path = self.expert_until_first_goal(inflation=3.0, time_limit=time_limit / 5.0)
            c_time = time.time() - start_time
            if c_time > time_limit or path is None:
                succeed = False
            return path, succeed, c_time

        assert max_length > 0
        frames, computing_time_list = [], []
        target_reached, step_count, episode_status = 0, 0, 'succeed'

        while step_count < max_length:
            path_piece, succeed_piece, c_time = compute_path_piece(time_limit)
            computing_time_list.append(c_time)
            if not succeed_piece:
                if c_time > time_limit:
                    episode_status = 'timeout'
                    break
                else:
                    episode_status = 'no-solution'
                    break
            else:
                step_count = parse_path(path_piece, step_count)

        for agentID in range(1, self.num_agents + 1):
            target_reached += self.world.getDone(agentID)

        return target_reached, computing_time_list, 0, episode_status, episode_status == 'succeed', step_count, frames


class ContinuousTestsRunner:
    def __init__(self, env_path, result_path, Planner, resume_testing=False, GIF_prob=0.):
        print('starting {}...'.format(self.__class__.__name__))
        self.env_path = env_path
        self.result_path = result_path
        self.resume_testing = resume_testing
        self.GIF_prob = float(GIF_prob)
        self.worker = Planner
        self.test_method = self.worker.method

        if not os.path.exists(self.result_path):
            os.mkdir(self.result_path)

    def read_single_env(self, name):
        root = self.env_path
        assert name.split('.')[-1] == 'npy'
        print('loading a single testing env...')
        if self.resume_testing:
            env_name = name[:name.rfind('.')]
            if os.path.exists(self.result_path + env_name + self.test_method + ".txt"):
                return None
        maps = np.load(root + name, allow_pickle=True)
        return maps

    def _apply_movingai_env(self, maps):
        """
        Support a MovingAI-only npy:
            maps[0][0] -> state_map (2D)
            maps[0][1] -> goals_map (2D)
        """
        state_map = np.array(maps[0][0])
        goals_map = np.array(maps[0][1])

        self.worker.world = World(
            map_generator=manual_generator(state_map, goals_map),
            num_agents=int(np.max(goals_map)),
            isDiagonal=self.worker.IsDiagonal
        )
        self.worker.num_agents = self.worker.world.num_agents
        self.worker.observer.set_env(self.worker.world)
        self.worker.fresh = True
        if self.worker.viewer is not None:
            self.worker.viewer = None

        self.worker.agent_states = []
        for _ in range(self.worker.num_agents):
            self.worker.agent_states.append(self.worker.network.state_init)

    def run_1_test(self, name, maps):

        if _is_movingai_only_maps(maps):
            self._apply_movingai_env(maps)
            state_shape = np.array(maps[0][0]).shape
        else:
            self.worker._reset(
                map_generator=manual_generator(maps[0][0], maps[0][1]),
                worldInfo=maps
            )
            state_shape = np.array(maps[0][0]).shape

        env_name = name[:name.rfind('.')]

        # ✅ NEW: use paper logic instead of env_size
        max_length = get_max_steps(env_name)

        print(f"[INFO] working on {env_name}")
        print(f"[INFO] map shape = {state_shape}, max_steps = {max_length}")

        results = dict()

        result = self.worker.find_path(
            max_length=int(max_length),
            saveImage=np.random.rand() < self.GIF_prob
        )

        target_reached, computing_time_list, num_crash, episode_status, succeed_episode, step_count, frames = result

        # ✅ DEBUG (important for verifying correctness)
        total_time = sum(computing_time_list) if computing_time_list else 0
        throughput = (target_reached / total_time) if total_time > 0 else 0

        print(f"[RESULT] targets={target_reached}, steps={step_count}/{max_length}, time={total_time:.4f}")
        print(f"[RESULT] throughput={throughput:.4f}, status={episode_status}")

        results['target_reached'] = target_reached
        results['computing time'] = computing_time_list
        results['num_crash'] = num_crash
        results['status'] = episode_status
        results['isSuccessful'] = succeed_episode
        results['steps'] = str(step_count) + '/' + str(max_length)

        self.make_gif(frames, env_name, self.test_method)
        self.write_files(results, env_name, self.test_method)

        return

    def make_gif(self, image, env_name, ext):
        if image:
            gif_name = self.result_path + env_name + ext + ".gif"
            images = np.array(image)
            make_gif(images, gif_name)

    def write_files(self, results, env_name, ext):
        txt_filename = self.result_path + env_name + ext + ".txt"
        f = open(txt_filename, 'w')
        f.write(json.dumps(results))
        f.close()


if __name__ == "__main__":
    import time

    model_path = '../../models/model_primal2_continuous_heatmap/'
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_path", default="./testing_result/")
    parser.add_argument("--env_path", default='./primal2_testing_envs50/')
    parser.add_argument("-r", "--resume_testing", default=True, help="resume testing")
    parser.add_argument("-g", "--GIF_prob", default=0., help="write GIF")
    parser.add_argument("-t", "--type", default='continuous', help="choose between oneShot and continuous")
    parser.add_argument("-p", "--planner", default='mstar', help="choose between mstar and RL")
    parser.add_argument("-n", "--mapName", default=None, help="single map name for multiprocessing")
    args = parser.parse_args()

    if args.planner == 'mstar':
        print("Starting {} {} tests...".format(args.planner, args.type))
        tester = ContinuousTestsRunner(args.env_path,
                                       args.result_path,
                                       Planner=MstarContinuousPlanner(),
                                       resume_testing=args.resume_testing,
                                       GIF_prob=args.GIF_prob)

    elif args.planner == 'RL':
        print("Starting {} {} tests...".format(args.planner, args.type))
        tester = ContinuousTestsRunner(args.env_path,
                                       args.result_path,
                                       Planner=RL_Planner(
                                           observer=PRIMAL2_Observer(observation_size=11, num_future_steps=3),
                                           model_path=model_path,
                                           isOneShot=False),
                                       resume_testing=args.resume_testing,
                                       GIF_prob=args.GIF_prob)
    else:
        raise NameError('invalid planner type')

    maps = tester.read_single_env(args.mapName)
    if maps is None:
        print(args.mapName, " already completed")
    else:
        tester.run_1_test(args.mapName, maps)