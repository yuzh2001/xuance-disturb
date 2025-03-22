import numpy as np
from pettingzoo.sisl import multiwalker_v9

from xuance.environment import RawMultiAgentEnv


class MultiWalkerEnv(RawMultiAgentEnv):
    """
    The implementation of Multiwalker environments from PettingZoo.
    This environment involves multiple bipedal robots working together to carry a package.

    Parameters:
        config: The configurations of the environment, including:
            - n_walkers: number of bipedal walker agents
            - position_noise: noise applied to neighbor observations
            - angle_noise: noise applied to angle observations
            - forward_reward: reward scaling for moving forward
            - terminate_reward: reward when environment terminates
            - fall_reward: reward applied when walker falls
            - shared_reward: whether to share rewards among agents
            - terminate_on_fall: whether to terminate when a walker falls
            - remove_on_fall: whether to remove fallen walkers
            - terrain_length: length of terrain
    """

    def __init__(self, config):
        super(MultiWalkerEnv, self).__init__()
        # Prepare raw environment
        self.render_mode = config.render_mode
        self.continuous_actions = True  # Multiwalker has continuous action space

        # Initialize environment with config parameters
        self.env = multiwalker_v9.parallel_env(
            n_walkers=int(getattr(config, "n_walkers", 3)),
            position_noise=float(getattr(config, "position_noise", 1e-3)),
            angle_noise=float(getattr(config, "angle_noise", 1e-3)),
            forward_reward=float(getattr(config, "forward_reward", 1.0)),
            terminate_reward=float(getattr(config, "terminate_reward", -100.0)),
            fall_reward=float(getattr(config, "fall_reward", -10.0)),
            shared_reward=getattr(config, "shared_reward", True),
            terminate_on_fall=getattr(config, "terminate_on_fall", True),
            remove_on_fall=getattr(config, "remove_on_fall", True),
            terrain_length=int(getattr(config, "terrain_length", 200)),
            render_mode=self.render_mode,
        )
        self.env.reset(config.env_seed)

        # Set basic attributes
        self.metadata = self.env.metadata
        self.agents = self.env.agents
        self.num_agents = getattr(config, "n_walkers", 3)
        self.state_space = self.env.state_space

        # Set spaces
        self.observation_space = {
            agent: self.env.observation_space(agent) for agent in self.agents
        }
        self.action_space = {
            agent: self.env.action_space(agent) for agent in self.agents
        }

        # Additional attributes
        self.max_episode_steps = getattr(config, "max_cycles", 500)
        self.individual_episode_reward = {k: 0.0 for k in self.agents}
        self._episode_step = 0

    def close(self):
        """Close the environment."""
        self.env.close()

    def render(self, *args):
        """Get the rendered images of the environment."""
        return self.env.render()

    def reset(self):
        """Reset the environment to its initial state."""
        observations, infos = self.env.reset()
        for agent_key in self.agents:
            self.individual_episode_reward[agent_key] = 0.0
        reset_info = {
            "infos": infos,
            "individual_episode_rewards": self.individual_episode_reward,
        }
        self._episode_step = 0
        return observations, reset_info

    def step(self, actions):
        """Take an action as input, perform a step in the environment."""
        # Clip continuous actions to action space bounds
        for k, v in actions.items():
            actions[k] = np.clip(v, self.action_space[k].low, self.action_space[k].high)
        observations, rewards, terminated, truncated, info = self.env.step(actions)
        for k, v in rewards.items():
            self.individual_episode_reward[k] += v
        step_info = {
            "infos": info,
            "individual_episode_rewards": self.individual_episode_reward,
        }
        self._episode_step += 1
        truncated = True if self._episode_step >= self.max_episode_steps else False
        return observations, rewards, terminated, truncated, step_info

    def state(self):
        """Returns the global state of the environment."""
        return self.env.state()

    def agent_mask(self):
        """
        Create a boolean mask indicating which agents are currently alive.
        For Multiwalker, agents may be removed if remove_on_fall is True.
        """
        return {agent: True for agent in self.agents}

    def avail_actions(self):
        """
        Returns None for continuous action spaces in Multiwalker.
        """
        return None
