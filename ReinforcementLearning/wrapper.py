from gym import spaces
import gym 
import gym_minigrid

CELL_PIXELS = 32

class RGBImgObsWrapper(gym.core.ObservationWrapper):
    """
    Wrapper to use fully observable RGB image as the only observation output,
    no language/mission. This can be used to have the agent to solve the
    gridworld in pixel space.
    """

    def __init__(self, env):
        super().__init__(env)

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.env.width*CELL_PIXELS, self.env.height*CELL_PIXELS, 3),
            dtype='uint8'
        )
        

    def observation(self, obs):
        env = self.unwrapped
        return {'obs': obs['image'], 'image': env.render(mode = 'rgb_array', highlight = False)}


class ImgObsWrapper(gym.core.ObservationWrapper):
    """
    Use the image as the only observation output, no language/mission.
    """

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = env.observation_space.spaces['image']

    def observation(self, obs):
        return obs['image']