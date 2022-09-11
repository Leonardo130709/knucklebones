import abc

import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk
from IPython.display import clear_output

from src.game import GameState


class Agent(abc.ABC):

    @abc.abstractmethod
    def __call__(self,
            state: GameState,
            ) -> int:
        """Observe boards and dice and take action."""


class RandomAgent(Agent):
    def __call__(self, state):
        mask = state.action_mask
        chance = mask / mask.sum()
        return np.random.choice(len(mask), p=chance)


class ManualControl(Agent):

    def __init__(self, clear_screen: bool = True):
        self._cls = clear_screen

    def __call__(self, state):
        if self._cls:
            clear_output()
        _am = lambda x: np.argmax(x, axis=-1)
        print("Opp_board:")
        print(_am(state.opponent_board).T)
        print("Player board:")
        print(_am(state.player_board).T)
        print("Dice:")
        print(_am(state.dice))
        return int(input())


class NeuralAgent(Agent):

    def __init__(self, logdir: str):
        import pickle
        with open(f"{logdir}/weights.pickle", "rb") as w:
            self._params = pickle.load(w)
        with open(f"{logdir}/config.pickle", "rb") as cfg:
            self._config = pickle.load(cfg)
        from src.alg.networks import make_networks
        self._nets = make_networks(self._config)
        self._device = jax.devices("cpu")[0]
        self._params = jax.device_put(self._params, self._device)

    def __call__(self, state):
        state = jax.device_put(state, self._device)
        logits = self._nets.actor(self._params, state)
        return int(jnp.argmax(logits, axis=-1))
