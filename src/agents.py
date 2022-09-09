import abc

import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk

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
    def __call__(self, state):
        _am = lambda x: np.argmax(x, axis=-1)
        print("Opp_board:")
        print(_am(state.opponent_board).T)
        print("Player board:")
        print(_am(state.player_board).T)
        print("Dice:")
        print(_am(state.dice))
        return int(input())


class NeuralAgent(Agent):
    def __init__(self, rng_key, policy_fn, params):
        self._policy_fn = policy_fn
        self._params = params
        self._rng = hk.PRNGSequence(
            jax.random.PRNGKey(rng_key)
        )

    def __call__(self, state):
        state = jax.device_put(state)
        dist = self._policy_fn(self._params, next(self._rng), state)
        # if training
        return 0
