import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk
import reverb
import chex

from rltools.loggers import TFSummaryLogger, TerminalOutput

from .networks import Networks
from .config import Config
from src.game import GameState, Knucklebones
from src.agents import RandomAgent
from src.consts import MAX_BOARD_SCORE

MIN_LOGIT = -1e9


class Actor:
    def __init__(self,
                 rng_key: jax.random.PRNGKey,
                 config: Config,
                 networks: Networks,
                 client: reverb.Client,
                 shared_values
                 ):
        self._rng_seq = hk.PRNGSequence(rng_key)
        self._config = config
        self._nets = networks
        self._client = client
        self._params = None
        self._shared = shared_values
        self._ds = reverb.TimestepDataset.from_table_signature(
            client.server_address,
            table="weights",
            max_in_flight_samples_per_worker=2
        ).as_numpy_iterator()
        self._callback = TFSummaryLogger(
            self._config.logdir, "eval", step_key="step")
        self._printer = TerminalOutput()
        self._device = jax.devices("cpu")[shared_values.num_actors.value]
        shared_values.num_actors.value += 1

        @chex.assert_max_traces(n=2)
        def _act(params,
                 rng: jax.random.PRNGKey,
                 state: GameState, training: bool
                 ):
            logits = networks.actor(params, state)
            mask = state.action_mask
            masked_logits = jnp.where(
                mask,
                logits,
                MIN_LOGIT
            )
            if training:
                k1, k2, k3 = jax.random.split(rng, 3)
                dist = networks.make_dist(masked_logits)
                action = jax.lax.select(
                    jax.random.uniform(k1) < self._config.epsilon,
                    jax.random.choice(k2, mask.shape[0], (), p=mask),
                    dist.sample(seed=k3)
                )
            else:
                action = jnp.argmax(masked_logits, axis=-1)
            return action

        self._act = jax.jit(_act, device=self._device, static_argnums=(3,))
        policy = lambda state: self.act(state, training=True)
        self._env = Knucklebones(policy, policy)

    def act(self, state: GameState, training: bool):
        state = jax.device_put(state, self._device)
        rng = next(self._rng_seq)
        action = self._act(self._params, rng, state, training)
        return np.asarray(action)

    def get_params(self):
        params = next(self._ds).data
        return jax.device_put(params, self._device)

    def evaluate(self):
        rng_agent = RandomAgent()
        policy = lambda state: self.act(state, training=False)
        w_random = Knucklebones(rng_agent, policy)
        w_self = Knucklebones(policy, policy)

        w_rand_summaries = []
        w_self_summaries = []

        def _filter(d):
            keys = ("winner", "winner_score", "length")
            return {k: v for k, v in d.items() if k in keys}

        for i in range(200):
            w_rand_summaries.append(_filter(w_random.play()))
            w_self_summaries.append(_filter(w_self.play()))

        w_rand_summaries = jax.tree_util.tree_map(
            lambda *t: jnp.stack(t), *w_rand_summaries)
        w_self_summaries = jax.tree_util.tree_map(
            lambda *t: jnp.stack(t), *w_self_summaries)

        summaries = dict(
            step=self._shared.total_steps.value,
            w_random_wins=np.mean(w_rand_summaries["winner"]),
            w_random_score=np.mean(w_rand_summaries["winner_score"]),
            w_self_score=np.mean(w_self_summaries["winner_score"]),
            mean_length=np.mean(w_self_summaries["length"])
        )
        self._callback.write(summaries)
        self._printer.write(summaries)

    def run(self):
        writer = self._client.trajectory_writer(num_keep_alive_refs=1)
        while True:
            self._params = self.get_params()
            summary = self._env.play()
            self._shared.completed_games.value += 1
            states, actions, steps, score = map(
                summary.get,
                ("winner_states", "winner_actions", "length", "winner_score")
            )
            self._shared.total_steps.value += steps

            discounts = self._config.discount ** \
                        np.arange(len(actions) - 1, -1, -1)
            discounts = discounts.astype(np.float32)
            score = np.asarray(score / MAX_BOARD_SCORE, dtype=np.float32)

            for state, action, discount in zip(states, actions, discounts):
                writer.append({
                    "states": state,
                    "actions": action,
                    "discounts": discount
                })
                writer.create_item(
                    table="replay_buffer",
                    priority=score,
                    trajectory=jax.tree_util.tree_map(
                        lambda t: t[-1],
                        writer.history
                    )
                )
                writer.flush(block_until_num_items=10)

            if self._shared.completed_games.value % self._config.eval_steps == 0:
                self.evaluate()
