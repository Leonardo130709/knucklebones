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
        self._shared = shared_values

        self._ds = reverb.TimestepDataset.from_table_signature(
            client.server_address,
            table="weights",
            max_in_flight_samples_per_worker=1
        ).as_numpy_iterator()

        self._callback = TFSummaryLogger(
            self._config.logdir, "", step_key="step")
        self._printer = TerminalOutput()

        self._device = jax.devices("cpu")[shared_values.num_actors.value]
        shared_values.num_actors.value += 1

        self._opponents = {
            "rng": RandomAgent(),
            "self_train": lambda state: self.act(state, training=True),
            "self_eval": lambda state: self.act(state, training=False)
        }

        self._exploration_schedule = LinearDecay(
            min_value=self._config.min_epsilon,
            max_value=self._config.max_epsilon,
            total_steps=self._config.epsilon_decay_steps
        )

        @chex.assert_max_traces(n=2)
        def _act(params,
                 rng,
                 state: GameState,
                 epsilon,
                 training: bool
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
                    jax.random.uniform(k1) < epsilon,
                    jax.random.choice(k2, len(mask), p=mask),
                    dist.sample(seed=k3)
                )
            else:
                action = jnp.argmax(masked_logits, axis=-1)

            return action

        self._act = jax.jit(_act, device=self._device, static_argnums=(4,))

        self._env = Knucklebones(
            self._opponents["self_train"],
            self._opponents["self_eval"]
        )

    def act(self, state: GameState, training: bool):
        state = jax.device_put(state, self._device)
        rng = next(self._rng_seq)
        epsilon = self._exploration_schedule(
            self._shared.total_steps.value)
        action = self._act(self._params, rng, state, epsilon, training)
        return np.asarray(action)

    def _update_params(self):
        params = next(self._ds).data
        self._params = jax.device_put(params, self._device)
    
    def _evaluate_against(self, opponent):
        game = Knucklebones(
            opponent,
            self._opponents["self_eval"]
        )
        
        def _scalars_filter(d):
            keys = (
                "winner",
                "total_score0",
                "total_score1",
                "length",
                "winner_score",
                "scores_difference"
            )
            return {k: v for k, v in d.items() if k in keys}
        
        summary = [
            _scalars_filter(game.play()) for _ in range(self._config.eval_games)
        ]
        summary = jax.tree_util.tree_map(
            lambda *t: np.stack(t), *summary)
        summary = dict(
            wins=summary["winner"],
            score=summary["total_score1"],
            difference=summary["scores_difference"],
            length=summary["length"],
        )
        return jax.tree_util.tree_map(np.mean, summary)
            
    def evaluate(self):
        summaries = {}
        for opp in ("rng", "self_eval"):
            opp_summary = self._evaluate_against(self._opponents[opp])
            summaries.update(
                {f"{opp}/{k}": v for k, v in opp_summary.items()}
            )
        summaries["step"] = self._shared.total_steps.value
        self._callback.write(summaries)
        self._printer.write(summaries)

    def run(self):
        while True:
            self._update_params()
            summary = self._env.play()
            self._shared.completed_games.value += 1
            states, actions, steps = map(
                summary.get,
                ("winner_states", "winner_actions", "length")
            )
            self._shared.total_steps.value += steps

            discounts = self._config.discount ** \
                        np.arange(len(actions) - 1, -1, -1)
            scores = discounts.astype(np.float32)

            with self._client.trajectory_writer(num_keep_alive_refs=1) as writer:
                for state, action, score in zip(states, actions, scores):
                    writer.append({
                        "states": state,
                        "actions": action,
                        "scores": score
                    })
                    writer.create_item(
                        table="replay_buffer",
                        priority=1.,
                        trajectory=jax.tree_util.tree_map(
                            lambda t: t[-1],
                            writer.history
                        )
                    )
                    writer.flush(block_until_num_items=5)

            if self._shared.completed_games.value % self._config.eval_steps == 0:
                # todo: proper lock
                self.evaluate()


class LinearDecay:
    def __init__(self, min_value, max_value, total_steps):
        self._decay = (max_value - min_value) / float(total_steps)
        self._min = min_value
        self._max = max_value

    def __call__(self, step):
        return np.clip(self._max - self._decay * step,
                       a_min=self._min,
                       a_max=self._max
                       )
