from typing import Any, NamedTuple
import pickle

import jax
import jax.numpy as jnp
import haiku as hk
import optax
import reverb
import chex

from rltools.loggers import TFSummaryLogger
from .networks import Networks
from .config import Config


class LearnerState(NamedTuple):
    params: Any
    optim_state: Any


class Learner:
    def __init__(self,
                 rng_key: jax.random.PRNGKey,
                 config: Config,
                 networks: Networks,
                 optim: optax.GradientTransformation,
                 dataset,
                 client: reverb.Client
                 ):
        params = networks.init(rng_key)
        print("Learnable params: ", hk.data_structures.tree_size(params))
        optim_state = optim.init(params)
        self._state = LearnerState(params, optim_state)
        self._config = config
        self._ds = dataset
        self._client = client
        self.gradient_steps = 0

        self._callback = TFSummaryLogger(
            self._config.logdir, "train", step_key="step")
        with open(f"{config.logdir}/config.pickle", "wb") as cfg_f:
            pickle.dump(config, cfg_f)

        @chex.assert_max_traces(n=2)
        def _step(learner_state, data):
            params, optim_state = learner_state
            states, actions, scores = map(
                data.get,
                ("states", "actions", "scores")
            )
            
            def loss(actor_params, states, actions, scores):
                logits = networks.actor(actor_params, states)
                dist = networks.make_dist(logits)
                log_probs = dist.log_prob(actions)

                entropy = dist.entropy()
                chex.assert_equal_shape([log_probs, scores, entropy])

                cross_entropy_loss = - jnp.mean(scores * log_probs)
                entropy_gain = jnp.mean(entropy)

                loss =\
                    cross_entropy_loss -\
                    config.entropy_coef * entropy_gain

                metrics = dict(
                    ce_loss=cross_entropy_loss,
                    entropy=entropy_gain
                )
                return loss, metrics

            grads_fn = jax.grad(loss, has_aux=True)
            grads, metrics = grads_fn(params, states, actions, scores)

            update, optim_state = optim.update(grads, optim_state)
            grad_norm = optax.global_norm(update)

            metrics.update(
                grad_norm=grad_norm,
                mean_score=jnp.mean(scores)
            )
            params = optax.apply_updates(params, update)

            return LearnerState(params, optim_state), metrics

        self._step = jax.jit(_step, donate_argnums=())

    def run(self):
        while True:
            sample = next(self._ds)
            info, data = sample
            data = jax.device_put(data)
            self._state, metrics = self._step(self._state, data)
            self.gradient_steps += 1
            metrics["step"] = self.gradient_steps
            self._callback.write(metrics)
            self._client.insert(self._state.params, priorities={"weights": 1.})
            with open(f"{self._config.logdir}/weights.pickle", "wb") as f:
                pickle.dump(self._state.params, f)
