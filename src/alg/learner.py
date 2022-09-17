from typing import Any, NamedTuple
import pickle

import jax
import jax.numpy as jnp
import haiku as hk
import optax
import reverb
import chex
import tensorflow_probability.substrates.jax.distributions as tfd

from rltools.loggers import TFSummaryLogger
from .networks import Networks
from .config import Config


class LearnerState(NamedTuple):
    params: Any
    target_params: Any
    optim_state: Any


class Learner:
    def __init__(self,
                 rng_key: jax.random.PRNGKey,
                 config: Config,
                 networks: Networks,
                 optim: optax.GradientTransformation,
                 dataset,
                 client: reverb.Client,
                 shared_valued: "MPValues"
                 ):
        params = networks.init(rng_key)
        print("Learnable params: ", hk.data_structures.tree_size(params))
        optim_state = optim.init(params)
        self._state = LearnerState(params, params, optim_state)
        self._config = config
        self._ds = dataset
        self._client = client
        self._grad_steps = shared_valued.gradient_steps

        self._callback = TFSummaryLogger(
            self._config.logdir, "train", step_key="step")
        with open(f"{config.logdir}/config.pickle", "wb") as cfg_f:
            pickle.dump(config, cfg_f)

        @chex.assert_max_traces(n=2)
        def _step(learner_state, data):
            params, target_params, optim_state = learner_state
            states, actions, scores = map(
                data.get,
                ("states", "actions", "scores")
            )
            
            def actor_loss_fn(actor_params, target_params, states, actions, advantages):
                logits = networks.actor(actor_params, states)
                target_logits = networks.actor(target_params, states)
                dist = networks.make_dist(logits)
                target_dist = networks.make_dist(target_logits)
                log_probs = dist.log_prob(actions)
                kl_div = tfd.kl_divergence(dist, target_dist)

                entropy = dist.entropy()
                chex.assert_equal_shape([log_probs, advantages, entropy, kl_div])

                cross_entropy = - jnp.mean(advantages * log_probs)
                entropy = jnp.mean(entropy)
                kl_loss = jnp.mean(kl_div)

                loss =\
                    cross_entropy \
                    + config.kl_coef * kl_loss \
                    - config.entropy_coef * entropy

                metrics = dict(
                    ce_loss=cross_entropy,
                    entropy=entropy,
                    kl_div=kl_loss
                )
                return loss, metrics

            def critic_loss_fn(params, target_params, states, target_values):
                values = networks.critic(params, states)
                chex.assert_equal_shape([values, target_values])
                loss = jnp.square(values - target_values)
                return .5 * jnp.mean(loss), values

            def loss_fn(params, target_params, states, actions, scores):
                critic_loss, values = critic_loss_fn(params, target_params, states, scores)
                advantages = jax.lax.stop_gradient(scores - values)
                actor_loss, metrics = actor_loss_fn(params, target_params, states, actions, advantages)
                metrics.update(critic_loss=critic_loss)
                return critic_loss + actor_loss, metrics

            grads_fn = jax.grad(loss_fn, has_aux=True)
            grads, metrics = grads_fn(
                params, target_params, states, actions, scores)

            update, optim_state = optim.update(grads, optim_state)
            grad_norm = optax.global_norm(update)

            metrics.update(
                grad_norm=grad_norm,
            )
            params = optax.apply_updates(params, update)
            target_params = optax.incremental_update(
                params,
                target_params,
                config.target_polyak
            )

            return LearnerState(params, target_params, optim_state), metrics

        self._step = jax.jit(_step, donate_argnums=())

    def run(self):
        while True:
            sample = next(self._ds)
            info, data = sample
            data = jax.device_put(data)
            self._state, metrics = self._step(self._state, data)
            with self._grad_steps.get_lock():
                self._grad_steps.value += 1
                metrics["step"] = self._grad_steps.value
            self._callback.write(metrics)
            self._client.insert(self._state.params, priorities={"weights": 1.})
            with open(f"{self._config.logdir}/weights.pickle", "wb") as f:
                pickle.dump(self._state.params, f)
