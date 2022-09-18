from typing import Any, NamedTuple
import pickle

import jax
import jax.scipy
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
    dual_params: Any
    optim_state: Any
    dual_optim_state: Any


class Learner:
    def __init__(self,
                 rng_key: jax.random.PRNGKey,
                 config: Config,
                 networks: Networks,
                 dataset,
                 client: reverb.Client,
                 shared_valued: "MPValues"
                 ):
        params = networks.init(rng_key)
        init_dual = jnp.log(jnp.exp(config.init_duals) - 1.)
        temperature = jnp.full((), init_dual, jnp.float32)
        print("Learnable params: ",
              hk.data_structures.tree_size((params, temperature))
              )

        optim = optax.chain(
            optax.clip_by_global_norm(config.max_grad),
            optax.adam(learning_rate=config.actor_critic_lr)
        )
        dual_optim = optax.adam(config.dual_lr)
        optim_state = optim.init(params)
        dual_optim_state = dual_optim.init(temperature)

        self._state = LearnerState(
            params=params,
            target_params=params,
            dual_params=temperature,
            optim_state=optim_state,
            dual_optim_state=dual_optim_state
        )
        self._config = config
        self._ds = dataset
        self._client = client
        self._grad_steps = shared_valued.gradient_steps

        self._callback = TFSummaryLogger(
            self._config.logdir, "train", step_key="step")
        with open(f"{config.logdir}/config.pickle", "wb") as cfg_f:
            pickle.dump(config, cfg_f)

        @chex.assert_max_traces(n=2)
        def _step(learner_state: LearnerState, data):
            params, target_params, dual_params, \
                optim_state, dual_optim_state = learner_state
            states, actions, scores = map(
                data.get,
                ("states", "actions", "scores")
            )

            def actor_loss_fn(
                    actor_params, target_params, states, actions, weights):
                logits = networks.actor(actor_params, states)
                dist = networks.make_dist(logits)
                target_logits = networks.actor(target_params, states)
                target_dist = networks.make_dist(target_logits)
                log_probs = dist.log_prob(actions)
                kl_div = tfd.kl_divergence(dist, target_dist)
                entropy = dist.entropy()

                chex.assert_equal_shape([log_probs, weights, entropy, kl_div])
                entropy = jnp.mean(entropy)
                kl_div = jnp.mean(kl_div)

                cross_entropy = - jnp.sum(weights * log_probs)

                metrics = dict(
                    ce_loss=cross_entropy,
                    entropy=entropy,
                    kl_div=kl_div
                )
                return cross_entropy, metrics

            def critic_loss_fn(
                    critic_params, states, scores):
                values = networks.critic(critic_params, states)
                chex.assert_equal_shape([values, scores])
                advantages = scores - values
                loss = jnp.square(advantages)
                return .5 * jnp.mean(loss), jax.lax.stop_gradient(advantages)

            def dual_loss_fn(temperature, advantages):
                tempered_adv = advantages / temperature
                normalized_weights = jax.nn.softmax(tempered_adv, axis=0)
                normalized_weights = jax.lax.stop_gradient(normalized_weights)

                log_batch_size = jnp.log(advantages.shape[0])
                adv_logsumexp = jax.scipy.special.logsumexp(
                    tempered_adv, axis=0)
                loss = config.epsilon_kl + adv_logsumexp - log_batch_size
                loss = loss * temperature
                return loss, normalized_weights

            def loss_fn(params, dual_params, target_params, states, actions, scores):
                critic_loss, adv = critic_loss_fn(
                    params, target_params, states, scores)
                eta = jax.nn.softplus(dual_params)
                dual_loss, normalized_weights = dual_loss_fn(
                    eta, adv)
                actor_loss, metrics = actor_loss_fn(
                    params, target_params, states, actions, normalized_weights
                )
                metrics.update(
                    critic_loss=critic_loss,
                    dual_loss=dual_loss,
                    temperature=eta,
                )
                return critic_loss + actor_loss + dual_loss, metrics

            grads_fn = jax.grad(loss_fn, argnums=(0, 1), has_aux=True)
            (params_grads, dual_params_grads), metrics = grads_fn(
                params, dual_params, target_params, states, actions, scores)

            params_updates, optim_state = optim.update(
                params_grads, optim_state)
            dual_updates, dual_optim_state = dual_optim.update(
                dual_params_grads, dual_optim_state)
            grad_norm = optax.global_norm(params_updates)

            metrics.update(
                grad_norm=grad_norm,
            )
            params = optax.apply_updates(params, params_updates)
            dual_params = optax.apply_updates(dual_params, dual_updates)

            target_params = optax.incremental_update(
                params,
                target_params,
                config.target_polyak
            )

            return LearnerState(
                params,
                target_params,
                dual_params,
                optim_state,
                dual_optim_state
            ), metrics

        self._step = jax.jit(_step, donate_argnums=())

    def run(self):
        while True:
            sample = next(self._ds)
            info, data = sample
            data = jax.device_put(data)
            for _ in range(self._config.num_sgd_steps):
                self._state, metrics = self._step(self._state, data)
                with self._grad_steps.get_lock():
                    self._grad_steps.value += 1
                    metrics["step"] = self._grad_steps.value
                self._callback.write(metrics)
            self._client.insert(self._state.target_params, priorities={"weights": 1.})
            with open(f"{self._config.logdir}/weights.pickle", "wb") as f:
                pickle.dump(self._state.params, f)
