from typing import Any, NamedTuple
import pickle

import jax
import jax.scipy
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
    target_params: Any
    optim_state: Any

    ema_state: Any
    rng_key: jax.random.PRNGKey
    step: int


class Learner:
    def __init__(self,
                 rng_key: jax.random.PRNGKey,
                 config: Config,
                 networks: Networks,
                 dataset,
                 client: reverb.Client,
                 shared_valued: "MPValues"
                 ):

        rng_key, k1, k2 = jax.random.split(rng_key, 3)
        params = networks.init(k1)
        print("Learnable params: ", hk.data_structures.tree_size(params))

        @hk.without_apply_rng
        @hk.transform_with_state
        def normalization_fn(x):
            mean = jnp.mean(x)
            std = jnp.std(x)
            mean, std = hk.ExponentialMovingAverage(
                config.normalization_tau)((mean, std))
            return (x - mean) / jnp.fmax(1e-5, std)

        _, ema_state = normalization_fn.init(k2, 0)

        optim = optax.chain(
            optax.clip_by_global_norm(config.max_grad),
            optax.adam(learning_rate=config.actor_critic_lr)
        )
        optim_state = optim.init(params)
        self._state = LearnerState(
            params=params,
            target_params=params,
            optim_state=optim_state,
            ema_state=ema_state,
            rng_key=rng_key,
            step=0
        )
        self._config = config
        self._ds = dataset
        self._client = client
        self._grad_steps = shared_valued.gradient_steps

        self._callback = TFSummaryLogger(
            self._config.logdir, "train", step_key="step")
        with open(f"{config.logdir}/config.pickle", "wb") as cfg_f:
            pickle.dump(config, cfg_f)

        def ppo_loss(
                params,
                states,
                actions,
                target_values,
                behaviour_advantages,
                behaviour_log_probs
        ):
            logits = networks.actor(params, states)
            dist = networks.make_dist(logits)
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy()

            chex.assert_equal_shape(
                [log_probs, behaviour_log_probs,
                 behaviour_advantages, entropy]
            )
            entropy = jnp.mean(entropy)

            rhos = jnp.exp(log_probs - behaviour_log_probs)
            clipped_rhos = jnp.clip(
                    rhos,
                    1 - config.ppo_clipping_epsilon,
                    1 + config.ppo_clipping_epsilon
                )
            clipped_values = jnp.fmin(
                rhos * behaviour_advantages,
                clipped_rhos * behaviour_advantages
            )
            ppo_loss = - jnp.mean(clipped_values)
            policy_loss = ppo_loss + config.entropy_coef * entropy

            values = networks.critic(params, states)
            chex.assert_equal_shape([values, target_values])
            critic_loss = jnp.square(target_values - values)
            critic_loss = .5 * jnp.mean(critic_loss)

            total_loss = policy_loss + config.critic_coef * critic_loss

            return total_loss, {
                "ppo_loss": ppo_loss,
                "entropy": entropy,
                "critic_loss": critic_loss
            }

        @chex.assert_max_traces(n=2)
        def _step(learner_state, data):
            params, target_params, optim_state, \
            ema_state, rng_key, step = learner_state
            states, actions, returns = map(
                data.get,
                ("states", "actions", "scores")
            )

            target_values = networks.critic(target_params, states)
            target_logits = networks.actor(target_params, states)
            target_dist = networks.make_dist(target_logits)
            target_log_probs = target_dist.log_prob(actions)
            advantages = returns - target_values
            normalized_advantages, ema_state = normalization_fn.apply(
                {}, ema_state, advantages)

            grad_fn = jax.grad(ppo_loss, has_aux=True)

            grads, metrics = grad_fn(
                params, states, actions, returns,
                normalized_advantages, target_log_probs
            )

            updates, optim_state = optim.update(
                grads, optim_state)
            grad_norm = optax.global_norm(updates)
            metrics.update(
                grad_norm=grad_norm,
            )
            params = optax.apply_updates(params, updates)

            target_params = optax.periodic_update(
                params,
                target_params,
                step,
                config.targets_update_every
            )

            return LearnerState(
                params,
                target_params,
                optim_state,
                ema_state,
                rng_key,
                step+1
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
