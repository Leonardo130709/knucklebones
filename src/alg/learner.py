from typing import Any, NamedTuple
import pickle

import jax
import jax.numpy as jnp
import optax
import reverb
import chex

from rltools.loggers import TFSummaryLogger, TerminalOutput
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
        optim_state = optim.init(params)
        self._state = LearnerState(params, optim_state)
        self._config = config
        self._ds = dataset
        self._client = client
        self.gradient_steps = 0
        self._callback = TFSummaryLogger('logdir', 'train', step_key='step')
        self._printer = TerminalOutput()

        @chex.assert_max_traces(n=3)
        def _step(state, data):
            params, optim_state = state
            states, actions, discounts = map(
                data.get,
                ("states", "actions", "discounts")
            )
            
            def policy_loss(actor_params, states, actions, advantages):
                logits, _ = networks.actor(actor_params, states)
                dist = networks.make_dist(logits)
                log_prob = dist.log_prob(actions)

                entropy = dist.entropy()
                chex.assert_equal_shape([log_prob, advantages, discounts, entropy])

                cross_entropy_loss = - jnp.mean(advantages * log_prob)
                entropy_loss = - jnp.mean(entropy)

                loss =\
                    cross_entropy_loss +\
                    config.entropy_coef * entropy_loss

                metrics = dict(
                    ce_loss=cross_entropy_loss,
                    entropy=-entropy_loss
                )
                return loss, metrics
            
            def value_loss(critic_params, states, scores):
                values = networks.critic(critic_params, states)
                chex.assert_equal_shape([values, discounts])
                loss = jnp.square(values - scores)
                return jnp.mean(loss), values

            def model_loss(params, states, actions, returns):
                critic_loss, values = value_loss(params, states, returns)
                adv = jnp.clip(
                    jax.lax.stop_gradient(returns - values),
                    a_min=-config.adv_clip,
                    a_max=config.adv_clip
                )
                actor_loss, metrics = policy_loss(params, states, actions, adv)
                loss = actor_loss + config.critic_loss_coef * critic_loss

                r2 = 1 - critic_loss / jnp.var(returns)
                chex.assert_rank(r2, 0)
                metrics.update(
                    dict(
                        critic_loss=critic_loss,
                        mean_value=jnp.mean(values),
                        r2=r2,
                        mean_score=jnp.mean(returns)
                    )
                )
                return loss, metrics

            grads, metrics = jax.grad(model_loss, has_aux=True)(params, states, actions, discounts)
            update, optim_state = optim.update(grads, optim_state)
            grad_norm = optax.global_norm(update)
            metrics['grad_norm'] = grad_norm
            params = optax.apply_updates(params, update)

            return LearnerState(params, optim_state), metrics

        self._step = jax.jit(_step)

    def run(self):
        while True:
            sample = next(self._ds)
            info, data = sample
            data = jax.device_put(data)
            self._state, metrics = self._step(self._state, data)
            self.gradient_steps += 1
            metrics["step"] = self.gradient_steps
            self._callback.write(metrics)
            self._printer.write(metrics)
            self._client.insert(self._state.params, priorities={"weights": 1.})
            with open("weights.pickle", "wb") as f:
                pickle.dump(self._state.params, f)
