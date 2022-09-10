import multiprocessing as mp
from typing import NamedTuple

import jax
import jax.numpy as jnp
import haiku as hk
import optax
import reverb
import tensorflow as tf

from .networks import make_networks
from .actor import Actor
from .config import Config
from .learner import Learner
from src.game import GameState


class MultiprocessValues(NamedTuple):
    num_actors: mp.Value
    completed_games: mp.Value
    total_steps: mp.Value


class Builder:
    def __init__(self, config: Config):
        self.cfg = config
        rng = jax.random.PRNGKey(self.cfg.seed)
        self.actor_rng, self.learner_rng = jax.random.split(rng)
        self._actors_shared_values = MultiprocessValues(
            mp.Value('i', 0), mp.Value('i', 0), mp.Value('i', 0)
        )

    def make_actor(self):
        networks = make_networks(self.cfg)
        client = reverb.Client(f'localhost:{self.cfg.port}')
        self.actor_rng, rng = jax.random.split(self.actor_rng)
        return Actor(rng,
                     self.cfg,
                     networks,
                     client,
                     self._actors_shared_values
                     )

    def make_server(self):
        networks = make_networks(self.cfg)
        to_specs = lambda ar: tf.TensorSpec(ar.shape, dtype=ar.dtype)

        trajectory_signature = jax.tree_util.tree_map(
            to_specs,
            {
                "states": GameState.zeroes(),
                "actions": jnp.array(0, dtype=jnp.int32),
                "scores": jnp.array(0, dtype=jnp.float32)
             }
        )
        params = networks.init(jax.random.PRNGKey(0))
        params_signature = jax.tree_util.tree_map(
            to_specs,
            params
        )
        tables = [
            reverb.Table(
                name="weights",
                sampler=reverb.selectors.Lifo(),
                remover=reverb.selectors.Fifo(),
                max_size=1,
                rate_limiter=reverb.rate_limiters.MinSize(1),
                signature=params_signature
            ),
            reverb.Table(
                name="opponents_weights",
                sampler=reverb.selectors.Uniform(),
                remover=reverb.selectors.Fifo(),
                max_size=self.cfg.weights_history,
                rate_limiter=reverb.rate_limiters.MinSize(1),
                signature=params_signature
            ),
            reverb.Table(
                name="replay_buffer",
                sampler=reverb.selectors.Lifo(),
                remover=reverb.selectors.Fifo(),
                max_size=int(self.cfg.batch_size),
                rate_limiter=reverb.rate_limiters.SampleToInsertRatio(
                    1., self.cfg.batch_size, 1),
                signature=trajectory_signature
            )
        ]
        return reverb.Server(tables, self.cfg.port)

    def make_dataset_iterator(self):
        ds = reverb.TrajectoryDataset.from_table_signature(
            f"localhost:{self.cfg.port}",
            table="replay_buffer",
            max_in_flight_samples_per_worker=self.cfg.batch_size
        )
        ds = ds.batch(self.cfg.batch_size, drop_remainder=True)
        return ds.as_numpy_iterator()

    def make_learner(self):
        networks = make_networks(self.cfg)
        params = networks.init(jax.random.PRNGKey(0))
        client = reverb.Client(f'localhost:{self.cfg.port}')
        client.insert(params, priorities={
            "weights": 1.,
            "opponents_weights": 1.
        })
        ds = self.make_dataset_iterator()

        optim = optax.chain(
            optax.clip_by_global_norm(self.cfg.max_grad),
            optax.adam(self.cfg.learning_rate)
        )
        self.learner_rng, rng = jax.random.split(self.learner_rng)
        return Learner(rng,
                       self.cfg,
                       networks,
                       optim,
                       ds,
                       client)
