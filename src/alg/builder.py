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


class MPValues(NamedTuple):
    num_actors: mp.Value
    completed_games: mp.Value
    total_steps: mp.Value
    gradient_steps: mp.Value


class Builder:
    def __init__(self, config: Config):
        self.cfg = config
        rng = jax.random.PRNGKey(self.cfg.seed)
        self.actor_rng, self.learner_rng = jax.random.split(rng)
        self._shared_values = MPValues(*[mp.Value("i", 0) for _ in range(4)])

    def make_actor(self):
        networks = make_networks(self.cfg)
        client = reverb.Client(f"localhost:{self.cfg.port}")
        self.actor_rng, rng = jax.random.split(self.actor_rng)
        return Actor(
            rng,
            self.cfg,
            networks,
            client,
            self._shared_values
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
            # reverb.Table(
            #     name="replay_buffer",
            #     sampler=reverb.selectors.Uniform(),
            #     remover=reverb.selectors.Fifo(),
            #     max_times_sampled=2,
            #     max_size=self.cfg.buffer_size,
            #     rate_limiter=reverb.rate_limiters.SampleToInsertRatio(
            #         2.,
            #         self.cfg.batch_size,
            #         .1 * self.cfg.batch_size
            #     ),
            #     signature=trajectory_signature
            # ),
            reverb.Table.queue(
                name="replay_buffer",
                max_size=self.cfg.buffer_size,
                signature=trajectory_signature
            )
        ]
        server = reverb.Server(tables, self.cfg.port)
        client = reverb.Client(f"localhost:{self.cfg.port}")
        client.insert(params, priorities={"weights": 1.})

        return server

    def make_dataset_iterator(self):
        ds = reverb.TrajectoryDataset.from_table_signature(
            f"localhost:{self.cfg.port}",
            table="replay_buffer",
            max_in_flight_samples_per_worker=2*self.cfg.batch_size
        )
        ds = ds.batch(self.cfg.batch_size, drop_remainder=True)
        ds = ds.prefetch(5)
        return ds.as_numpy_iterator()

    def make_learner(self):
        networks = make_networks(self.cfg)
        client = reverb.Client(f"localhost:{self.cfg.port}")
        ds = self.make_dataset_iterator()
        self.learner_rng, rng = jax.random.split(self.learner_rng)
        return Learner(
            rng,
            self.cfg,
            networks,
            ds,
            client,
            self._shared_values
        )
