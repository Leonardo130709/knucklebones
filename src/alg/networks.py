from typing import NamedTuple, Callable, Iterable

import jax
import jax.numpy as jnp
import haiku as hk
import tensorflow_probability.substrates.jax as tfp

from .config import Config
from src.consts import COLUMNS
from src.game import GameState
from src.consts import MAX_COLUMN_SCORE

tfd = tfp.distributions
MAX_COLUMN_SCORE = float(MAX_COLUMN_SCORE)


def ln_factory():
    return hk.LayerNorm(
        axis=-1,
        create_scale=True,
        create_offset=True
    )


_ACTIVATION = dict(
    relu=jax.nn.relu,
    elu=jax.nn.elu,
    gelu=jax.nn.gelu,
    swish=jax.nn.swish
)


class TransformerLayer(hk.Module):
    def __init__(
            self,
            attention_dim: int,
            num_heads: int,
            activation: str
    ):
        super().__init__()
        self._attention_dim = attention_dim
        self._num_heads = num_heads
        self._activation = activation

    def __call__(self, x):
        input_dim = x.shape[-1]
        att = self._attention_path(x, input_dim)
        x = ln_factory()(x + att)
        dense = self._dense_path(x, input_dim)

        return ln_factory()(x + dense)

    @hk.transparent
    def _attention_path(self, x, output_dim):
        x = hk.Linear(3 * self._attention_dim)(x)
        qkv = jnp.split(x, 3, axis=-1)
        att = hk.MultiHeadAttention(
            self._num_heads,
            self._attention_dim,
            w_init_scale=2.
        )(*qkv)
        return hk.Linear(output_dim)(att)

    @hk.transparent
    def _dense_path(self, x, output_dim):
        return hk.nets.MLP(
            2 * [output_dim],
            w_init=hk.initializers.VarianceScaling(),
            b_init=jnp.zeros,
            activation=_ACTIVATION[self._activation],
        )(x)


class Transformer(hk.Module):
    def __init__(
            self,
            feedforward_dim: int,
            attention_dim: int,
            num_heads: int,
            num_layers: int,
            activation: str
    ):
        super().__init__()
        self._feedforward_dim = feedforward_dim
        self._attention_dim = attention_dim
        self._num_heads = num_heads
        self._num_layers = num_layers
        self._activation = activation

    def __call__(self, x):
        x = hk.Linear(self._feedforward_dim)(x)

        for _ in range(self._num_layers):
            x = TransformerLayer(
                self._attention_dim,
                self._num_heads,
                self._activation
            )(x)

        return x

    
class BoardEncoder(hk.Module):
    def __init__(
            self,
            feedforward_dim: int,
            attention_dim: int,
            row_encoder_layer: int,
            row_num_heads: int,
            col_encoder_layer: int,
            col_num_heads: int,
            activation: str
    ):
        super().__init__()
        self._feedforward_dim = feedforward_dim
        self._row_encoder = Transformer(
            feedforward_dim,
            attention_dim,
            row_num_heads,
            row_encoder_layer,
            activation
        )
        self._col_encoder = Transformer(
            feedforward_dim,
            attention_dim,
            col_num_heads,
            col_encoder_layer,
            activation
        )
        
    def __call__(self, board):
        *prefix_shape, columns, rows, dice = board.shape

        board = self._row_encoder(board)
        board = jnp.reshape(board, prefix_shape + [columns, -1])

        board = self._col_encoder(board)
        board = jnp.reshape(board, prefix_shape + [-1])

        board = hk.Linear(self._feedforward_dim)(board)
        return ln_factory()(board)
        
        
class Actor(hk.Module):
    def __init__(
            self,
            act_dim: int,
            layers: Iterable[int],
            activation: str
    ):
        super().__init__()
        self.act_dim = act_dim
        self._layers = layers
        self._act = activation

    def __call__(self, state):
        state = hk.nets.MLP(
            self._layers,
            w_init=hk.initializers.VarianceScaling(),
            activation=_ACTIVATION[self._act],
            activate_final=True
        )(state)

        logits = hk.Linear(
            self.act_dim,
            w_init=jnp.zeros
        )(state)

        return logits


class Networks(NamedTuple):
    init: Callable
    actor: Callable
    critic: Callable
    make_dist: Callable


def make_networks(cfg: Config):
    dummy_state = GameState.zeroes()
    dummy_state = jax.tree_util.tree_map(jnp.float32, dummy_state)

    @hk.without_apply_rng
    @hk.multi_transform
    def forward():
        encoder = BoardEncoder(
            cfg.board_emb_dim,
            cfg.attention_dim,
            cfg.row_encoder_layers,
            cfg.row_num_heads,
            cfg.col_encoder_layers,
            cfg.col_num_heads,
            cfg.activation
        )
        actor = Actor(
            COLUMNS,
            cfg.actor_layers,
            cfg.activation
        )
        critic = hk.nets.MLP(
            cfg.critic_layers + (1,),
            w_init=hk.initializers.VarianceScaling(),
            activation=_ACTIVATION[cfg.activation],
            name="critic"
        )

        def encode_fn(state):
            return jnp.concatenate([
                encoder(state.player_board),
                encoder(state.opponent_board),
                state.player_col_scores / MAX_COLUMN_SCORE,
                state.opponent_col_scores / MAX_COLUMN_SCORE,
                state.dice
            ], axis=-1)

        def actor_fn(state):
            state = encode_fn(state)
            return actor(state)

        def critic_fn(state):
            state = encode_fn(state)
            value = critic(state)
            return jnp.squeeze(value, axis=-1)

        def init():
            return actor_fn(dummy_state), critic_fn(dummy_state)

        return init, (actor_fn, critic_fn)

    def make_dist(logits):
        return tfd.Categorical(logits)

    actor_fn, critic_fn = forward.apply
    return Networks(
        init=forward.init,
        actor=actor_fn,
        critic=critic_fn,
        make_dist=make_dist
    )
