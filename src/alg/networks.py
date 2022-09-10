from typing import NamedTuple, Callable

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
    gelu=jax.nn.gelu
)


class TransformerLayer(hk.Module):
    def __init__(
            self,
            hidden_dim: int,
            num_heads: int,
            activation: str
    ):
        super().__init__()
        self._hidden_dim = hidden_dim
        self._num_heads = num_heads
        self._activation = activation

    def __call__(self, x, mask=None):
        att = self._attention_path(x, mask)
        x = ln_factory()(x + att)
        dense = self._dense_path(x)

        return ln_factory()(x + dense)

    def _attention_path(self, x, mask):
        x = hk.Linear(3 * self._hidden_dim)(x)
        qkv = jnp.split(x, 3, axis=-1)
        att = hk.MultiHeadAttention(
            self._num_heads,
            self._hidden_dim,
            w_init_scale=1.
        )(*qkv, mask=mask)

        return hk.Linear(self._hidden_dim)(att)

    def _dense_path(self, x):
        return hk.nets.MLP(
            2 * [self._hidden_dim],
            w_init=hk.initializers.VarianceScaling(),
            b_init=jnp.zeros,
            activation=_ACTIVATION[self._activation],
        )(x)


class Transformer(hk.Module):
    def __init__(
            self,
            hidden_dim: int,
            num_heads: int,
            num_layers: int,
            activation: str
    ):
        super().__init__()
        self._hidden_dim = hidden_dim
        self._num_heads = num_heads
        self._num_layers = num_layers
        self._activation = activation

    def __call__(self, x, mask=None):
        x = hk.Linear(self._hidden_dim)(x)
        x = ln_factory()(x)

        for _ in range(self._num_layers):
            x = TransformerLayer(
                self._hidden_dim,
                self._num_heads,
                self._activation
            )(x, mask)

        return x

    
class BoardEncoder(hk.Module):
    def __init__(
            self,
            hidden_dim: int,
            row_encoder_layer: int,
            row_num_heads: int,
            col_encoder_layer: int,
            col_num_heads: int,
            board_emb_dim: int,
            activation: str
    ):
        super().__init__()
        self._board_emb_dim = board_emb_dim
        self._row_encoder = Transformer(
            hidden_dim,
            row_num_heads,
            row_encoder_layer,
            activation
        )
        self._col_encoder = Transformer(
            hidden_dim,
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

        return hk.Linear(self._board_emb_dim)(board)
        
        
class Actor(hk.Module):
    def __init__(
            self,
            act_dim: int,
            hidden_dim: int,
            layers: int,
            activation: str
    ):
        super().__init__()
        self.act_dim = act_dim
        self._layers = layers
        self._hidden_dim = hidden_dim
        self._act = activation

    def __call__(self, state):
        flatten_state = jnp.concatenate([
            state.player_board,
            state.opponent_board,
            state.player_col_scores,
            state.opponent_col_scores,
            state.dice
        ], axis=-1)
        flatten_state = hk.nets.MLP(
            self._layers * [self._hidden_dim],
            w_init=hk.initializers.VarianceScaling(),
            activation=_ACTIVATION[self._act],
            activate_final=True
        )(flatten_state)

        logits = hk.Linear(
            self.act_dim,
            w_init=jnp.zeros
        )(flatten_state)

        return logits


class Networks(NamedTuple):
    init: Callable
    encoder: Callable
    actor: Callable
    critic: Callable
    make_dist: Callable


def make_networks(cfg: Config):
    dummy_state = GameState.zeroes()
    dummy_state = jax.tree_util.tree_map(jnp.float32, dummy_state)

    @hk.without_apply_rng
    @hk.multi_transform
    def factory():
        encoder = BoardEncoder(
            cfg.hidden_dim,
            cfg.row_encoder_layers,
            cfg.row_num_heads,
            cfg.col_encoder_layers,
            cfg.col_num_heads,
            cfg.board_emb_dim,
            cfg.activation
        )
        actor = Actor(COLUMNS, cfg.hidden_dim, cfg.actor_layers, cfg.activation)
        critic = hk.nets.MLP(
            cfg.critic_layers * [cfg.hidden_dim] + [1],
            w_init=hk.initializers.VarianceScaling(),
            activation=_ACTIVATION[cfg.activation],
            name="critic"
        )

        def init():
            val = value_fn(dummy_state)
            logits = actor_fn(dummy_state)
            return val, logits

        def encoder_fn(state: GameState):
            state = state._replace(
                player_board=encoder(state.player_board),
                opponent_board=encoder(state.opponent_board),
                player_col_scores=state.player_col_scores / MAX_COLUMN_SCORE,
                opponent_col_scores=state.opponent_col_scores / MAX_COLUMN_SCORE
            )
            return state

        def actor_fn(state: GameState):
            state = encoder_fn(state)
            return actor(state)

        def value_fn(state: GameState):
            state = encoder_fn(state)
            flatten_state = jnp.concatenate(
                [
                    state.player_board,
                    state.opponent_board,
                    # state.player_col_scores,
                    # state.opponent_col_scores,
                    # state.dice
                ],
                axis=-1
            )
            v = critic(flatten_state)
            return jnp.squeeze(v, axis=-1)

        return init, (encoder_fn, actor_fn, value_fn)

    def make_dist(logits):
        return tfd.Categorical(logits)

    encoder, actor, critic = factory.apply
    return Networks(
        init=factory.init,
        encoder=encoder,
        actor=actor,
        critic=critic,
        make_dist=make_dist
    )
