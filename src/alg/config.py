import dataclasses

from rltools.config import Config as BaseConfig


@dataclasses.dataclass
class Config(BaseConfig):
    # alg
    discount: float = .9
    init_duals: float = .1
    epsilon_kl: float = .01
    tv_constraint: float = 1.

    # architecture
    board_emb_dim: int = 32
    attention_dim: int = 16
    row_encoder_layers: int = 2
    row_num_heads: int = 1
    col_encoder_layers: int = 2
    col_num_heads: int = 1

    actor_layers: tuple = (64, 64)
    critic_layers: tuple = (64, 64)
    activation: str = "swish"

    # train
    actor_critic_lr: float = 1e-4
    dual_lr: float = 1e-2
    batch_size: int = 1024
    buffer_size: int = 2048
    target_polyak: float = 5e-3
    num_sgd_steps: int = 2
    max_grad: float = 10.
    eval_steps: int = 5000
    eval_games: int = 300

    num_actors: int = 5
    seed: int = 0
    port: int = 41922
    logdir: str = "logdir/no_kl"

