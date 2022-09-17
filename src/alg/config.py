import dataclasses

from rltools.config import Config as BaseConfig


@dataclasses.dataclass
class Config(BaseConfig):
    # alg
    discount: float = .99
    entropy_coef: float = 0.
    kl_coef: float = 0.

    # architecture
    #   board transformer
    board_emb_dim: int = 32
    attention_dim: int = 16
    row_encoder_layers: int = 2
    row_num_heads: int = 1
    col_encoder_layers: int = 2
    col_num_heads: int = 1
    #   actor-critic
    actor_layers: tuple = (64, 64)
    critic_layers: tuple = (64, 64)
    activation: str = "swish"

    # train
    learning_rate: float = 1e-4
    batch_size: int = 1024
    buffer_size: int = 2048
    target_polyak: float = 5e-3
    max_grad: float = 10.
    eval_steps: int = 5000
    eval_games: int = 500

    num_actors: int = 32
    seed: int = 0
    port: int = 41922
    logdir: str = "logdir/w_critic"

