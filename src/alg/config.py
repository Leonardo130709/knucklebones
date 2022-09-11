import dataclasses

from rltools.config import Config as BaseConfig


@dataclasses.dataclass
class Config(BaseConfig):
    # alg
    discount: float = .995
    critic_loss_coef: float = 1.
    adv_clip: float = 1e-2
    entropy_coef: float = 0.
    epsilon: float = .1

    # architecture
    hidden_dim: int = 64
    row_encoder_layers: int = 2
    row_num_heads: int = 1
    col_encoder_layers: int = 2
    col_num_heads: int = 1
    board_emb_dim: int = 16
    critic_layers: int = 2
    actor_layers: int = 2
    activation: str = "elu"

    # train
    learning_rate: float = 1e-4
    buffer_size: int = 4096
    batch_size: int = 256
    max_grad: float = 20.
    eval_steps: int = 5000
    eval_games: int = 200

    num_actors: int = 5
    seed: int = 0
    port: int = 41922
    logdir: str = "logdir/night"

