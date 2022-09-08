import dataclasses

from rltools.config import Config as BaseConfig


@dataclasses.dataclass
class Config(BaseConfig):
    # agent
    hidden_dim: int = 64
    row_encoder_layers: int = 2
    row_num_heads: int = 2
    col_encoder_layers: int = 2
    col_num_heads: int = 2
    board_emb_dim: int = 16
    critic_layers: int = 2
    actor_layers: int = 2
    activation: str = "relu"

    reinforce_conf: float = 1.
    critic_loss_coef: float = 1.
    epsilon: float = .1

    # train
    learning_rate: float = 1e-4
    max_grad: float = 10.
    seed: int = 0

    # reverb
    port: int = 41922
    batch_size: int = 2048
