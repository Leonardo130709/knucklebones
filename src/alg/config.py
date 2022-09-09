import dataclasses

from rltools.config import Config as BaseConfig


@dataclasses.dataclass
class Config(BaseConfig):
    # alg
    discount: float = .99
    reinforce_conf: float = 1.
    critic_loss_coef: float = 1.
    entropy_coef: float = 1e-3
    epsilon: float = .1

    # architecture
    hidden_dim: int = 256
    row_encoder_layers: int = 2
    row_num_heads: int = 2
    col_encoder_layers: int = 2
    col_num_heads: int = 2
    board_emb_dim: int = 16
    critic_layers: int = 2
    actor_layers: int = 2
    activation: str = "relu"

    # train
    learning_rate: float = 1e-3
    batch_size: int = 4096
    max_grad: float = 10.
    seed: int = 0
    port: int = 41922
