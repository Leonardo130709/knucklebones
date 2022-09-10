import dataclasses

from rltools.config import Config as BaseConfig


@dataclasses.dataclass
class Config(BaseConfig):
    # alg
    discount: float = .99
    reinforce_conf: float = 0.
    critic_loss_coef: float = 0.
    entropy_coef: float = 0.
    epsilon: float = 0.

    # architecture
    hidden_dim: int = 64
    row_encoder_layers: int = 1
    row_num_heads: int = 1
    col_encoder_layers: int = 1
    col_num_heads: int = 1
    board_emb_dim: int = 16
    critic_layers: int = 2
    actor_layers: int = 2
    activation: str = "elu"

    # train
    learning_rate: float = 1e-3
    batch_size: int = 2048
    max_grad: float = 40.
    eval_steps: int = 10000
    seed: int = 0
    port: int = 41922