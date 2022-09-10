import dataclasses

from rltools.config import Config as BaseConfig


@dataclasses.dataclass
class Config(BaseConfig):
    # alg
    discount: float = .99
    critic_loss_coef: float = 1.
    adv_clip: float = 5e-2
    entropy_coef: float = 1e-2
    epsilon: float = 0.

    # architecture
    hidden_dim: int = 128
    row_encoder_layers: int = 2
    row_num_heads: int = 1
    col_encoder_layers: int = 2
    col_num_heads: int = 1
    board_emb_dim: int = 32
    critic_layers: int = 2
    actor_layers: int = 2
    activation: str = "elu"

    # train
    weights_history: int = 5
    learning_rate: float = 1e-2
    batch_size: int = 2048
    max_grad: float = 10.
    eval_steps: int = 5000

    num_actors: int = 6
    seed: int = 0
    port: int = 41922
    logdir: str = "logdir/w_negative2"

