import dataclasses

from rltools.config import Config as BaseConfig


@dataclasses.dataclass
class Config(BaseConfig):
    # alg
    discount: float = 1.
    critic_loss_coef: float = 1.
    entropy_coef: float = 1e-4
    epsilon: float = .1

    # architecture
    hidden_dim: int = 32
    row_encoder_layers: int = 2
    row_num_heads: int = 1
    col_encoder_layers: int = 2
    col_num_heads: int = 1
    board_emb_dim: int = 16
    actor_layers: int = 2
    activation: str = "elu"

    # train
    learning_rate: float = 1e-3
    batch_size: int = 512
    buffer_size: int = 2048
    max_grad: float = 10.
    eval_steps: int = 10000
    eval_games: int = 200

    num_actors: int = 5
    seed: int = 0
    port: int = 41922
    logdir: str = "logdir/wo_critic"

