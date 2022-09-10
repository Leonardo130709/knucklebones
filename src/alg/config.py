import dataclasses

from rltools.config import Config as BaseConfig


@dataclasses.dataclass
class Config(BaseConfig):
    # alg
    discount: float = .99
    critic_loss_coef: float = 10.
    adv_clip: float = 1e-1
    entropy_coef: float = 1e-2
    epsilon: float = .1

    # architecture
    hidden_dim: int = 64
    row_encoder_layers: int = 1
    row_num_heads: int = 1
    col_encoder_layers: int = 1
    col_num_heads: int = 1
    board_emb_dim: int = 32
    critic_layers: int = 2
    actor_layers: int = 2
    activation: str = "elu"

    # train
    weights_history: int = 5
    learning_rate: float = 1e-4
    batch_size: int = 1024
    max_grad: float = 1.
    eval_steps: int = 5000
    eval_games: int = 200

    num_actors: int = 6
    seed: int = 0
    port: int = 41922
    logdir: str = "logdir/w_negative4"

