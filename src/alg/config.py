import dataclasses

from rltools.config import Config as BaseConfig


@dataclasses.dataclass
class Config(BaseConfig):
    # alg
    discount: float = 1.
    entropy_coef: float = 1e-4

    # architecture
    hidden_dim: int = 64
    attention_dim: int = 16
    row_encoder_layers: int = 2
    row_num_heads: int = 1
    col_encoder_layers: int = 2
    col_num_heads: int = 1
    board_emb_dim: int = 16
    actor_layers: int = 3
    activation: str = "swish"

    # train
    learning_rate: float = 1e-3
    batch_size: int = 512
    buffer_size: int = 4096
    max_grad: float = 10.
    eval_steps: int = 10000
    eval_games: int = 300

    num_actors: int = 5
    seed: int = 0
    port: int = 41922
    logdir: str = "logdir/lesser_sampler"

