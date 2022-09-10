from typing import NamedTuple
import numpy as np
from collections import defaultdict

from src.consts import ROWS, COLUMNS, DICEp1, BOARD_SHAPE


class BoardStatus:
    """The game ends when the board is filled completely."""
    Game = 0
    Fin = 1


class GameState(NamedTuple):
    """All the information required to make informed decision.
    However, action_mask can be inferred from the board."""

    player_board: np.ndarray
    opponent_board: np.ndarray
    player_col_scores: np.ndarray
    opponent_col_scores: np.ndarray
    dice: np.ndarray
    action_mask: np.ndarray
    
    @classmethod
    def zeroes(cls):
        return cls(
            player_board=np.zeros((COLUMNS, ROWS, DICEp1), dtype=np.uint8),
            opponent_board=np.zeros((COLUMNS, ROWS, DICEp1), dtype=np.uint8),
            player_col_scores=np.zeros(COLUMNS, dtype=np.uint8),
            opponent_col_scores=np.zeros(COLUMNS, dtype=np.uint8),
            action_mask=np.zeros(COLUMNS, dtype=np.bool_),
            dice=np.zeros(DICEp1, dtype=np.uint8)
        )


class Board:
    """Player board for the Knucklebones game."""

    def __init__(self):
        self.reset()

    def update(self, column: int, dice: int, enemy_turn: bool = False):
        """Main step."""
        if enemy_turn:
            self._on_enemy_turn(column, dice)
        else:
            idx = self._occupancy[column]
            self._state[column, idx] = dice
            self._occupancy[column] += 1

    def reset(self):
        """Initial state."""
        self._state = np.zeros(BOARD_SHAPE, dtype=np.uint8).T
        self._occupancy = np.zeros(COLUMNS, dtype=np.uint8)

    def _on_enemy_turn(self, column: int, enemy_dice: int):
        """Enemy can destroy dice."""
        col = self._state[column]

        equal_dice = col == enemy_dice
        self._occupancy[column] -= np.sum(equal_dice)

        mask = np.logical_or(equal_dice, col == 0)
        occ = self._occupancy[column]

        self._state[column, :occ] = col[~mask]
        self._state[column, occ:] = np.zeros_like(col[mask])

    def __repr__(self):
        return self.state.__repr__()

    def as_one_hot(self):
        """One-hot encoding for the board."""
        s = self._state
        ar = np.zeros((s.size, DICEp1), dtype=s.dtype)
        ar[range(s.size), s.ravel()] += 1
        return ar.reshape(BOARD_SHAPE + (DICEp1,))

    def evaluate(self):
        """Compute sums in each of the columns."""
        partial_sums = []
        for column in self._state:
            val, count = np.unique(column, return_counts=True)
            multiplier = count * count
            partial_sums.append(
                np.sum(val * multiplier, dtype=np.uint8)
            )

        return np.sum(partial_sums, dtype=np.uint8), partial_sums

    @property
    def possible_actions(self):
        """Unfinished columns."""
        return self._occupancy < ROWS

    @property
    def status(self):
        if np.any(self.possible_actions):
            return BoardStatus.Game
        else:
            return BoardStatus.Fin

    @property
    def state(self):
        return self._state.T


class Knucklebones:
    """Game of knucklebones for two players."""

    def __init__(self, actor0, actor1):
        self._players = [actor0, actor1]
        self._boards = None

    def reset(self):
        self._boards = [Board(), Board()]
        return np.random.randint(2)

    @staticmethod
    def roll_dice():
        dice = np.random.randint(1, DICEp1)
        one_hot = np.zeros(DICEp1, dtype=np.uint8)
        one_hot[dice] += 1
        return dice, one_hot

    def play(self):
        actions = defaultdict(list)
        states = defaultdict(list)

        turn = self.reset()
        first_turn = turn ^ 1
        steps = 0
        while self._boards[turn].status != BoardStatus.Fin:
            steps += 1
            turn = turn ^ 1
            dice, one_hot_dice = self.roll_dice()
            pl_board = self._boards[turn]
            opp_board = self._boards[turn ^ 1]
            _, pl_scores = pl_board.evaluate()
            _, opp_scores = opp_board.evaluate()

            state = GameState(
                player_board=pl_board.as_one_hot(),
                opponent_board=opp_board.as_one_hot(),
                player_col_scores=np.asarray(pl_scores),
                opponent_col_scores=np.asarray(opp_scores),
                action_mask=pl_board.possible_actions,
                dice=one_hot_dice
            )
            column = self._players[turn](state)

            states[turn].append(state)
            actions[turn].append(column)

            self._boards[0].update(column, dice, enemy_turn=turn == 1)
            self._boards[1].update(column, dice, enemy_turn=turn == 0)

        score0, column_scores0 = self._boards[0].evaluate()
        score1, column_scores1 = self._boards[1].evaluate()

        # Ambiguous win cond when scores are equal.
        winner = int(score1 > score0)
        winner_score = score1 if winner else score0

        return {
            'winner': winner,
            'total_score0': score0,
            'total_score1': score1,
            'length': steps,
            'fin_turn': turn,
            'first_turn': first_turn,
            'column_scores0': column_scores0,
            'column_scores1': column_scores1,
            'board0': self._boards[0].state,
            'board1': self._boards[1].state,
            'winner_states': states[winner],
            'winner_actions': actions[winner],
            'winner_score': winner_score
        }
