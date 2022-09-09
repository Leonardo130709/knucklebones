"""Game parametrization."""
DICE = 6
DICEp1 = DICE + 1  # plus empty tile.
ROWS, COLUMNS = 3, 3
BOARD_SHAPE = (ROWS, COLUMNS)
MAX_COLUMN_SCORE = ROWS ** 2 * DICE
MAX_BOARD_SCORE = COLUMNS * MAX_COLUMN_SCORE
