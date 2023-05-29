import chess
import chess.engine
import random
import numpy as np
from typing import Generator, Tuple
from tqdm import tqdm


SIZE = 100_000

PATH_TO_STOCKFISH = './stockfish_15.1_win_x64_avx2/stockfish-windows-2022-x86-64-avx2.exe'

STOCKFISH_DEPTH = 10

MAXIMUM_DEPTH = 100

VERBOSE = True


def generate_random_positions(count: int, maximum_depth: int) -> Generator[str, None, None]:
    board = chess.Board()
    for _ in range(count):
        depth = random.randint(0, maximum_depth)
        for _ in range(depth):
            all_moves = tuple(board.legal_moves)
            board.push(random.choice(all_moves))
            if board.is_game_over():
                break
        yield board.fen()
        board.reset()


PIECES = {
    'P': 1,
    'N': 2,
    'B': 3,
    'R': 4,
    'Q': 5,
    'K': 6,
    'p': -1,
    'n': -2,
    'b': -3,
    'r': -4,
    'q': -5,
    'k': -6
}

PIECES_INVERTED = {value: key for key, value in PIECES.items()}


def fen_to_array(fen: str) -> np.ndarray:
    pieces, color, castling, en_passant, _, _ = fen.split()
    rows = pieces.split('/')
    array = []
    for row in rows:
        for character in row:
            if character.isdigit():
                array.extend([0]*int(character))
            else:
                array.append(PIECES[character])
    array.append(0 if color == 'w' else 1)
    for x in ('K', 'Q', 'k', 'q'):
        array.append(1 if x in castling else 0)
    try:
        array.append(chess.parse_square(en_passant))
    except ValueError:
        array.append(-1)
    return np.array(array, dtype=np.short)


def array_to_fen(array: np.ndarray) -> str:
    pieces = array[:64]
    color, white_kingside, white_queenside, black_kingside, black_queenside, en_passant = array[64:]
    new_rows = []
    for row in np.reshape(pieces, (8, 8)):
        new_row = []
        for square in row:
            if square == 0:
                if new_row:
                    if new_row[-1].isdigit():
                        new_row[-1] = str(int(new_row[-1]) + 1)
                        continue
                new_row.append('1')
            else:
                new_row.append(PIECES_INVERTED[square])
        new_rows.append(''.join(new_row))
    fen = '/'.join(new_rows) + f' {"b" if color else "w"} '
    if any([white_kingside, white_queenside, black_kingside, black_queenside]):
        if white_kingside:
            fen += 'K'
        if white_queenside:
            fen += 'Q'
        if black_kingside:
            fen += 'k'
        if black_queenside:
            fen += 'q'
        fen += ' '
    else:
        fen += '- '
    fen += chess.square_name(en_passant) if en_passant != -1 else '-'

    return fen + ' 0 0'


def category(score: int):
    if score <= -100:
        return 0
    if score <= 100:
        return 1
    else:
        return 2


def create(size: int, maximum_depth: int, stockfish_depth: int, gui=False) -> Tuple[np.ndarray, np.ndarray]:
    limit = chess.engine.Limit(depth=stockfish_depth)
    board = chess.Board()
    x = None
    y = []
    with chess.engine.SimpleEngine.popen_uci(PATH_TO_STOCKFISH) as stockfish:
        positions = generate_random_positions(size, maximum_depth)
        for position in tqdm(positions, total=size, disable=not gui):
            board.set_fen(position)
            score = stockfish.analyse(board, limit)['score']
            cp = score.white().score(mate_score=10000)
            if x is None:
                x = fen_to_array(position)
            else:
                x = np.vstack((x, fen_to_array(position)))
            y.append(category(cp))
    return x, np.array(y)


if __name__ == '__main__':
    x_train, y_train = create(SIZE, MAXIMUM_DEPTH, STOCKFISH_DEPTH, gui=VERBOSE)
    x_train.tofile('x.csv', sep=',')
    y_train.tofile('y.csv', sep=',')
