// Tic Tac Two — pure game logic (no DOM).
// Rules: 5x5 board, movable 3x3 grid. Each player has 4 pieces.
// A player's first two turns must be placements. After both players have
// placed two pieces, a turn is one of: place a piece in the grid, slide the
// grid one step in any of 8 directions, or move one of your pieces (from
// anywhere on the board) to an empty cell inside the grid.
// First player with 3-in-a-row of their pieces inside the grid wins; a grid
// slide that produces a line for both players at once is a tie.
// No take-backs: immediately after a grid slide, the responding player may
// not slide the grid straight back to the position it just left.

export const EMPTY = 0;
export const X = 1;
export const O = 2;

export const SIZE = 5; // board is SIZE x SIZE
export const PIECES = 4; // pieces per player
export const MIN_PLACED = 2; // placements required before grid/piece moves unlock

// Grid centers must keep the 3x3 grid on the board.
export const CENTER_MIN = 1;
export const CENTER_MAX = 3;

export const idx = (r, c) => r * SIZE + c;
export const rc = (i) => [Math.floor(i / SIZE), i % SIZE];

export const other = (p) => (p === X ? O : X);

export function newGame() {
  return {
    board: new Array(SIZE * SIZE).fill(EMPTY),
    center: { r: 2, c: 2 },
    turn: X,
    reserve: { [X]: PIECES, [O]: PIECES },
    placed: { [X]: 0, [O]: 0 },
    ply: 0,
    bannedCenter: null, // {r,c} a grid slide may not land on this turn
    result: null, // {type:'win', winner, line} | {type:'tie', xLine, oLine}
  };
}

export function cloneState(s) {
  return {
    board: s.board.slice(),
    center: { r: s.center.r, c: s.center.c },
    turn: s.turn,
    reserve: { [X]: s.reserve[X], [O]: s.reserve[O] },
    placed: { [X]: s.placed[X], [O]: s.placed[O] },
    ply: s.ply,
    bannedCenter: s.bannedCenter ? { r: s.bannedCenter.r, c: s.bannedCenter.c } : null,
    result: s.result,
  };
}

export function inGrid(s, r, c) {
  return Math.abs(r - s.center.r) <= 1 && Math.abs(c - s.center.c) <= 1;
}

export function gridCells(s) {
  const cells = [];
  for (let r = s.center.r - 1; r <= s.center.r + 1; r++) {
    for (let c = s.center.c - 1; c <= s.center.c + 1; c++) {
      cells.push(idx(r, c));
    }
  }
  return cells;
}

export function actionsUnlocked(s) {
  return s.placed[X] >= MIN_PLACED && s.placed[O] >= MIN_PLACED;
}

export const DIRS = [
  { dr: -1, dc: -1 }, { dr: -1, dc: 0 }, { dr: -1, dc: 1 },
  { dr: 0, dc: -1 },                     { dr: 0, dc: 1 },
  { dr: 1, dc: -1 },  { dr: 1, dc: 0 },  { dr: 1, dc: 1 },
];

export function gridMoveValid(s, dr, dc) {
  const nr = s.center.r + dr;
  const nc = s.center.c + dc;
  if (nr < CENTER_MIN || nr > CENTER_MAX || nc < CENTER_MIN || nc > CENTER_MAX) return false;
  // No take-backs: can't slide the grid straight back where it just was.
  if (s.bannedCenter && nr === s.bannedCenter.r && nc === s.bannedCenter.c) return false;
  return true;
}

export function legalMoves(s) {
  if (s.result) return [];
  const moves = [];
  const empties = gridCells(s).filter((i) => s.board[i] === EMPTY);

  if (s.reserve[s.turn] > 0) {
    for (const to of empties) moves.push({ type: 'place', to });
  }

  if (actionsUnlocked(s)) {
    for (const { dr, dc } of DIRS) {
      if (gridMoveValid(s, dr, dc)) moves.push({ type: 'grid', dr, dc });
    }
    for (let from = 0; from < SIZE * SIZE; from++) {
      if (s.board[from] !== s.turn) continue;
      for (const to of empties) moves.push({ type: 'move', from, to });
    }
  }
  return moves;
}

// The 8 winning lines of the 3x3 grid, as offsets from the grid center.
export const LINES = [
  [[-1, -1], [-1, 0], [-1, 1]],
  [[0, -1], [0, 0], [0, 1]],
  [[1, -1], [1, 0], [1, 1]],
  [[-1, -1], [0, -1], [1, -1]],
  [[-1, 0], [0, 0], [1, 0]],
  [[-1, 1], [0, 1], [1, 1]],
  [[-1, -1], [0, 0], [1, 1]],
  [[-1, 1], [0, 0], [1, -1]],
];

export function findLine(s, player) {
  const { r: cr, c: cc } = s.center;
  for (const line of LINES) {
    const cells = line.map(([dr, dc]) => idx(cr + dr, cc + dc));
    if (cells.every((i) => s.board[i] === player)) return cells;
  }
  return null;
}

// Applies a move (assumed legal) and returns the new state.
export function applyMove(s, m) {
  const n = cloneState(s);
  if (m.type === 'place') {
    n.board[m.to] = n.turn;
    n.reserve[n.turn]--;
    n.placed[n.turn]++;
  } else if (m.type === 'grid') {
    n.center = { r: n.center.r + m.dr, c: n.center.c + m.dc };
  } else if (m.type === 'move') {
    n.board[m.from] = EMPTY;
    n.board[m.to] = n.turn;
  }
  // A slide arms the take-back ban for the reply; any other move clears it.
  n.bannedCenter = m.type === 'grid' ? { r: s.center.r, c: s.center.c } : null;
  n.ply++;

  // A tie can only arise from a grid slide (a placement or piece move never
  // completes an opponent line), but checking both uniformly costs nothing.
  const xLine = findLine(n, X);
  const oLine = findLine(n, O);
  if (xLine && oLine) {
    n.result = { type: 'tie', xLine, oLine };
  } else if (xLine) {
    n.result = { type: 'win', winner: X, line: xLine };
  } else if (oLine) {
    n.result = { type: 'win', winner: O, line: oLine };
  }

  if (!n.result) n.turn = other(n.turn);
  return n;
}

export function isLegal(s, m) {
  if (!m) return false;
  // Compare only the fields relevant to each move type, so moves that have
  // been serialized (JSON turns absent fields into null) still validate.
  return legalMoves(s).some((lm) => {
    if (lm.type !== m.type) return false;
    if (m.type === 'place') return lm.to === m.to;
    if (m.type === 'move') return lm.from === m.from && lm.to === m.to;
    return lm.dr === m.dr && lm.dc === m.dc;
  });
}
