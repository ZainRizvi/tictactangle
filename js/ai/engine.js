// Pure-JS alpha-beta engine implementing the PlayerController port.
// Serves as the fallback opponent when the WASM model is unavailable.

import { legalMoves, applyMove, LINES, idx, X, O, EMPTY, SIZE } from '../domain/rules.js';

const WIN = 10000;
const TIME_BUDGET_MS = 650;

function evaluate(s, me) {
  const opp = me === X ? O : X;
  const { r: cr, c: cc } = s.center;
  let score = 0;

  for (const line of LINES) {
    let mine = 0, theirs = 0;
    for (const [dr, dc] of line) {
      const v = s.board[idx(cr + dr, cc + dc)];
      if (v === me) mine++;
      else if (v === opp) theirs++;
    }
    if (theirs === 0) score += mine === 2 ? 25 : mine === 1 ? 3 : 0;
    if (mine === 0) score -= theirs === 2 ? 25 : theirs === 1 ? 3 : 0;
  }

  // The grid's center cell touches 4 lines; owning it is strong.
  const centerVal = s.board[idx(cr, cc)];
  if (centerVal === me) score += 6;
  else if (centerVal === opp) score -= 6;

  // Pieces stranded in the dark are close to dead weight.
  for (let i = 0; i < SIZE * SIZE; i++) {
    const v = s.board[i];
    if (v === EMPTY) continue;
    const r = (i / SIZE) | 0;
    const c = i % SIZE;
    const lit = Math.abs(r - cr) <= 1 && Math.abs(c - cc) <= 1;
    if (!lit) score += v === me ? -2 : 2;
  }

  return score + Math.random() * 0.8 - 0.4;
}

function orderMoves(moves) {
  const rank = { place: 0, move: 1, grid: 2 };
  return moves
    .map((m) => ({ m, k: rank[m.type] + Math.random() * 0.9 }))
    .sort((a, b) => a.k - b.k)
    .map((e) => e.m);
}

function search(s, depth, alpha, beta, me, budget) {
  if (s.result) {
    if (s.result.type === 'tie') return 0;
    // Prefer faster wins / slower losses.
    return s.result.winner === me ? WIN + depth : -(WIN + depth);
  }
  if (depth === 0 || budget.nodes-- <= 0) return evaluate(s, me);

  const moves = orderMoves(legalMoves(s));
  const maxing = s.turn === me;
  let best = maxing ? -Infinity : Infinity;

  for (const m of moves) {
    const v = search(applyMove(s, m), depth - 1, alpha, beta, me, budget);
    if (maxing) {
      if (v > best) best = v;
      if (best > alpha) alpha = best;
    } else {
      if (v < best) best = v;
      if (best < beta) beta = best;
    }
    if (alpha >= beta) break;
  }
  return best;
}

function searchRoot(s, deadline) {
  const me = s.turn;
  const moves = orderMoves(legalMoves(s));
  if (moves.length === 0) return null;

  let bestMove = moves[0];
  for (let depth = 2; depth <= 5; depth++) {
    let roundBest = -Infinity;
    let roundMove = null;
    let alpha = -Infinity;
    const budget = { nodes: 400000 };
    let aborted = false;

    for (const m of moves) {
      const v = search(applyMove(s, m), depth - 1, alpha, Infinity, me, budget);
      if (v > roundBest) {
        roundBest = v;
        roundMove = m;
      }
      if (v > alpha) alpha = v;
      if (performance.now() > deadline || budget.nodes <= 0) {
        aborted = true;
        break;
      }
    }

    if (!aborted && roundMove) {
      bestMove = roundMove;
      if (roundBest >= WIN) break; // found a forced win
    } else {
      // Partial round: trust it only if it found a win outright.
      if (roundMove && roundBest >= WIN) bestMove = roundMove;
      break;
    }
  }
  return bestMove;
}

/** @returns {import('../app/ports.js').PlayerController} */
export function createJsEngine() {
  return {
    async chooseMove(state) {
      await new Promise((r) => setTimeout(r, 30)); // let the UI paint first
      return searchRoot(state, performance.now() + TIME_BUDGET_MS);
    },
  };
}
