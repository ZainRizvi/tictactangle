// GameSession — application layer. Owns the authoritative game state and
// turn orchestration. Knows nothing about the DOM; UIs subscribe and issue
// intents. Automated players (AI, remote peers) are injected per seat as
// PlayerController objects (see ports.js).

import * as rules from '../domain/rules.js';

export class GameSession {
  /**
   * @param {{[seat: number]: import('./ports.js').PlayerController|null}} [seats]
   *   Controller per seat (rules.X / rules.O); null means local human.
   */
  constructor(seats = { [rules.X]: null, [rules.O]: null }) {
    this.seats = seats;
    this.state = rules.newGame();
    this.lastMove = null;
    this.busySeat = null; // seat whose controller is currently thinking
    this.seatFault = null; // seat whose controller failed or stalled
    this.plyCap = null; // adjudicate a draw at this many plies (null = never)
    this.adjudicatedDraw = false; // game stopped by the ply cap
    this._listeners = new Set();
    this._token = 0;
  }

  /** Subscribe to snapshots; immediately called with the current one. */
  subscribe(fn) {
    this._listeners.add(fn);
    fn(this.snapshot());
    return () => this._listeners.delete(fn);
  }

  /**
   * Snapshots share the underlying state object by reference for cheap
   * change detection (a new state object per move). Subscribers must treat
   * them as read-only; all mutation goes through intents.
   */
  snapshot() {
    return {
      state: this.state,
      lastMove: this.lastMove,
      busySeat: this.busySeat,
      seatFault: this.seatFault,
      adjudicatedDraw: this.adjudicatedDraw,
      plyCap: this.plyCap,
      seats: this.seats,
    };
  }

  /** True when the game has ended for any reason (win, tie, or ply cap). */
  isOver() {
    return this.state.result !== null || this.adjudicatedDraw;
  }

  /** True when the seat to move is a local human free to act. */
  canAct() {
    return !this.isOver() && this.busySeat === null && this.seats[this.state.turn] === null;
  }

  /** Replace seat controllers and start a fresh game. */
  setSeats(seats) {
    this.seats = seats;
    this.newGame();
  }

  newGame() {
    this._token++;
    this.state = rules.newGame();
    this.lastMove = null;
    this.busySeat = null;
    this.seatFault = null;
    this.adjudicatedDraw = false;
    this._emit();
    this._pump();
  }

  // ---- local-human intents ----

  place(to) {
    return this._humanMove({ type: 'place', to });
  }

  movePiece(from, to) {
    return this._humanMove({ type: 'move', from, to });
  }

  slideGrid(dr, dc) {
    return this._humanMove({ type: 'grid', dr, dc });
  }

  _humanMove(move) {
    if (!this.canAct() || !rules.isLegal(this.state, move)) return false;
    this._apply(move);
    return true;
  }

  // ---- internals ----

  _apply(move) {
    this.state = rules.applyMove(this.state, move);
    this.lastMove = move;
    // Spectated games could shuffle forever under official rules; the ply
    // cap adjudicates a dead heat so a series can continue.
    if (!this.state.result && this.plyCap !== null && this.state.ply >= this.plyCap) {
      this.adjudicatedDraw = true;
    }
    this._emit();
    this._pump();
  }

  /** If the seat to move has a controller, ask it for a move. */
  async _pump() {
    const controller = this.seats[this.state.turn];
    if (!controller || this.isOver()) return;

    this.busySeat = this.state.turn;
    const token = ++this._token;
    this._emit();

    let move = null;
    try {
      move = await controller.chooseMove(rules.cloneState(this.state));
    } catch (err) {
      console.error('seat controller failed', err);
    }
    if (token !== this._token) return; // game was reset/reconfigured meanwhile

    this.busySeat = null;
    if (move && rules.isLegal(this.state, move)) {
      this._apply(move);
    } else {
      if (move) console.error('seat controller returned illegal move', move);
      // The game cannot continue without this seat's move; surface the fault
      // so UIs can tell the player instead of freezing silently.
      this.seatFault = this.state.turn;
      this._emit();
    }
  }

  _emit() {
    const snap = this.snapshot();
    for (const fn of this._listeners) fn(snap);
  }
}
