// DOM view — one possible UI over a GameSession. Holds only presentation
// state (selection, chosen mode/side); all game state lives in the session.
// Swapping this file for a canvas/terminal/network view requires no changes
// to js/domain or js/app.

import * as rules from '../domain/rules.js';

const COLORS = { [rules.X]: '#ff5d4d', [rules.O]: '#33e0c0' };

function svgFor(p) {
  if (p === rules.X) {
    return `<svg viewBox="0 0 100 100" aria-hidden="true"><path d="M22 22 L78 78 M78 22 L22 78" stroke="${COLORS[rules.X]}" stroke-width="15" stroke-linecap="round" fill="none"/></svg>`;
  }
  return `<svg viewBox="0 0 100 100" aria-hidden="true"><circle cx="50" cy="50" r="31" stroke="${COLORS[rules.O]}" stroke-width="14" fill="none"/></svg>`;
}

/**
 * @param {{ session: import('../app/session.js').GameSession,
 *           configureSeats: (mode: 'pvp'|'ai', humanSide: number) => void }} deps
 */
export function mountDomView({ session, configureSeats }) {
  const $ = (id) => document.getElementById(id);
  const boardEl = $('board');
  const spotlightEl = $('spotlight');
  const endgameEl = $('endgame');
  const endTitleEl = $('endTitle');
  const endSubEl = $('endSub');
  const hintEl = $('hint');
  const turnGlyphEl = $('turnGlyph');
  const turnLabelEl = $('turnLabel');
  const phaseLabelEl = $('phaseLabel');
  const chipsXEl = $('chipsX');
  const chipsOEl = $('chipsO');
  const rulesDialog = $('rulesDialog');

  // presentation state
  let snap = session.snapshot();
  let selected = null; // board index of a selected own piece
  let mode = 'pvp';
  let humanSide = rules.X;

  // ---------- board construction ----------

  const cellEls = [];
  for (let i = 0; i < rules.SIZE * rules.SIZE; i++) {
    const [r, c] = rules.rc(i);
    const btn = document.createElement('button');
    btn.className = 'cell';
    btn.dataset.i = i;
    btn.setAttribute('aria-label', `row ${r + 1}, column ${c + 1}`);
    btn.addEventListener('click', () => onCell(i));
    boardEl.appendChild(btn);
    cellEls.push(btn);
  }

  const arrowEls = [...spotlightEl.querySelectorAll('.g-arrow')];
  for (const a of arrowEls) {
    a.addEventListener('click', () => session.slideGrid(+a.dataset.dr, +a.dataset.dc));
  }

  // ---------- interaction ----------

  const interactive = () => session.canAct();

  function hasEmptyGridCell() {
    return rules.gridCells(snap.state).some((i) => snap.state.board[i] === rules.EMPTY);
  }

  function pieceMoveLegal(from, to) {
    const s = snap.state;
    const [tr, tc] = rules.rc(to);
    return (
      rules.actionsUnlocked(s) &&
      s.board[from] === s.turn &&
      s.board[to] === rules.EMPTY &&
      rules.inGrid(s, tr, tc) &&
      from !== to
    );
  }

  function onCell(i) {
    if (!interactive()) return;
    const s = snap.state;
    const [r, c] = rules.rc(i);
    const occupant = s.board[i];
    const unlocked = rules.actionsUnlocked(s);

    if (selected !== null) {
      if (i === selected) { selected = null; render(); return; }
      if (pieceMoveLegal(selected, i)) { session.movePiece(selected, i); return; }
      if (occupant === s.turn && unlocked) { selected = i; render(); return; }
      nudge(i);
      return;
    }

    if (occupant === s.turn && unlocked && hasEmptyGridCell()) {
      selected = i;
      render();
      return;
    }

    if (occupant === rules.EMPTY && rules.inGrid(s, r, c) && s.reserve[s.turn] > 0) {
      session.place(i);
      return;
    }

    nudge(i);
  }

  function nudge(i) {
    const el = cellEls[i];
    el.classList.remove('nudge');
    void el.offsetWidth; // restart animation
    el.classList.add('nudge');
  }

  // ---------- rendering ----------

  function render() {
    const s = snap.state;
    const unlocked = rules.actionsUnlocked(s);
    const canPlace = interactive() && selected === null && s.reserve[s.turn] > 0;

    boardEl.dataset.turn = s.turn === rules.X ? 'x' : 'o';
    boardEl.dataset.canPlace = canPlace ? '1' : '0';

    spotlightEl.style.setProperty('--gr', s.center.r - 1);
    spotlightEl.style.setProperty('--gc', s.center.c - 1);

    for (const a of arrowEls) {
      const ok = interactive() && unlocked && rules.gridMoveValid(s, +a.dataset.dr, +a.dataset.dc);
      a.hidden = !ok;
    }

    const winCells = new Map(); // idx -> 'win-x' | 'win-o'
    if (s.result?.type === 'win') {
      const cls = s.result.winner === rules.X ? 'win-x' : 'win-o';
      for (const i of s.result.line) winCells.set(i, cls);
    } else if (s.result?.type === 'tie') {
      for (const i of s.result.xLine) winCells.set(i, 'win-x');
      for (const i of s.result.oLine) winCells.set(i, 'win-o');
    }

    for (let i = 0; i < cellEls.length; i++) {
      const el = cellEls[i];
      const [r, c] = rules.rc(i);
      const lit = rules.inGrid(s, r, c);
      const occupant = s.board[i];
      const isTarget = selected !== null && occupant === rules.EMPTY && lit && interactive();
      const isOpen = occupant === rules.EMPTY && lit && canPlace;
      const isOwn = interactive() && unlocked && occupant === s.turn && hasEmptyGridCell();

      el.className = 'cell';
      if (lit) el.classList.add('lit');
      if (isOpen) el.classList.add('open');
      if (isOwn) el.classList.add('own');
      if (selected === i) el.classList.add('selected');
      if (isTarget) el.classList.add('target');
      const win = winCells.get(i);
      if (win) el.classList.add(win);

      let html = '';
      if (occupant !== rules.EMPTY) {
        const cls = ['piece', occupant === rules.X ? 'px' : 'po'];
        if (!lit) cls.push('dark');
        if (snap.lastMove?.type === 'place' && snap.lastMove.to === i) cls.push('pop');
        html = `<span class="${cls.join(' ')}">${svgFor(occupant)}</span>`;
      } else if (isTarget || isOpen) {
        html = `<span class="piece ghost">${svgFor(s.turn)}</span>`;
      }
      el.innerHTML = html;
    }

    animateLastMove();
    renderHud(unlocked);
    renderEndgame();
  }

  let animatedPly = 0;
  function animateLastMove() {
    if (snap.lastMove?.type !== 'move' || snap.state.ply === animatedPly) return;
    animatedPly = snap.state.ply;
    const { from, to } = snap.lastMove;
    const fromRect = cellEls[from].getBoundingClientRect();
    const toRect = cellEls[to].getBoundingClientRect();
    const pieceEl = cellEls[to].querySelector('.piece');
    if (pieceEl) {
      pieceEl.animate(
        [
          { transform: `translate(${fromRect.left - toRect.left}px, ${fromRect.top - toRect.top}px)` },
          { transform: 'none' },
        ],
        { duration: 340, easing: 'cubic-bezier(0.2, 0.8, 0.2, 1)' }
      );
    }
  }

  function renderHud(unlocked) {
    const s = snap.state;
    turnGlyphEl.innerHTML = svgFor(s.turn);

    const name = s.turn === rules.X ? 'X' : 'O';
    const aiSeated = snap.seats[rules.X] || snap.seats[rules.O];
    if (s.result) {
      turnLabelEl.textContent = 'game over';
    } else if (aiSeated) {
      turnLabelEl.textContent = snap.seats[s.turn] ? `${name} to move — AI` : `${name} to move — you`;
    } else {
      turnLabelEl.textContent = `${name} to move`;
    }

    phaseLabelEl.textContent = s.result
      ? 'the light settles'
      : unlocked
        ? 'open play'
        : 'placement phase';

    renderChips(chipsXEl, rules.X);
    renderChips(chipsOEl, rules.O);

    if (s.result) {
      hintEl.textContent = '';
    } else if (snap.busySeat !== null) {
      hintEl.textContent = 'the machine is thinking…';
    } else if (!interactive()) {
      hintEl.textContent = '';
    } else if (selected !== null) {
      hintEl.textContent = 'tap an empty lit cell to move there — tap the piece again to cancel';
    } else if (!unlocked) {
      hintEl.textContent = 'placement phase — drop a piece on any lit cell';
    } else {
      const opts = [];
      if (s.reserve[s.turn] > 0) opts.push('place a piece');
      opts.push('move a piece', 'slide the grid');
      hintEl.textContent = opts.join(' · ');
    }
  }

  function renderChips(container, player) {
    let html = '';
    for (let k = 0; k < rules.PIECES; k++) {
      const spent = k >= snap.state.reserve[player];
      html += `<span class="chip${spent ? ' spent' : ''}">${svgFor(player)}</span>`;
    }
    container.innerHTML = html;
  }

  function renderEndgame() {
    const s = snap.state;
    if (!s.result) {
      endgameEl.hidden = true;
      return;
    }
    endTitleEl.className = 'endgame-title';
    if (s.result.type === 'tie') {
      endTitleEl.classList.add('tie');
      endTitleEl.textContent = 'DEAD HEAT';
      endSubEl.textContent = 'one slide lit up both lines';
    } else {
      const winner = s.result.winner;
      endTitleEl.classList.add(winner === rules.X ? 'win-x' : 'win-o');
      endTitleEl.textContent = winner === rules.X ? 'X WINS' : 'O WINS';
      const aiSeated = snap.seats[rules.X] || snap.seats[rules.O];
      if (aiSeated) {
        endSubEl.textContent = snap.seats[winner] ? 'outplayed by the machine' : 'the grid bends to your will';
      } else {
        endSubEl.textContent = 'three in the light';
      }
    }
    endgameEl.hidden = false;
  }

  // ---------- controls ----------

  $('newGameBtn').addEventListener('click', () => session.newGame());
  $('againBtn').addEventListener('click', () => session.newGame());

  const modeBtns = [$('modePvp'), $('modeAi')];
  for (const btn of modeBtns) {
    btn.addEventListener('click', () => {
      if (mode === btn.dataset.mode) return;
      mode = btn.dataset.mode;
      for (const b of modeBtns) b.classList.toggle('is-active', b === btn);
      $('sidePicker').hidden = mode !== 'ai';
      configureSeats(mode, humanSide);
    });
  }

  const sideBtns = [$('sideX'), $('sideO')];
  $('sideX').dataset.side = rules.X;
  $('sideO').dataset.side = rules.O;
  for (const btn of sideBtns) {
    btn.addEventListener('click', () => {
      const side = +btn.dataset.side;
      if (humanSide === side) return;
      humanSide = side;
      for (const b of sideBtns) b.classList.toggle('is-active', b === btn);
      if (mode === 'ai') configureSeats(mode, humanSide);
    });
  }

  $('rulesBtn').addEventListener('click', () => rulesDialog.showModal());
  $('rulesCloseBtn').addEventListener('click', () => rulesDialog.close());
  rulesDialog.addEventListener('click', (e) => {
    if (e.target === rulesDialog) rulesDialog.close();
  });

  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && selected !== null && !rulesDialog.open) {
      selected = null;
      render();
    }
  });

  // ---------- session subscription ----------

  let lastState = snap.state;
  session.subscribe((s) => {
    snap = s;
    if (s.state !== lastState) {
      lastState = s.state;
      selected = null; // any state change invalidates the selection
    }
    render();
  });
}
