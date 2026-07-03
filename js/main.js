// Composition root — the only place where layers are wired together.

import { X, O, other } from './domain/rules.js';
import { GameSession } from './app/session.js';
import { createAiPlayer, withMinDelay } from './ai/index.js';
import { mountDomView } from './ui/view.js';

const session = new GameSession();

// Spectated AI-vs-AI games are adjudicated as draws at this many plies so a
// series can continue; human games are never cut off.
const SPECTATE_PLY_CAP = 100;
// Pacing floors: replies against a human just need to read as moves; in
// spectator mode both sides slow down so the game is watchable.
const HUMAN_PACE_MS = 550;
const SPECTATE_PACE_MS = 900;

// One raw engine per difficulty, created lazily and cached; pacing wrappers
// are cheap and applied per configuration.
const rawEngines = new Map();
function getEngine(difficulty) {
  if (!rawEngines.has(difficulty)) {
    rawEngines.set(difficulty, createAiPlayer(difficulty));
  }
  return rawEngines.get(difficulty);
}

// Generation counter guards against a slow AI load landing after the user
// has already switched mode, side, or difficulty again.
let seatGeneration = 0;

/**
 * @param {{ mode: 'pvp'|'ai'|'aivai', humanSide?: number,
 *           difficulty?: string, diffX?: string, diffO?: string }} config
 */
async function configureSeats(config) {
  const gen = ++seatGeneration;
  if (config.mode === 'pvp') {
    session.plyCap = null;
    session.setSeats({ [X]: null, [O]: null });
    return;
  }
  if (config.mode === 'ai') {
    const ai = withMinDelay(await getEngine(config.difficulty ?? 'medium'), HUMAN_PACE_MS);
    if (gen !== seatGeneration) return;
    const humanSide = config.humanSide ?? X;
    session.plyCap = null;
    session.setSeats({ [humanSide]: null, [other(humanSide)]: ai });
    return;
  }
  // aivai: both seats are engines, possibly different difficulties.
  const [engineX, engineO] = await Promise.all([
    getEngine(config.diffX ?? 'medium'),
    getEngine(config.diffO ?? 'medium'),
  ]);
  if (gen !== seatGeneration) return;
  session.plyCap = SPECTATE_PLY_CAP;
  session.setSeats({
    [X]: withMinDelay(engineX, SPECTATE_PACE_MS),
    [O]: withMinDelay(engineO, SPECTATE_PACE_MS),
  });
}

mountDomView({ session, configureSeats });
