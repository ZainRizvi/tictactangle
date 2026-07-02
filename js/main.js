// Composition root — the only place where layers are wired together.

import { X, O, other } from './domain/rules.js';
import { GameSession } from './app/session.js';
import { createAiPlayer, withMinDelay } from './ai/index.js';
import { mountDomView } from './ui/view.js';

const session = new GameSession();
const aiPlayer = createAiPlayer().then((ai) => withMinDelay(ai, 550));

// Generation counter guards against a slow AI load landing after the user
// has already switched back to PvP.
let seatGeneration = 0;

async function configureSeats(mode, humanSide) {
  const gen = ++seatGeneration;
  if (mode === 'pvp') {
    session.setSeats({ [X]: null, [O]: null });
    return;
  }
  const ai = await aiPlayer;
  if (gen !== seatGeneration) return;
  session.setSeats({ [humanSide]: null, [other(humanSide)]: ai });
}

mountDomView({ session, configureSeats });
