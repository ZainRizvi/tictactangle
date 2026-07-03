// Composition root — the only place where layers are wired together.

import { X, O, other } from './domain/rules.js';
import { GameSession } from './app/session.js';
import { createAiPlayer, withMinDelay } from './ai/index.js';
import { mountDomView } from './ui/view.js';

const session = new GameSession();

// One paced player per difficulty, created lazily and cached.
const aiPlayers = new Map();
function getAiPlayer(difficulty) {
  if (!aiPlayers.has(difficulty)) {
    aiPlayers.set(
      difficulty,
      createAiPlayer(difficulty).then((ai) => withMinDelay(ai, 550))
    );
  }
  return aiPlayers.get(difficulty);
}

// Generation counter guards against a slow AI load landing after the user
// has already switched mode, side, or difficulty again.
let seatGeneration = 0;

async function configureSeats(mode, humanSide, difficulty = 'medium') {
  const gen = ++seatGeneration;
  if (mode === 'pvp') {
    session.setSeats({ [X]: null, [O]: null });
    return;
  }
  const ai = await getAiPlayer(difficulty);
  if (gen !== seatGeneration) return;
  session.setSeats({ [humanSide]: null, [other(humanSide)]: ai });
}

mountDomView({ session, configureSeats });
