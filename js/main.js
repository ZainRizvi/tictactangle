// Composition root — the only place where layers are wired together.

import { X, O, other } from './domain/rules.js';
import { GameSession } from './app/session.js';
import { createAiPlayer, withMinDelay } from './ai/index.js';
import { mountDomView } from './ui/view.js';

const session = new GameSession();
const aiPlayer = createAiPlayer().then((ai) => withMinDelay(ai, 550));

async function configureSeats(mode, humanSide) {
  if (mode === 'pvp') {
    session.setSeats({ [X]: null, [O]: null });
  } else {
    const ai = await aiPlayer;
    session.setSeats({ [humanSide]: null, [other(humanSide)]: ai });
  }
}

mountDomView({ session, configureSeats });
