// Contracts between layers. The domain and app layers know nothing about who
// implements these — an AI engine, a network peer, or a scripted replay all
// plug in the same way.

/**
 * A seat controller decides moves for one side. A seat with no controller
 * (null) is a local human: its moves arrive through GameSession intents
 * (place / movePiece / slideGrid) issued by whatever UI is mounted.
 *
 * @typedef {Object} PlayerController
 * @property {(state: object) => Promise<object|null>} chooseMove
 *   Given a read-only game state, resolves with a legal move
 *   ({type:'place',to} | {type:'move',from,to} | {type:'grid',dr,dc})
 *   or null if it has none. Must not mutate the state.
 */

/**
 * A view/UI is anything that subscribes to a GameSession and renders its
 * snapshots. Swapping the UI (DOM, canvas, terminal, remote) must never
 * require touching js/domain or js/app.
 *
 * @typedef {(snapshot: object) => void} SessionListener
 */

export {};
