export type Point = {
  x: number;
  y: number;
};

export type Stroke = { points: Point[]; width: number };

type State = {
  history: Stroke[];
  redo: Stroke[];
};

type Action =
  | { type: "commit"; stroke: Stroke; maxHistory: number }
  | { type: "undo" }
  | { type: "redo" }
  | { type: "clear" };

export function drawReducer(state: State, action: Action): State {
  if (action.type === "commit") {
    const nextHistory =
      state.history.length >= action.maxHistory
        ? state.history.slice(state.history.length - action.maxHistory + 1)
        : state.history;

    return { history: [...nextHistory, action.stroke], redo: [] };
  }

  if (action.type === "undo") {
    if (state.history.length === 0) return state;
    const stroke = state.history[state.history.length - 1];
    return {
      history: state.history.slice(0, -1),
      redo: [stroke, ...state.redo],
    };
  }

  if (action.type === "redo") {
    if (state.redo.length === 0) return state;
    const stroke = state.redo[0];
    return { history: [...state.history, stroke], redo: state.redo.slice(1) };
  }

  return { history: [], redo: [] };
}
