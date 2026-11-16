// Message shapes shared with server.py

export type Cell = [number, number]; // [row, col]
export type Result = "win" | "trap" | "timeout";
export type Source = "trained" | "random" | "human" | "preview";

export const ACTIONS = ["up", "down", "left", "right"] as const;
export type ActionName = (typeof ACTIONS)[number];

export interface Board {
  size: number;
  agent: Cell;
  treasure: Cell;
  traps: Cell[];
}

export interface Frame {
  type: "frame";
  source: Source;
  episode: number;
  step: number;
  board: Board;
  obs: number[];
  probs: number[] | null; // chance of each next move: up, down, left, right
  value: number | null; // critic's estimate of future reward
  action: number | null; // move that led to this frame
  reward: number | null;
  total: number;
  done: boolean;
  result: Result | null;
  max_steps: number;
}

export interface TrainPoint {
  update: number;
  steps: number;
  avg_reward: number;
  win: number;
  trap: number;
  entropy: number;
  kl: number;
  clip_frac: number;
  elapsed: number;
}

export interface EvalResult {
  win: number;
  trap: number;
  timeout: number;
}

export interface TrainState {
  running: boolean;
  update: number;
  updates: number;
}

export type ServerMessage =
  | {
      type: "hello";
      model_available: boolean;
      training: TrainState;
      history: TrainPoint[];
      before: EvalResult | null;
      after: EvalResult | null;
      config: { rollout_steps: number };
    }
  | Frame
  | { type: "train_start"; updates: number; before: EvalResult }
  | { type: "train_progress"; point: TrainPoint; updates: number }
  | { type: "train_preview"; update: number; frames: Frame[] }
  | {
      type: "train_done";
      stopped: boolean;
      updates_done: number;
      before: EvalResult | null;
      after: EvalResult | null;
      model_available: boolean;
    }
  | { type: "error"; message: string; code?: string };

export type ClientMessage =
  | { type: "watch"; agent: "trained" | "random" }
  | { type: "stop" }
  | { type: "speed"; delay_ms: number }
  | { type: "human_reset" }
  | { type: "human_move"; action: number }
  | { type: "train"; updates: number }
  | { type: "stop_train" };
