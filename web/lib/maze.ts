"use client";

import { useCallback, useEffect, useReducer, useRef, useState } from "react";
import type { Cell } from "./protocol";

// ---------------------------------------------------------------------- //
// Messages from maze_session.py
// ---------------------------------------------------------------------- //
export interface MazeInfo {
  size: number;
  seed: number;
  walls: Cell[];
  start: Cell;
  goal: Cell;
  shortest: number | null;
  max_steps: number;
}

export interface Settings {
  alpha: number;
  gamma: number;
  epsilon: number;
  decay: boolean;
}

export interface EpisodePoint {
  episode: number;
  steps: number;
  return: number;
  reached: boolean;
  epsilon: number;
}

export interface Update {
  cell: Cell;
  action: number;
  old: number;
  new: number;
  target: number;
  max_next: number;
  terminal: boolean;
}

export interface StepMsg {
  type: "step";
  episode: number;
  step: number;
  agent: Cell;
  start_of_episode?: boolean;
  from?: Cell;
  action?: number;
  explored?: boolean;
  bumped?: boolean;
  reward?: number;
  update?: Update;
  q_row?: { state: number; values: number[] };
  total?: number;
  done?: boolean;
  result?: "goal" | "timeout" | null;
  epsilon: number;
}

export interface WalkMsg {
  type: "walk_step";
  agent: Cell;
  action?: number;
  step: number;
  done: boolean;
  result?: "goal" | "loop" | "timeout" | null;
  shortest?: number;
}

export type RunStatus = "idle" | "learning" | "training" | "walking";

type MazeMessage =
  | ({ type: "maze" } & MazeInfo)
  | {
      type: "snapshot";
      q: number[][];
      visits: number[];
      epsilon: number;
      episode: number;
      history: EpisodePoint[];
      settings: Settings;
      status: RunStatus;
    }
  | StepMsg
  | { type: "episode"; point: EpisodePoint; epsilon: number }
  | { type: "episodes"; points: EpisodePoint[] }
  | WalkMsg
  | { type: "status"; status: RunStatus }
  | ({ type: "settings" } & Settings)
  | { type: "error"; message: string };

export type MazeCommand =
  | { type: "new_maze"; size: number; seed?: number }
  | { type: "settings"; alpha?: number; gamma?: number; epsilon?: number; decay?: boolean }
  | { type: "speed"; delay_ms: number }
  | { type: "learn" }
  | { type: "pause" }
  | { type: "train_fast"; episodes: number }
  | { type: "walk" }
  | { type: "reset" };

// ---------------------------------------------------------------------- //
// State
// ---------------------------------------------------------------------- //
export interface MazeState {
  maze: MazeInfo | null;
  q: number[][];
  visits: number[];
  epsilon: number;
  episode: number;
  history: EpisodePoint[];
  settings: Settings;
  run: RunStatus;
  agent: Cell | null;
  trail: Cell[];
  lastStep: StepMsg | null; // most recent learning move (has the Q update)
  walk: WalkMsg | null; // most recent greedy-walk move
  view: "learn" | "walk";
}

const initial: MazeState = {
  maze: null,
  q: [],
  visits: [],
  epsilon: 1,
  episode: 0,
  history: [],
  settings: { alpha: 0.5, gamma: 0.95, epsilon: 1, decay: true },
  run: "idle",
  agent: null,
  trail: [],
  lastStep: null,
  walk: null,
  view: "learn",
};

const same = (a: Cell | undefined, b: Cell) => !!a && a[0] === b[0] && a[1] === b[1];
const extend = (trail: Cell[], c: Cell) => (same(trail.at(-1), c) ? trail : [...trail, c]);

function reduce(s: MazeState, m: MazeMessage): MazeState {
  switch (m.type) {
    case "maze": {
      const { type: _t, ...info } = m;
      return { ...s, maze: info, agent: info.start, trail: [info.start], lastStep: null, walk: null, view: "learn" };
    }
    case "snapshot":
      return {
        ...s,
        q: m.q,
        visits: m.visits,
        epsilon: m.epsilon,
        episode: m.episode,
        history: m.history,
        settings: m.settings,
        run: m.status,
      };
    case "step": {
      let { q, visits } = s;
      if (m.q_row) {
        q = s.q.slice();
        q[m.q_row.state] = m.q_row.values;
      }
      if (s.maze) {
        visits = s.visits.slice();
        const idx = m.agent[0] * s.maze.size + m.agent[1];
        visits[idx] = (visits[idx] ?? 0) + 1;
      }
      const trail = m.start_of_episode ? [m.agent] : extend(s.trail, m.agent);
      return {
        ...s,
        q,
        visits,
        agent: m.agent,
        trail,
        epsilon: m.epsilon,
        episode: m.episode,
        lastStep: m.update ? m : s.lastStep,
        view: "learn",
      };
    }
    case "episode":
      return { ...s, history: [...s.history, m.point], epsilon: m.epsilon };
    case "episodes":
      return { ...s, history: [...s.history, ...m.points] };
    case "walk_step":
      return {
        ...s,
        agent: m.agent,
        trail: m.step === 0 ? [m.agent] : extend(s.trail, m.agent),
        walk: m,
        view: "walk",
      };
    case "status":
      return { ...s, run: m.status };
    case "settings": {
      const { type: _t, ...settings } = m;
      return { ...s, settings, epsilon: settings.epsilon };
    }
    default:
      return s;
  }
}

// ---------------------------------------------------------------------- //
// Hook
// ---------------------------------------------------------------------- //
const BASE = process.env.NEXT_PUBLIC_WS_URL ?? "ws://localhost:8000/ws";
const MAZE_URL = BASE.replace(/\/ws\/?$/, "/ws/maze");

export type ConnStatus = "connecting" | "open" | "closed";

export function useMazeSocket(onOpen?: (send: (m: MazeCommand) => void) => void) {
  const [state, dispatch] = useReducer(reduce, initial);
  const [status, setStatus] = useState<ConnStatus>("connecting");
  const wsRef = useRef<WebSocket | null>(null);
  const onOpenRef = useRef(onOpen);
  onOpenRef.current = onOpen;

  const send = useCallback((m: MazeCommand) => {
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(m));
  }, []);

  useEffect(() => {
    let stopped = false;
    let retry: ReturnType<typeof setTimeout> | undefined;
    const connect = () => {
      setStatus("connecting");
      const ws = new WebSocket(MAZE_URL);
      wsRef.current = ws;
      ws.onopen = () => {
        setStatus("open");
        onOpenRef.current?.(send);
      };
      ws.onclose = () => {
        if (stopped) return;
        setStatus("closed");
        retry = setTimeout(connect, 1500);
      };
      ws.onmessage = (ev) => dispatch(JSON.parse(ev.data) as MazeMessage);
    };
    connect();
    return () => {
      stopped = true;
      clearTimeout(retry);
      wsRef.current?.close();
    };
  }, [send]);

  return { ...state, status, send };
}
