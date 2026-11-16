"use client";

import { useCallback, useEffect, useReducer, useRef, useState } from "react";

// ---------------------------------------------------------------------- //
// Messages from rlhf_session.py
// ---------------------------------------------------------------------- //
export interface Palette {
  colors: string[];
  score: number | null; // raw reward-model score (null before any picks)
}

export interface Feature {
  key: string;
  label: string;
  description: string;
}

export interface TunePoint {
  update: number;
  score: number; // normalised reward-model score: 0 = an average random palette
  kl: number; // distance from the original generator, per palette
  unique: number; // share of distinct palettes in the batch
}

export interface RewardModelInfo {
  weights: number[];
  agreement: number | null;
  picks: number;
  skipped: number;
  simulated: number;
  hidden_taste: number[] | null;
  extremes: { best: Palette; worst: Palette } | null;
}

export interface Gallery {
  original: Palette[];
  tuned: Palette[];
  original_avg: number | null;
  tuned_avg: number | null;
}

type Msg =
  | {
      type: "state";
      features: Feature[];
      min_picks: number;
      simulated_description: string;
      beta: number;
      tuned_updates: number;
      history: TunePoint[];
      tuning: boolean;
    }
  | { type: "pair"; a: Palette; b: Palette; source: "original" | "tuned" }
  | ({ type: "reward_model" } & RewardModelInfo)
  | { type: "tune_start"; updates: number; beta: number }
  | { type: "tune_progress"; point: TunePoint; done_in_run: number; updates: number; samples: Palette[] }
  | { type: "tune_done"; stopped: boolean; tuned_updates: number }
  | ({ type: "gallery" } & Gallery)
  | { type: "error"; message: string; code?: string };

export type RLHFCommand =
  | { type: "pair" }
  | { type: "choose"; pick: "a" | "b" | "skip" }
  | { type: "simulate"; n: number }
  | { type: "tune"; updates: number; beta: number }
  | { type: "stop_tune" }
  | { type: "gallery" }
  | { type: "reset" };

// ---------------------------------------------------------------------- //
export interface RLHFState {
  features: Feature[];
  minPicks: number;
  simulatedDescription: string;
  pair: { a: Palette; b: Palette; source: "original" | "tuned" } | null;
  rm: RewardModelInfo | null;
  tuning: { running: boolean; done: number; updates: number; beta: number };
  tunedUpdates: number;
  history: TunePoint[];
  samples: Palette[];
  gallery: Gallery | null;
  lastRun: { stopped: boolean } | null;
  error: string | null;
}

const initial: RLHFState = {
  features: [],
  minPicks: 8,
  simulatedDescription: "",
  pair: null,
  rm: null,
  tuning: { running: false, done: 0, updates: 0, beta: 0.1 },
  tunedUpdates: 0,
  history: [],
  samples: [],
  gallery: null,
  lastRun: null,
  error: null,
};

function reduce(s: RLHFState, m: Msg | { type: "clear_error" }): RLHFState {
  switch (m.type) {
    case "state":
      return {
        ...initial,
        features: m.features,
        minPicks: m.min_picks,
        simulatedDescription: m.simulated_description,
        tunedUpdates: m.tuned_updates,
        history: m.history,
        tuning: { ...initial.tuning, running: m.tuning, beta: m.beta },
      };
    case "pair":
      return { ...s, pair: { a: m.a, b: m.b, source: m.source }, error: null };
    case "reward_model": {
      const { type: _t, ...rm } = m;
      return { ...s, rm };
    }
    case "tune_start":
      return { ...s, tuning: { running: true, done: 0, updates: m.updates, beta: m.beta }, lastRun: null, error: null };
    case "tune_progress":
      return {
        ...s,
        tuning: { ...s.tuning, running: true, done: m.done_in_run, updates: m.updates },
        history: [...s.history, m.point],
        tunedUpdates: m.point.update,
        samples: m.samples,
      };
    case "tune_done":
      return { ...s, tuning: { ...s.tuning, running: false }, tunedUpdates: m.tuned_updates, lastRun: { stopped: m.stopped } };
    case "gallery": {
      const { type: _t, ...g } = m;
      return { ...s, gallery: g };
    }
    case "error":
      return { ...s, error: m.message };
    case "clear_error":
      return { ...s, error: null };
    default:
      return s;
  }
}

const BASE = process.env.NEXT_PUBLIC_WS_URL ?? "ws://localhost:8000/ws";
const URL = BASE.replace(/\/ws\/?$/, "/ws/rlhf");

export type ConnStatus = "connecting" | "open" | "closed";

export function useRLHFSocket() {
  const [state, dispatch] = useReducer(reduce, initial);
  const [status, setStatus] = useState<ConnStatus>("connecting");
  const wsRef = useRef<WebSocket | null>(null);

  const send = useCallback((m: RLHFCommand) => {
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(m));
  }, []);

  useEffect(() => {
    let stopped = false;
    let retry: ReturnType<typeof setTimeout> | undefined;
    const connect = () => {
      setStatus("connecting");
      const ws = new WebSocket(URL);
      wsRef.current = ws;
      ws.onopen = () => setStatus("open");
      ws.onclose = () => {
        if (stopped) return;
        setStatus("closed");
        retry = setTimeout(connect, 1500);
      };
      ws.onmessage = (ev) => dispatch(JSON.parse(ev.data) as Msg);
    };
    connect();
    return () => {
      stopped = true;
      clearTimeout(retry);
      wsRef.current?.close();
    };
  }, []);

  const clearError = useCallback(() => dispatch({ type: "clear_error" }), []);
  return { ...state, status, send, clearError };
}
