"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type {
  ClientMessage,
  EvalResult,
  Frame,
  ServerMessage,
  TrainPoint,
  TrainState,
} from "./protocol";

const WS_URL = process.env.NEXT_PUBLIC_WS_URL ?? "ws://localhost:8000/ws";

export type ConnStatus = "connecting" | "open" | "closed";

export interface Preview {
  update: number;
  frames: Frame[];
}

export interface SocketState {
  status: ConnStatus;
  modelAvailable: boolean;
  frame: Frame | null;
  training: TrainState;
  history: TrainPoint[];
  before: EvalResult | null;
  after: EvalResult | null;
  lastRun: { stopped: boolean; updatesDone: number } | null;
  preview: Preview | null;
  error: { message: string; code?: string } | null;
  rolloutSteps: number;
}

const initial: SocketState = {
  status: "connecting",
  modelAvailable: false,
  frame: null,
  training: { running: false, update: 0, updates: 0 },
  history: [],
  before: null,
  after: null,
  lastRun: null,
  preview: null,
  error: null,
  rolloutSteps: 2048,
};

/** One WebSocket to server.py, reconnecting automatically. */
export function useTreasureSocket(onOpen?: (send: (m: ClientMessage) => void) => void) {
  const [state, setState] = useState<SocketState>(initial);
  const wsRef = useRef<WebSocket | null>(null);
  const onOpenRef = useRef(onOpen);
  onOpenRef.current = onOpen;

  const send = useCallback((msg: ClientMessage) => {
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(msg));
  }, []);

  useEffect(() => {
    let stopped = false;
    let retry: ReturnType<typeof setTimeout> | undefined;

    const connect = () => {
      setState((s) => ({ ...s, status: "connecting" }));
      const ws = new WebSocket(WS_URL);
      wsRef.current = ws;

      ws.onopen = () => {
        setState((s) => ({ ...s, status: "open", error: null }));
        onOpenRef.current?.(send);
      };

      ws.onclose = () => {
        if (stopped) return;
        setState((s) => ({ ...s, status: "closed" }));
        retry = setTimeout(connect, 1500);
      };

      ws.onmessage = (ev) => {
        const msg = JSON.parse(ev.data) as ServerMessage;
        setState((s) => reduce(s, msg));
      };
    };

    connect();
    return () => {
      stopped = true;
      clearTimeout(retry);
      wsRef.current?.close();
    };
  }, [send]);

  const clearError = useCallback(() => setState((s) => ({ ...s, error: null })), []);

  return { ...state, send, clearError };
}

function reduce(s: SocketState, msg: ServerMessage): SocketState {
  switch (msg.type) {
    case "hello":
      return {
        ...s,
        modelAvailable: msg.model_available,
        training: msg.training,
        history: msg.history,
        before: msg.before,
        after: msg.after,
        rolloutSteps: msg.config.rollout_steps,
      };
    case "frame":
      return { ...s, frame: msg };
    case "train_start":
      return {
        ...s,
        training: { running: true, update: 0, updates: msg.updates },
        history: [],
        before: msg.before,
        after: null,
        lastRun: null,
        preview: null,
      };
    case "train_progress":
      return {
        ...s,
        training: { running: true, update: msg.point.update, updates: msg.updates },
        history: [...s.history, msg.point],
      };
    case "train_preview":
      return { ...s, preview: { update: msg.update, frames: msg.frames } };
    case "train_done":
      return {
        ...s,
        training: { ...s.training, running: false },
        before: msg.before,
        after: msg.after,
        modelAvailable: msg.model_available,
        lastRun: { stopped: msg.stopped, updatesDone: msg.updates_done },
      };
    case "error":
      return { ...s, error: { message: msg.message, code: msg.code } };
    default:
      return s;
  }
}
