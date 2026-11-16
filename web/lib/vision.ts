"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { PreviewOf } from "./usePreviewPlayer";

// ---------------------------------------------------------------------- //
// Messages from vision_session.py
// ---------------------------------------------------------------------- //
export type Box = [number, number, number]; // centre x, centre y, half size (pixels)
export type VisionResult = "found" | "wrong" | "timeout";
export type VisionSource = "trained" | "random" | "human" | "preview";

export interface VisionFrame {
  type: "frame";
  source: VisionSource;
  episode: number;
  step: number;
  size: number;
  box: Box;
  target: Box;
  iou: number;
  fovea: number[]; // 12x12, 0..1
  probs: number[] | null;
  value: number | null;
  action: number | null;
  reward: number | null;
  total: number;
  done: boolean;
  result: VisionResult | null;
  max_steps: number;
  image?: string; // base64 of size*size grayscale bytes, first frame of an episode only
  context?: number[]; // 8x8 blurry whole image, first frame only
}

export interface VisionPoint {
  update: number;
  steps: number;
  avg_reward: number;
  found: number;
  entropy: number;
  kl: number;
  elapsed: number;
}

export interface VisionEval {
  found: number;
  wrong: number;
  timeout: number;
  steps: number;
}

export interface TrainState {
  running: boolean;
  update: number;
  updates: number;
}

type Msg =
  | {
      type: "hello";
      model_available: boolean;
      training: TrainState;
      history: VisionPoint[];
      before: VisionEval | null;
      after: VisionEval | null;
      actions: string[];
      hit_iou: number;
    }
  | VisionFrame
  | { type: "train_start"; updates: number; before: VisionEval }
  | { type: "train_progress"; point: VisionPoint; updates: number }
  | { type: "train_preview"; update: number; frames: VisionFrame[] }
  | {
      type: "train_done";
      stopped: boolean;
      updates_done: number;
      before: VisionEval | null;
      after: VisionEval | null;
      model_available: boolean;
    }
  | { type: "error"; message: string; code?: string };

export type VisionCommand =
  | { type: "watch"; agent: "trained" | "random" }
  | { type: "stop" }
  | { type: "speed"; delay_ms: number }
  | { type: "human_reset" }
  | { type: "human_move"; action: number }
  | { type: "train"; updates: number }
  | { type: "stop_train" };

/** An episode as the page draws it: the scan plus every box so far. */
export interface Episode {
  key: string;
  image: Uint8Array | null;
  context: number[] | null;
  boxes: Box[];
}

export interface VisionState {
  status: "connecting" | "open" | "closed";
  modelAvailable: boolean;
  actions: string[];
  hitIou: number;
  frame: VisionFrame | null;
  episode: Episode | null;
  training: TrainState;
  history: VisionPoint[];
  before: VisionEval | null;
  after: VisionEval | null;
  lastRun: { stopped: boolean; updatesDone: number } | null;
  preview: PreviewOf<VisionFrame> | null;
  error: { message: string; code?: string } | null;
}

export function decodeImage(b64: string): Uint8Array {
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

/** Fold a stream of frames into the current episode (image + box history). */
export function nextEpisode(prev: Episode | null, f: VisionFrame): Episode {
  const key = `${f.source}-${f.episode}`;
  if (f.image || !prev || prev.key !== key) {
    return {
      key,
      image: f.image ? decodeImage(f.image) : (prev?.key === key ? prev.image : null),
      context: f.context ?? null,
      boxes: [f.box],
    };
  }
  return { ...prev, boxes: [...prev.boxes, f.box] };
}

const BASE = process.env.NEXT_PUBLIC_WS_URL ?? "ws://localhost:8000/ws";
const URL = BASE.replace(/\/ws\/?$/, "/ws/vision");

const initial: Omit<VisionState, "status"> = {
  modelAvailable: false,
  actions: ["up", "down", "left", "right", "bigger", "smaller", "found it"],
  hitIou: 0.5,
  frame: null,
  episode: null,
  training: { running: false, update: 0, updates: 0 },
  history: [],
  before: null,
  after: null,
  lastRun: null,
  preview: null,
  error: null,
};

export function useVisionSocket(onOpen?: (send: (m: VisionCommand) => void) => void) {
  const [state, setState] = useState(initial);
  const [status, setStatus] = useState<VisionState["status"]>("connecting");
  const wsRef = useRef<WebSocket | null>(null);
  const onOpenRef = useRef(onOpen);
  onOpenRef.current = onOpen;

  const send = useCallback((m: VisionCommand) => {
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
      ws.onopen = () => {
        setStatus("open");
        onOpenRef.current?.(send);
      };
      ws.onclose = () => {
        if (stopped) return;
        setStatus("closed");
        retry = setTimeout(connect, 1500);
      };
      ws.onmessage = (ev) => {
        const m = JSON.parse(ev.data) as Msg;
        setState((s) => reduce(s, m));
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
  return { ...state, status, send, clearError };
}

function reduce(s: typeof initial, m: Msg): typeof initial {
  switch (m.type) {
    case "hello":
      return {
        ...s,
        modelAvailable: m.model_available,
        training: m.training,
        history: m.history,
        before: m.before,
        after: m.after,
        actions: m.actions,
        hitIou: m.hit_iou,
      };
    case "frame":
      return { ...s, frame: m, episode: nextEpisode(s.episode, m) };
    case "train_start":
      return {
        ...s,
        training: { running: true, update: 0, updates: m.updates },
        history: [],
        before: m.before,
        after: null,
        lastRun: null,
        preview: null,
      };
    case "train_progress":
      return {
        ...s,
        training: { running: true, update: m.point.update, updates: m.updates },
        history: [...s.history, m.point],
      };
    case "train_preview":
      return { ...s, preview: { update: m.update, frames: m.frames } };
    case "train_done":
      return {
        ...s,
        training: { ...s.training, running: false },
        before: m.before,
        after: m.after,
        modelAvailable: m.model_available,
        lastRun: { stopped: m.stopped, updatesDone: m.updates_done },
      };
    case "error":
      return { ...s, error: { message: m.message, code: m.code } };
    default:
      return s;
  }
}
