"use client";

import { useEffect, useRef, useState } from "react";

const HOLD_TICKS = 4; // pause on the last frame before loading the next game

export interface PreviewOf<F> {
  update: number;
  frames: F[];
}

/**
 * Plays training preview games one frame at a time.
 * When a game ends it pauses briefly, then jumps to the newest preview.
 */
export function usePreviewPlayer<F>(preview: PreviewOf<F> | null, delayMs: number, active: boolean) {
  const latest = useRef(preview);
  latest.current = preview;
  const [st, setSt] = useState<{ p: PreviewOf<F> | null; i: number }>({ p: null, i: 0 });

  useEffect(() => {
    if (!active) return;
    const t = setInterval(() => {
      setSt(({ p, i }) => {
        const next = latest.current;
        if (!p) return next ? { p: next, i: 0 } : { p, i };
        if (i >= p.frames.length - 1 + HOLD_TICKS) {
          return next && next !== p ? { p: next, i: 0 } : { p, i };
        }
        return { p, i: i + 1 };
      });
    }, delayMs);
    return () => clearInterval(t);
  }, [active, delayMs]);

  if (!st.p) return { frame: null as F | null, frames: [] as F[], update: null as number | null };
  const i = Math.min(st.i, st.p.frames.length - 1);
  return { frame: st.p.frames[i], frames: st.p.frames.slice(0, i + 1), update: st.p.update };
}
