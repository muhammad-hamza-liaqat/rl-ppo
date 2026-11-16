"use client";

import { useEffect, useRef } from "react";
import type { Box } from "@/lib/vision";

/** Draw a grayscale image (0..1 floats or 0..255 bytes) onto a canvas. */
export function GrayCanvas({
  pixels,
  side,
  className,
  smooth,
  label,
}: {
  pixels: ArrayLike<number> | null;
  side: number;
  className?: string;
  smooth?: boolean;
  label: string;
}) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const c = ref.current;
    if (!c) return;
    const ctx = c.getContext("2d");
    if (!ctx) return;
    const data = ctx.createImageData(side, side);
    if (pixels) {
      const bytes = pixels instanceof Uint8Array;
      for (let i = 0; i < side * side; i++) {
        const v = bytes ? pixels[i] : Math.round(Math.max(0, Math.min(1, pixels[i])) * 255);
        data.data[i * 4] = v;
        data.data[i * 4 + 1] = v;
        data.data[i * 4 + 2] = v;
        data.data[i * 4 + 3] = 255;
      }
    }
    ctx.putImageData(data, 0, 0);
  }, [pixels, side]);
  return (
    <canvas
      ref={ref}
      width={side}
      height={side}
      className={`${className ?? ""} ${smooth ? "canvas--smooth" : "canvas--pixel"}`}
      role="img"
      aria-label={label}
    />
  );
}

/** The scan with the agent's search box, its path, and (optionally) the answer. */
export function ScanView({
  image,
  size,
  boxes,
  target,
  showTarget,
  result,
  moveMs,
}: {
  image: Uint8Array | null;
  size: number;
  boxes: Box[];
  target: Box | null;
  showTarget: boolean;
  result: "found" | "wrong" | "timeout" | null;
  moveMs: number;
}) {
  const box = boxes.at(-1) ?? null;
  const rect = (b: Box) => ({ x: b[0] - b[2], y: b[1] - b[2], width: b[2] * 2, height: b[2] * 2 });
  return (
    <div className={`scan ${result ? `scan--${result}` : ""}`}>
      <GrayCanvas pixels={image} side={size} smooth className="scan__img" label="Synthetic grayscale scan" />
      <svg className="scan__overlay" viewBox={`0 0 ${size} ${size}`} aria-hidden>
        {boxes.slice(0, -1).map((b, i, arr) => (
          <rect
            key={i}
            {...rect(b)}
            className="scan__trail"
            opacity={Math.max(0.12, 0.6 - (arr.length - i) * 0.05)}
          />
        ))}
        {target && showTarget && <rect {...rect(target)} className="scan__target" />}
        {box && (
          <rect
            {...rect(box)}
            className="scan__box"
            style={{ transitionDuration: `${Math.min(moveMs, 220)}ms` }}
          />
        )}
      </svg>
    </div>
  );
}
