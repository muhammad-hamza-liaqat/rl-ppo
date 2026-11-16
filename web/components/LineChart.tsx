"use client";

import { useEffect, useRef, useState } from "react";

export interface Point {
  x: number;
  y: number;
}

interface Props {
  title: string;
  description: string;
  points: Point[];
  format: (v: number) => string;
  domain: [number, number] | "auto";
  xMax: number;
  xLabel: string; // "update", "episode"
  hover: number | null; // x under the pointer, shared by sibling charts
  onHover: (x: number | null) => void;
  reference?: { y: number; label: string };
}

const H = 132;
const PAD = { top: 10, right: 12, bottom: 22, left: 44 };

function useWidth<T extends HTMLElement>() {
  const ref = useRef<T>(null);
  const [w, setW] = useState(320);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver(([e]) => setW(Math.max(200, e.contentRect.width)));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  return [ref, w] as const;
}

export function LineChart({
  title,
  description,
  points,
  format,
  domain,
  xMax: xMaxProp,
  xLabel,
  hover,
  onHover,
  reference,
}: Props) {
  const [ref, width] = useWidth<HTMLDivElement>();

  let [lo, hi] =
    domain === "auto"
      ? [Math.min(0, ...points.map((p) => p.y)), Math.max(0.02, reference?.y ?? 0, ...points.map((p) => p.y)) * 1.1]
      : domain;
  if (hi === lo) hi = lo + 1;

  const xMax = Math.max(xMaxProp, points.at(-1)?.x ?? 1, 2);
  const innerW = width - PAD.left - PAD.right;
  const innerH = H - PAD.top - PAD.bottom;
  const x = (u: number) => PAD.left + ((u - 1) / (xMax - 1)) * innerW;
  const y = (v: number) => PAD.top + (1 - (Math.min(hi, Math.max(lo, v)) - lo) / (hi - lo)) * innerH;

  const path = points.map((p, i) => `${i ? "L" : "M"}${x(p.x).toFixed(1)},${y(p.y).toFixed(1)}`).join("");
  const ticks = [lo, (lo + hi) / 2, hi];
  const last = points.at(-1);
  const hovered = hover !== null ? points.find((p) => p.x === hover) : undefined;
  const shown = hovered ?? last;

  const pick = (clientX: number, target: SVGRectElement) => {
    if (!points.length) return;
    const box = target.getBoundingClientRect();
    const u = 1 + ((clientX - box.left) / box.width) * (xMax - 1);
    let best = points[0];
    for (const p of points) if (Math.abs(p.x - u) < Math.abs(best.x - u)) best = p;
    onHover(best.x);
  };

  return (
    <figure className="chart">
      <figcaption className="chart__head">
        <span className="chart__title">{title}</span>
        <span className="chart__value">
          {shown ? format(shown.y) : "–"}
          {hovered && (
            <span className="chart__at">
              {" "}
              at {xLabel} {hovered.x}
            </span>
          )}
        </span>
      </figcaption>
      <p className="chart__desc">{description}</p>
      <div ref={ref} className="chart__plot">
        <svg width={width} height={H} role="img" aria-label={`${title} over ${points.length} ${xLabel}s`}>
          {ticks.map((t) => (
            <g key={t} className="chart__grid">
              <line x1={PAD.left} x2={width - PAD.right} y1={y(t)} y2={y(t)} />
              <text x={PAD.left - 8} y={y(t) + 4} textAnchor="end">
                {format(t)}
              </text>
            </g>
          ))}
          <g className="chart__axis">
            <text x={PAD.left} y={H - 5}>
              1
            </text>
            <text x={width - PAD.right} y={H - 5} textAnchor="end">
              {xLabel} {xMax}
            </text>
          </g>

          {reference && (
            <g className="chart__ref">
              <line x1={PAD.left} x2={width - PAD.right} y1={y(reference.y)} y2={y(reference.y)} />
              <text x={width - PAD.right} y={y(reference.y) - 6} textAnchor="end">
                {reference.label}
              </text>
            </g>
          )}

          {points.length > 0 && <path className="chart__line" d={path} />}
          {last && !hovered && <circle className="chart__dot" cx={x(last.x)} cy={y(last.y)} r={4.5} />}

          {hovered && (
            <g>
              <line className="chart__cross" x1={x(hovered.x)} x2={x(hovered.x)} y1={PAD.top} y2={H - PAD.bottom} />
              <circle className="chart__dot" cx={x(hovered.x)} cy={y(hovered.y)} r={4.5} />
            </g>
          )}

          <rect
            className="chart__hit"
            x={PAD.left}
            y={0}
            width={innerW}
            height={H}
            onPointerMove={(e) => pick(e.clientX, e.currentTarget)}
            onPointerDown={(e) => pick(e.clientX, e.currentTarget)}
            onPointerLeave={() => onHover(null)}
          />
        </svg>
      </div>
    </figure>
  );
}
