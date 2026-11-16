import type { Cell, Frame } from "@/lib/protocol";
import { ACTIONS } from "@/lib/protocol";

const C = 100; // cell size in SVG units
const M = 30; // margin for row/column numbers

// direction vectors in (x, y) for up, down, left, right
const DIRS: [number, number][] = [
  [0, -1],
  [0, 1],
  [-1, 0],
  [1, 0],
];

interface Props {
  frame: Frame | null;
  trail: Cell[];
  moveMs: number;
  showArrows: boolean;
  size?: number;
}

export function Board({ frame, trail, moveMs, showArrows, size = 6 }: Props) {
  const n = frame?.board.size ?? size;
  const span = n * C;
  const view = `${-M} ${-M} ${span + M + 6} ${span + M + 6}`;

  const center = (cell: Cell) => [cell[1] * C + C / 2, cell[0] * C + C / 2] as const;

  const label = frame
    ? describe(frame)
    : "Empty board. The game appears here once it starts.";

  const resultClass = frame?.done && frame.result ? `board--${frame.result}` : "";

  return (
    <svg className={`board ${resultClass}`} viewBox={view} role="img" aria-label={label}>
      {/* row and column numbers, matching the observation values */}
      {Array.from({ length: n }, (_, i) => (
        <g key={`n${i}`} className="board__coord">
          <text x={i * C + C / 2} y={-11} textAnchor="middle">
            {i}
          </text>
          <text x={-13} y={i * C + C / 2 + 5} textAnchor="middle">
            {i}
          </text>
        </g>
      ))}

      <rect className="board__sheet" x={0} y={0} width={span} height={span} rx={6} />
      {Array.from({ length: n - 1 }, (_, i) => (
        <g key={`g${i}`} className="board__grid">
          <line x1={(i + 1) * C} y1={0} x2={(i + 1) * C} y2={span} />
          <line x1={0} y1={(i + 1) * C} x2={span} y2={(i + 1) * C} />
        </g>
      ))}

      {frame && (
        <>
          {/* where the agent has been this game */}
          {trail.slice(0, -1).map((cell, i, arr) => {
            const [x, y] = center(cell);
            const age = arr.length - i;
            return (
              <circle
                key={`t${i}`}
                className="board__trail"
                cx={x}
                cy={y}
                r={7}
                opacity={Math.max(0.12, 0.7 - age * 0.06)}
              />
            );
          })}

          {frame.board.traps.map((cell) => {
            const [x, y] = center(cell);
            return (
              <g key={`x${cell}`} className="board__trap" transform={`translate(${x} ${y})`}>
                <rect x={-40} y={-40} width={80} height={80} rx={10} />
                <path d="M -17 -17 L 17 17 M 17 -17 L -17 17" />
              </g>
            );
          })}

          {(() => {
            const [x, y] = center(frame.board.treasure);
            return (
              <g className="board__treasure" transform={`translate(${x} ${y})`}>
                <circle r={27} />
                <circle className="board__treasure-ring" r={17} />
              </g>
            );
          })()}

          <Agent
            frame={frame}
            pos={center(frame.board.agent)}
            moveMs={moveMs}
            showArrows={showArrows}
          />
        </>
      )}
    </svg>
  );
}

function Agent({
  frame,
  pos,
  moveMs,
  showArrows,
}: {
  frame: Frame;
  pos: readonly [number, number];
  moveMs: number;
  showArrows: boolean;
}) {
  const probs = frame.probs;
  const best = probs ? probs.indexOf(Math.max(...probs)) : -1;
  const uniform = probs ? Math.max(...probs) - Math.min(...probs) < 0.01 : false;

  return (
    <g
      className="board__agent-wrap"
      style={{
        transform: `translate(${pos[0]}px, ${pos[1]}px)`,
        transitionDuration: `${Math.min(moveMs, 260)}ms`,
      }}
    >
      {showArrows && probs && !frame.done &&
        probs.map((p, i) => {
          const [dx, dy] = DIRS[i];
          const start = 34;
          const len = 8 + p * 46;
          const end = start + len;
          const w = 2 + p * 7;
          const head = 7 + p * 6;
          const tipX = dx * (end + head);
          const tipY = dy * (end + head);
          // perpendicular for the arrowhead
          const px = -dy;
          const py = dx;
          return (
            <g
              key={ACTIONS[i]}
              className={`board__arrow ${i === best && !uniform ? "board__arrow--best" : ""}`}
              opacity={0.3 + p * 0.7}
            >
              <line
                x1={dx * start}
                y1={dy * start}
                x2={dx * end}
                y2={dy * end}
                strokeWidth={w}
              />
              <polygon
                points={`${tipX},${tipY} ${dx * end + px * head},${dy * end + py * head} ${dx * end - px * head},${dy * end - py * head}`}
              />
            </g>
          );
        })}
      <circle className="board__halo" r={38} />
      <circle className="board__agent" r={26} />
      <circle className="board__agent-eye" cx={-8} cy={-6} r={4} />
      <circle className="board__agent-eye" cx={8} cy={-6} r={4} />
    </g>
  );
}

function describe(f: Frame) {
  const a = f.board.agent;
  const t = f.board.treasure;
  const traps = f.board.traps.map((c) => `row ${c[0]} column ${c[1]}`).join(", ");
  return `Agent at row ${a[0]} column ${a[1]}. Treasure at row ${t[0]} column ${t[1]}. Traps at ${traps}.`;
}
