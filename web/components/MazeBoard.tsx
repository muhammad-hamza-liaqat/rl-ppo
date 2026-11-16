import { useMemo } from "react";
import type { Cell } from "@/lib/protocol";
import type { MazeInfo } from "@/lib/maze";

const C = 60; // cell size in SVG units
const M = 26; // margin for row/column numbers
const DIRS: [number, number][] = [
  [0, -1],
  [0, 1],
  [-1, 0],
  [1, 0],
];

interface Props {
  maze: MazeInfo | null;
  q: number[][];
  visits: number[];
  agent: Cell | null;
  trail: Cell[];
  updated: { cell: Cell; action: number } | null;
  selected: Cell | null;
  showValues: boolean;
  showArrows: boolean;
  moveMs: number;
  onHoverCell: (cell: Cell | null) => void;
}

export function MazeBoard({
  maze,
  q,
  visits,
  agent,
  trail,
  updated,
  selected,
  showValues,
  showArrows,
  moveMs,
  onHoverCell,
}: Props) {
  const n = maze?.size ?? 11;
  const span = n * C;
  const walls = useMemo(() => new Set((maze?.walls ?? []).map(([r, c]) => `${r},${c}`)), [maze]);

  // V(s) = best Q-value in each square; scale the tint by the best value on the board.
  const values = q.map((row) => (row.length ? Math.max(...row) : 0));
  const vmax = Math.max(0.05, ...values);

  const cells: React.ReactNode[] = [];
  if (maze) {
    for (let r = 0; r < n; r++) {
      for (let c = 0; c < n; c++) {
        const x = c * C;
        const y = r * C;
        if (walls.has(`${r},${c}`)) {
          cells.push(<rect key={`w${r}-${c}`} className="maze__wall" x={x} y={y} width={C} height={C} />);
          continue;
        }
        const s = r * n + c;
        const v = values[s] ?? 0;
        const seen = (visits[s] ?? 0) > 0;
        const share = Math.max(0, Math.min(1, v / vmax));
        const fill =
          showValues && v > 0
            ? `color-mix(in oklab, var(--treasure) ${Math.round(12 + share * 70)}%, var(--sheet))`
            : seen
              ? "var(--visited)"
              : "var(--sheet)";
        cells.push(<rect key={`c${r}-${c}`} x={x} y={y} width={C} height={C} style={{ fill }} />);

        const row = q[s];
        // Only draw an arrow when exactly one move is best (untried moves tie at 0 early on).
        const top = row ? Math.max(...row) : 0;
        const unique = row ? row.filter((v) => Math.abs(v - top) < 1e-9).length === 1 : false;
        if (showArrows && row && seen && unique) {
          const best = row.indexOf(top);
          const [dx, dy] = DIRS[best];
          const cx = x + C / 2;
          const cy = y + C / 2;
          const px = -dy;
          const py = dx;
          const tip = 15;
          const back = 11;
          const wing = 8;
          cells.push(
            <polygon
              key={`a${r}-${c}`}
              className={v > 0 ? "maze__arrow maze__arrow--good" : "maze__arrow"}
              opacity={v > 0 ? 0.55 + share * 0.45 : 0.45}
              points={`${cx + dx * tip},${cy + dy * tip} ${cx - dx * back + px * wing},${cy - dy * back + py * wing} ${cx - dx * (back - 6)},${cy - dy * (back - 6)} ${cx - dx * back - px * wing},${cy - dy * back - py * wing}`}
            />,
          );
        }
      }
    }
  }

  const center = (cell: Cell) => [cell[1] * C + C / 2, cell[0] * C + C / 2] as const;

  const onMove = (e: React.PointerEvent<SVGSVGElement>) => {
    if (!maze) return;
    const svg = e.currentTarget;
    const pt = svg.createSVGPoint();
    pt.x = e.clientX;
    pt.y = e.clientY;
    const ctm = svg.getScreenCTM();
    if (!ctm) return;
    const p = pt.matrixTransform(ctm.inverse());
    const col = Math.floor(p.x / C);
    const row = Math.floor(p.y / C);
    if (row < 0 || col < 0 || row >= n || col >= n || walls.has(`${row},${col}`)) onHoverCell(null);
    else onHoverCell([row, col]);
  };

  return (
    <svg
      className="maze"
      viewBox={`${-M} ${-M} ${span + M + 4} ${span + M + 4}`}
      role="img"
      aria-label={
        maze
          ? `Maze, ${n} by ${n}. Start at row ${maze.start[0]} column ${maze.start[1]}, goal at row ${maze.goal[0]} column ${maze.goal[1]}.`
          : "Maze loading"
      }
      onPointerMove={onMove}
      onPointerLeave={() => onHoverCell(null)}
    >
      {Array.from({ length: n }, (_, i) => (
        <g key={`n${i}`} className="board__coord maze__coord">
          <text x={i * C + C / 2} y={-9} textAnchor="middle">
            {i}
          </text>
          <text x={-12} y={i * C + C / 2 + 4} textAnchor="middle">
            {i}
          </text>
        </g>
      ))}
      <rect className="maze__sheet" x={0} y={0} width={span} height={span} />
      {cells}

      {maze && (
        <>
          {/* grid lines on top of the fills */}
          {Array.from({ length: n - 1 }, (_, i) => (
            <g key={`g${i}`} className="maze__grid">
              <line x1={(i + 1) * C} y1={0} x2={(i + 1) * C} y2={span} />
              <line x1={0} y1={(i + 1) * C} x2={span} y2={(i + 1) * C} />
            </g>
          ))}

          <rect
            className="maze__start"
            x={maze.start[1] * C + 6}
            y={maze.start[0] * C + 6}
            width={C - 12}
            height={C - 12}
            rx={6}
          />
          {(() => {
            const [x, y] = center(maze.goal);
            return (
              <g className="board__treasure" transform={`translate(${x} ${y})`}>
                <circle r={17} />
                <circle className="board__treasure-ring" r={10} />
              </g>
            );
          })()}

          {trail.slice(0, -1).map((cell, i, arr) => {
            const [x, y] = center(cell);
            return (
              <circle
                key={`t${i}`}
                className="board__trail"
                cx={x}
                cy={y}
                r={4.5}
                opacity={Math.max(0.15, 0.75 - (arr.length - i) * 0.03)}
              />
            );
          })}

          {updated && (
            <rect
              className="maze__updated"
              x={updated.cell[1] * C + 2}
              y={updated.cell[0] * C + 2}
              width={C - 4}
              height={C - 4}
              rx={4}
            />
          )}
          {selected && (
            <rect
              className="maze__selected"
              x={selected[1] * C + 1}
              y={selected[0] * C + 1}
              width={C - 2}
              height={C - 2}
            />
          )}

          {agent && (
            <g
              className="board__agent-wrap"
              style={{
                transform: `translate(${center(agent)[0]}px, ${center(agent)[1]}px)`,
                transitionDuration: `${Math.min(moveMs, 160)}ms`,
              }}
            >
              <circle className="board__agent" r={15} />
              <circle className="board__agent-eye" cx={-5} cy={-4} r={2.6} />
              <circle className="board__agent-eye" cx={5} cy={-4} r={2.6} />
            </g>
          )}
          <rect className="maze__frame" x={0} y={0} width={span} height={span} />
        </>
      )}
    </svg>
  );
}
