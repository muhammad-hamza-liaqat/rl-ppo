"use client";

import { useEffect, useRef, useState } from "react";
import { LineChart } from "@/components/LineChart";
import { MazeBoard } from "@/components/MazeBoard";
import { signed } from "@/components/Thinking";
import type { Cell } from "@/lib/protocol";
import { ACTIONS } from "@/lib/protocol";
import { type MazeCommand, type MazeState, type StepMsg, useMazeSocket } from "@/lib/maze";

const SIZES = [7, 9, 11, 13, 15];
const num = (v: number, d = 3) => (v < 0 ? `−${Math.abs(v).toFixed(d)}` : v.toFixed(d));
// Bracket negatives that follow an operator, so "0.5 × (−0.01)" never reads as "× −".
const op = (v: number, d = 3) => (v < 0 ? `(${num(v, d)})` : num(v, d));

export default function MazePage() {
  const [delayMs, setDelayMs] = useState(60);
  const [showValues, setShowValues] = useState(true);
  const [showArrows, setShowArrows] = useState(true);
  const [hoverCell, setHoverCell] = useState<Cell | null>(null);
  const [size, setSize] = useState(11);
  const delayRef = useRef(delayMs);
  delayRef.current = delayMs;

  const m = useMazeSocket((send) => send({ type: "speed", delay_ms: delayRef.current }));
  const { send, status } = m;
  const connected = status === "open";

  useEffect(() => {
    if (connected) send({ type: "speed", delay_ms: delayMs });
  }, [delayMs, connected, send]);

  useEffect(() => {
    if (m.maze) setSize(m.maze.size);
  }, [m.maze]);

  const busy = m.run === "training";
  const updated =
    m.view === "learn" && m.lastStep?.update ? { cell: m.lastStep.update.cell, action: m.lastStep.update.action } : null;
  const inspect = hoverCell ?? m.agent;

  return (
    <main className="page">
      <header className="masthead">
        <div>
          <h1>Maze</h1>
          <p className="lede">
            A Q-learning agent learns this one maze by keeping a table: one row per square, one value per move. It
            starts knowing nothing. Watch the values spread back from the goal as it explores.
          </p>
        </div>
        <p className={`conn conn--${status}`} role="status">
          <span className="conn__dot" aria-hidden />
          {status === "open" ? "Connected" : status === "connecting" ? "Connecting…" : "Server not running"}
        </p>
      </header>

      {status === "closed" && (
        <div className="notice notice--error">
          <p>
            <strong>Can&apos;t reach the Python server.</strong> In the <code>rl-ppo-demo</code> folder, run:
          </p>
          <pre>source env/bin/activate{"\n"}python run.py</pre>
          <p>This page reconnects by itself once it&apos;s running.</p>
        </div>
      )}

      <div className="stage">
        <div className="stage__board">
          <div className="board-frame">
            <MazeBoard
              maze={m.maze}
              q={m.q}
              visits={m.visits}
              agent={m.agent}
              trail={m.trail}
              updated={updated}
              selected={hoverCell}
              showValues={showValues}
              showArrows={showArrows}
              moveMs={delayMs}
              onHoverCell={setHoverCell}
            />
          </div>
          <MazeStepLine state={m} />
          <div className="board-controls">
            <label className="speed">
              <span>Move delay</span>
              <input
                type="range"
                min={0}
                max={400}
                step={10}
                value={delayMs}
                onChange={(e) => setDelayMs(Number(e.target.value))}
              />
              <output>{delayMs} ms</output>
            </label>
            <label className="check">
              <input type="checkbox" checked={showValues} onChange={(e) => setShowValues(e.target.checked)} />
              Shade by value
            </label>
            <label className="check">
              <input type="checkbox" checked={showArrows} onChange={(e) => setShowArrows(e.target.checked)} />
              Best-move arrows
            </label>
          </div>
        </div>

        <aside className="stage__side">
          <section className="panel">
            <h2>Learn</h2>
            <p className="panel__note">
              Episode {m.episode}. Exploration ε is {m.epsilon.toFixed(2)}, so {Math.round(m.epsilon * 100)}% of
              moves are random.
            </p>
            <div className="actions">
              {m.run === "learning" ? (
                <button type="button" className="btn" onClick={() => send({ type: "pause" })}>
                  Pause
                </button>
              ) : (
                <button type="button" className="btn" disabled={!connected || busy} onClick={() => send({ type: "learn" })}>
                  {m.episode ? "Keep learning" : "Start learning"}
                </button>
              )}
              <button
                type="button"
                className="btn btn--quiet"
                disabled={!connected || busy}
                onClick={() => send({ type: "train_fast", episodes: 50 })}
              >
                Learn 50 episodes instantly
              </button>
              <button
                type="button"
                className="btn btn--quiet"
                disabled={!connected || busy || m.run === "walking"}
                onClick={() => send({ type: "walk" })}
              >
                Walk the learned path
              </button>
            </div>
          </section>

          {m.view === "learn" && m.lastStep?.update ? (
            <UpdateExplainer step={m.lastStep} alpha={m.settings.alpha} gamma={m.settings.gamma} />
          ) : (
            <section className="panel">
              <h2>How one update works</h2>
              <p className="panel__note">
                After every move, the agent updates one number in its table: the value of the move it just made. The
                new value moves toward the reward plus the best value of the square it landed on. Start learning to see
                each update with real numbers.
              </p>
            </section>
          )}

          <QInspector cell={inspect} hovering={!!hoverCell} state={m} />

          <Settings state={m} send={send} connected={connected} size={size} setSize={setSize} />
        </aside>
      </div>

      <MazeCharts state={m} />
    </main>
  );
}

// ---------------------------------------------------------------------- //
function MazeStepLine({ state: m }: { state: MazeState }) {
  if (m.view === "walk" && m.walk) {
    const w = m.walk;
    const shortest = w.shortest ?? m.maze?.shortest;
    if (!w.done) return <p className="stepline">Following the best-known move from each square. Move {w.step}.</p>;
    if (w.result === "goal")
      return (
        <p className="stepline stepline--win" role="status">
          <strong>Reached the goal in {w.step} moves</strong>
          {shortest === w.step ? ", the shortest possible." : `. The shortest route is ${shortest}.`}
        </p>
      );
    return (
      <p className="stepline stepline--trap" role="status">
        <strong>{w.result === "loop" ? `Went in circles at move ${w.step}.` : "Ran out of moves."}</strong> The table
        doesn&apos;t know this part of the maze well enough yet. Learn more and try again.
      </p>
    );
  }
  const s = m.lastStep;
  if (!s || m.episode === 0) return <p className="stepline">The agent waits at the start (blue outline). The goal is the gold coin.</p>;
  if (s.done)
    return (
      <p className={`stepline ${s.result === "goal" ? "stepline--win" : "stepline--trap"}`} role="status">
        <strong>{s.result === "goal" ? `Reached the goal in ${s.step} moves.` : `Ran out of moves after ${s.step}.`}</strong>{" "}
        Episode {s.episode} is over.
      </p>
    );
  return (
    <p className="stepline">
      Episode {s.episode}, move {s.step}. Went {ACTIONS[s.action ?? 0]}
      {s.explored ? " (a random move)" : " (its best-known move)"}
      {s.bumped ? " and bumped into a wall" : ""}.
    </p>
  );
}

function UpdateExplainer({ step, alpha, gamma }: { step: StepMsg; alpha: number; gamma: number }) {
  const u = step.update!;
  const reward = step.reward ?? 0;
  const diff = u.new - u.old;
  const verdict =
    Math.abs(diff) < 1e-4
      ? "The estimate was already right, so nothing changed."
      : diff > 0
        ? "Better than expected, so this move's value went up."
        : "Worse than expected, so this move's value went down.";
  return (
    <section className="panel">
      <h2>The last update</h2>
      <p className="panel__note">
        Square row {u.cell[0]}, column {u.cell[1]}, move {ACTIONS[u.action]}.{" "}
        {step.explored ? "A random move, exploring." : "Its best-known move."}
      </p>
      <div className="formula">
        <p className="formula__line">
          <span className="formula__lhs">target</span>
          {u.terminal ? (
            <span>
              = reward = <strong>{num(u.target)}</strong>
              <span className="formula__aside"> The episode ended, so there is no future value.</span>
            </span>
          ) : (
            <span>
              = reward + γ × best value of next square
              <br />= {num(reward, 2)} + {gamma} × {op(u.max_next)} = <strong>{num(u.target)}</strong>
            </span>
          )}
        </p>
        <p className="formula__line">
          <span className="formula__lhs">new Q</span>
          <span>
            = old + α × (target − old)
            <br />= {num(u.old)} + {alpha} × ({num(u.target)} − {op(u.old)}) = <strong>{num(u.new)}</strong>
          </span>
        </p>
      </div>
      <p className="formula__verdict">{verdict}</p>
    </section>
  );
}

function QInspector({ cell, hovering, state: m }: { cell: Cell | null; hovering: boolean; state: MazeState }) {
  if (!cell || !m.maze) return null;
  const s = cell[0] * m.maze.size + cell[1];
  const row = m.q[s] ?? [0, 0, 0, 0];
  const max = Math.max(...row);
  const flat = max - Math.min(...row) < 1e-6;
  return (
    <section className="panel">
      <h2>
        Row {cell[0]}, column {cell[1]}
      </h2>
      <p className="panel__note">
        {hovering ? "The square under your pointer." : "The agent's square. Point at any square to inspect it."}{" "}
        {(m.visits[s] ?? 0) === 0 ? "Never visited yet." : `Visited ${m.visits[s]} times.`}
        {cell[0] === m.maze.goal[0] && cell[1] === m.maze.goal[1] &&
          " This is the goal: the episode ends here, so its own values are never needed and stay at 0."}
      </p>
      <table className="obs qtable">
        <tbody>
          {ACTIONS.map((a, i) => (
            <tr key={a} className={!flat && row[i] === max ? "qtable__best" : ""}>
              <th scope="row">Q(move {a})</th>
              <td className="obs__raw">{num(row[i])}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}

function Settings({
  state: m,
  send,
  connected,
  size,
  setSize,
}: {
  state: MazeState;
  send: (c: MazeCommand) => void;
  connected: boolean;
  size: number;
  setSize: (n: number) => void;
}) {
  const st = m.settings;
  return (
    <section className="panel">
      <h2>Settings</h2>
      <div className="sliders">
        <Slider
          label="Learning rate α"
          hint="How far each update moves the old value toward the target."
          min={0.05}
          max={1}
          step={0.05}
          value={st.alpha}
          onChange={(v) => send({ type: "settings", alpha: v })}
        />
        <Slider
          label="Discount γ"
          hint="How much future reward counts. Lower values make it short-sighted."
          min={0.5}
          max={0.99}
          step={0.01}
          value={st.gamma}
          onChange={(v) => send({ type: "settings", gamma: v })}
        />
        <Slider
          label="Exploration ε"
          hint="Chance of a random move instead of the best-known one."
          min={0}
          max={1}
          step={0.01}
          value={m.epsilon}
          onChange={(v) => send({ type: "settings", epsilon: v })}
        />
        <label className="check">
          <input
            type="checkbox"
            checked={st.decay}
            onChange={(e) => send({ type: "settings", decay: e.target.checked })}
          />
          Lower ε after every episode
        </label>
      </div>

      <div className="maze-pick">
        <label>
          <span>Maze size</span>
          <select value={size} onChange={(e) => setSize(Number(e.target.value))}>
            {SIZES.map((s) => (
              <option key={s} value={s}>
                {s} × {s}
              </option>
            ))}
          </select>
        </label>
        <button
          type="button"
          className="btn btn--quiet"
          disabled={!connected}
          onClick={() => send({ type: "new_maze", size })}
        >
          New maze
        </button>
        <button type="button" className="btn btn--quiet" disabled={!connected} onClick={() => send({ type: "reset" })}>
          Forget everything
        </button>
      </div>
      <p className="panel__note">
        A new maze starts from an empty table. Unlike the PPO agent in Treasure Grid, which learned a rule that works
        on any board, this agent only knows the squares of this maze.
      </p>
    </section>
  );
}

function Slider(p: {
  label: string;
  hint: string;
  min: number;
  max: number;
  step: number;
  value: number;
  onChange: (v: number) => void;
}) {
  return (
    <label className="slider">
      <span className="slider__head">
        <span>{p.label}</span>
        <output>{p.value.toFixed(2)}</output>
      </span>
      <input
        type="range"
        min={p.min}
        max={p.max}
        step={p.step}
        value={p.value}
        onChange={(e) => p.onChange(Number(e.target.value))}
      />
      <span className="slider__hint">{p.hint}</span>
    </label>
  );
}

function MazeCharts({ state: m }: { state: MazeState }) {
  const [hover, setHover] = useState<number | null>(null);
  if (!m.history.length)
    return <p className="charts__empty">Charts of moves and reward per episode appear once the agent finishes an episode.</p>;
  const common = { xMax: Math.max(m.history.length, 10), xLabel: "episode", hover, onHover: setHover };
  return (
    <section className="charts" aria-label="Learning progress charts">
      <div className="charts__grid">
        <LineChart
          {...common}
          title="Moves per episode"
          description="Falls toward the shortest route as the table fills in."
          points={m.history.map((p) => ({ x: p.episode, y: p.steps }))}
          format={(v) => Math.round(v).toString()}
          domain="auto"
          reference={m.maze?.shortest ? { y: m.maze.shortest, label: `Shortest route: ${m.maze.shortest}` } : undefined}
        />
        <LineChart
          {...common}
          title="Reward per episode"
          description="+1 for the goal, −0.01 for every move. Higher means a shorter trip."
          points={m.history.map((p) => ({ x: p.episode, y: p.return }))}
          format={(v) => signed(v)}
          domain="auto"
        />
      </div>
    </section>
  );
}
