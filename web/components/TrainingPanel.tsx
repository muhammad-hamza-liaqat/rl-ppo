"use client";

import { useState } from "react";
import type { ClientMessage, EvalResult, TrainPoint, TrainState } from "@/lib/protocol";
import { LineChart } from "./LineChart";
import { signed } from "./Thinking";

const pct = (v: number) => `${Math.round(v * 100)}%`;
const LENGTHS = [
  { updates: 50, label: "Short", time: "about 1 minute" },
  { updates: 150, label: "Standard", time: "about 3 minutes" },
  { updates: 300, label: "Long", time: "about 6 minutes" },
];

interface ControlsProps {
  training: TrainState;
  history: TrainPoint[];
  before: EvalResult | null;
  after: EvalResult | null;
  lastRun: { stopped: boolean; updatesDone: number } | null;
  modelAvailable: boolean;
  connected: boolean;
  rolloutSteps: number;
  send: (m: ClientMessage) => void;
}

export function TrainingControls(p: ControlsProps) {
  const [updates, setUpdates] = useState(150);
  const { running, update, updates: total } = p.training;
  const last = p.history.at(-1);

  return (
    <div className="train">
      <section className="panel">
        <h2>Train the agent with PPO</h2>
        {running ? (
          <>
            <p className="panel__note">
              Each update, the agent plays {p.rolloutSteps.toLocaleString()} moves, scores every move, then
              adjusts its network. After each update it plays one game on the board so you can see it improve.
            </p>
            <div
              className="progress"
              role="progressbar"
              aria-valuemin={0}
              aria-valuemax={total}
              aria-valuenow={update}
              aria-label="Training progress"
            >
              <span className="progress__bar" style={{ width: `${(update / Math.max(total, 1)) * 100}%` }} />
            </div>
            <p className="train__status">
              {update === 0 ? (
                "Testing the untrained agent first…"
              ) : (
                <>
                  Update {update} of {total}. {last ? `${last.steps.toLocaleString()} moves played in ${Math.round(last.elapsed)} s.` : ""}
                </>
              )}
            </p>
            <button type="button" className="btn btn--quiet" onClick={() => p.send({ type: "stop_train" })}>
              Stop training
            </button>
          </>
        ) : (
          <>
            <p className="panel__note">
              Training starts from a fresh, untrained network. Each update, the agent plays{" "}
              {p.rolloutSteps.toLocaleString()} moves and then adjusts its network.
              {p.modelAvailable && " When it finishes, the new model replaces the saved one."}
            </p>
            <fieldset className="lengths">
              <legend>How long to train</legend>
              {LENGTHS.map((l) => (
                <label key={l.updates} className="lengths__opt">
                  <input
                    type="radio"
                    name="updates"
                    value={l.updates}
                    checked={updates === l.updates}
                    onChange={() => setUpdates(l.updates)}
                  />
                  <span>
                    <strong>{l.label}</strong>: {l.updates} updates, {l.time}
                  </span>
                </label>
              ))}
            </fieldset>
            <button
              type="button"
              className="btn"
              disabled={!p.connected}
              onClick={() => p.send({ type: "train", updates })}
            >
              Start training
            </button>
          </>
        )}
      </section>

      {p.lastRun?.stopped && !running && (
        <p className="notice">
          Training stopped after {p.lastRun.updatesDone} updates. The saved model was not changed.
        </p>
      )}

      {(p.before || p.after) && (
        <section className="panel">
          <h2>Before and after</h2>
          <p className="panel__note">The agent plays fresh boards, always taking its most likely move.</p>
          <table className="compare">
            <thead>
              <tr>
                <th scope="col"></th>
                <th scope="col">Found treasure</th>
                <th scope="col">Hit a trap</th>
                <th scope="col">Ran out of moves</th>
              </tr>
            </thead>
            <tbody>
              {p.before && (
                <tr>
                  <th scope="row">Untrained</th>
                  <td>{pct(p.before.win)}</td>
                  <td>{pct(p.before.trap)}</td>
                  <td>{pct(p.before.timeout)}</td>
                </tr>
              )}
              <tr>
                <th scope="row">Trained</th>
                {p.after ? (
                  <>
                    <td className="compare__win">{pct(p.after.win)}</td>
                    <td>{pct(p.after.trap)}</td>
                    <td>{pct(p.after.timeout)}</td>
                  </>
                ) : (
                  <td colSpan={3} className="compare__pending">
                    {running ? "Measured when training finishes" : "Not measured"}
                  </td>
                )}
              </tr>
            </tbody>
          </table>
        </section>
      )}
    </div>
  );
}

export function TrainingCharts({ history, total }: { history: TrainPoint[]; total: number }) {
  const [hover, setHover] = useState<number | null>(null);
  const common = { xMax: total, xLabel: "update", hover, onHover: setHover };
  const series = (f: (pt: TrainPoint) => number) => history.map((pt) => ({ x: pt.update, y: f(pt) }));

  if (!history.length) {
    return (
      <p className="charts__empty">
        Charts of reward, win rate, entropy and KL appear here once training starts.
      </p>
    );
  }

  const rows = history.filter((pt, i) => pt.update % 10 === 0 || i === 0 || i === history.length - 1);

  return (
    <section className="charts" aria-label="Training progress charts">
      <div className="charts__grid">
        <LineChart
          {...common}
          title="Average reward"
          description="Over the last 200 games. Climbs toward +1 as it learns."
          points={series((pt) => pt.avg_reward)}
          format={(v) => signed(v)}
          domain={[-1, 1]}
        />
        <LineChart
          {...common}
          title="Win rate"
          description="Share of the last 200 games that reached the treasure."
          points={series((pt) => pt.win)}
          format={pct}
          domain={[0, 1]}
        />
        <LineChart
          {...common}
          title="Entropy"
          description="How random its moves are. Starts near 1.39 and falls as it gets confident."
          points={series((pt) => pt.entropy)}
          format={(v) => v.toFixed(2)}
          domain={[0, 1.4]}
        />
        <LineChart
          {...common}
          title="KL divergence"
          description="How much one update changed the policy. PPO keeps this small, around 0.02 or less."
          points={series((pt) => pt.kl)}
          format={(v) => v.toFixed(3)}
          domain="auto"
        />
      </div>
      <details className="charts__table">
        <summary>Show the numbers as a table</summary>
        <table>
          <thead>
            <tr>
              <th scope="col">Update</th>
              <th scope="col">Moves played</th>
              <th scope="col">Average reward</th>
              <th scope="col">Win rate</th>
              <th scope="col">Trap rate</th>
              <th scope="col">Entropy</th>
              <th scope="col">KL</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((pt) => (
              <tr key={pt.update}>
                <td>{pt.update}</td>
                <td>{pt.steps.toLocaleString()}</td>
                <td>{signed(pt.avg_reward)}</td>
                <td>{pct(pt.win)}</td>
                <td>{pct(pt.trap)}</td>
                <td>{pt.entropy.toFixed(2)}</td>
                <td>{pt.kl.toFixed(4)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </details>
    </section>
  );
}
