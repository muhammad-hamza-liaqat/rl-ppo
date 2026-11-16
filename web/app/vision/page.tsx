"use client";

import { useEffect, useRef, useState } from "react";
import { LineChart } from "@/components/LineChart";
import { GrayCanvas, ScanView } from "@/components/ScanView";
import { signed } from "@/components/Thinking";
import { usePreviewPlayer } from "@/lib/usePreviewPlayer";
import {
  type Episode,
  type VisionCommand,
  type VisionFrame,
  type VisionPoint,
  nextEpisode,
  useVisionSocket,
} from "@/lib/vision";

type Mode = "trained" | "random" | "human" | "train";

const TABS: { mode: Mode; label: string }[] = [
  { mode: "trained", label: "Watch the AI" },
  { mode: "random", label: "Watch random moves" },
  { mode: "human", label: "Try it yourself" },
  { mode: "train", label: "Train" },
];

const KEYS: Record<string, number> = {
  ArrowUp: 0,
  ArrowDown: 1,
  ArrowLeft: 2,
  ArrowRight: 3,
  "+": 4,
  "=": 4,
  "-": 5,
  _: 5,
  Enter: 6,
};

const RESULT_TEXT = {
  found: "Found the anomaly",
  wrong: "Said “found it” in the wrong place",
  timeout: "Ran out of steps",
} as const;

const pct = (v: number) => `${Math.round(v * 100)}%`;
const capital = (s: string) => s[0].toUpperCase() + s.slice(1);

export default function VisionPage() {
  const [mode, setMode] = useState<Mode>("trained");
  const [delayMs, setDelayMs] = useState(250);
  const [showTarget, setShowTarget] = useState(false);
  const delayRef = useRef(delayMs);
  delayRef.current = delayMs;

  const v = useVisionSocket((send) => send({ type: "speed", delay_ms: delayRef.current }));
  const { send, status, modelAvailable } = v;
  const connected = status === "open";

  useEffect(() => {
    if (!connected) return;
    v.clearError();
    if (mode === "trained") send(modelAvailable ? { type: "watch", agent: "trained" } : { type: "stop" });
    else if (mode === "random") send({ type: "watch", agent: "random" });
    else if (mode === "human") send({ type: "human_reset" });
    else send({ type: "stop" });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mode, connected, modelAvailable, v.after]);

  useEffect(() => {
    if (connected) send({ type: "speed", delay_ms: delayMs });
  }, [delayMs, connected, send]);

  // live frames for the current mode, or the training preview games
  const preview = usePreviewPlayer(v.preview, delayMs, mode === "train");
  const previewEpisode = preview.frames.reduce<Episode | null>((ep, f) => nextEpisode(ep, f), null);
  const liveOk = v.frame && v.frame.source === mode;
  const frame: VisionFrame | null = mode === "train" ? preview.frame : liveOk ? v.frame : null;
  const episode: Episode | null = mode === "train" ? previewEpisode : liveOk ? v.episode : null;

  // keyboard for "Try it yourself"
  useEffect(() => {
    if (mode !== "human") return;
    const onKey = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLElement && e.target.closest("input, select, textarea")) return;
      if (e.key in KEYS) {
        e.preventDefault();
        if (frame?.done) send({ type: "human_reset" });
        else send({ type: "human_move", action: KEYS[e.key] });
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [mode, frame?.done, send]);

  const trainingLabel = v.training.running && mode !== "train" ? ` (${v.training.update}/${v.training.updates})` : "";

  return (
    <main className="page">
      <header className="masthead">
        <div>
          <h1>Anomaly search</h1>
          <p className="lede">
            Instead of taking in the whole image at once, this agent moves a search box around a scan, step by step,
            until it says &ldquo;found it&rdquo;. The scans are synthetic: soft background, a few normal structures, and
            one bright anomaly.
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

      <nav className="tabs" aria-label="Mode">
        {TABS.map((t) => (
          <button
            key={t.mode}
            type="button"
            className="tabs__tab"
            aria-pressed={mode === t.mode}
            onClick={() => setMode(t.mode)}
          >
            {t.label}
            {t.mode === "train" && trainingLabel}
          </button>
        ))}
      </nav>

      <div className="stage">
        <div className="stage__board">
          <div className="board-frame">
            <ScanView
              image={episode?.image ?? null}
              size={frame?.size ?? 64}
              boxes={episode?.boxes ?? []}
              target={frame?.target ?? null}
              showTarget={showTarget || !!frame?.done}
              result={frame?.done ? frame.result : null}
              moveMs={delayMs}
            />
            {mode === "trained" && !modelAvailable && connected && (
              <div className="board-empty board-empty--scan">
                <p>There&apos;s no trained model yet. Train one first.</p>
                <button type="button" className="btn" onClick={() => setMode("train")}>
                  Go to Train
                </button>
              </div>
            )}
            {mode === "train" && !frame && (
              <div className="board-empty board-empty--scan">
                <p>
                  {v.training.running
                    ? "Waiting for the first update…"
                    : "Start training to watch the agent search one scan after every update."}
                </p>
              </div>
            )}
          </div>

          <VisionStepLine frame={frame} actions={v.actions} previewUpdate={mode === "train" ? preview.update : null} />

          <div className="board-controls">
            {mode === "human" ? (
              <HumanPad
                actions={v.actions}
                disabled={!frame}
                done={!!frame?.done}
                onAction={(a) => send({ type: "human_move", action: a })}
                onReset={() => send({ type: "human_reset" })}
              />
            ) : (
              <label className="speed">
                <span>Move delay</span>
                <input
                  type="range"
                  min={30}
                  max={800}
                  step={10}
                  value={delayMs}
                  onChange={(e) => setDelayMs(Number(e.target.value))}
                />
                <output>{delayMs} ms</output>
              </label>
            )}
            <label className="check">
              <input type="checkbox" checked={showTarget} onChange={(e) => setShowTarget(e.target.checked)} />
              Show where the anomaly is
            </label>
          </div>
        </div>

        <aside className="stage__side">
          {mode === "train" ? (
            <VisionTraining state={v} send={send} connected={connected} />
          ) : (
            <>
              <AgentEyes episode={episode} frame={frame} />
              <ActionOdds frame={frame} actions={v.actions} mode={mode} modelAvailable={modelAvailable} />
              <Overlap frame={frame} hit={v.hitIou} />
            </>
          )}
          {v.error && v.error.code !== "no_model" && (
            <p className="notice notice--error" role="alert">
              {v.error.message}
            </p>
          )}
        </aside>
      </div>

      {mode === "train" && <VisionCharts history={v.history} total={Math.max(v.training.updates, 1)} />}
    </main>
  );
}

// ---------------------------------------------------------------------- //
function VisionStepLine({
  frame,
  actions,
  previewUpdate,
}: {
  frame: VisionFrame | null;
  actions: string[];
  previewUpdate: number | null;
}) {
  const prefix = previewUpdate !== null ? `Update ${previewUpdate}: the current policy searches one scan. ` : "";
  if (!frame) return <p className="stepline">{prefix || " "}</p>;
  if (frame.done && frame.result)
    return (
      <p className={`stepline ${frame.result === "found" ? "stepline--win" : "stepline--trap"}`} role="status">
        {prefix}
        <strong>{RESULT_TEXT[frame.result]}</strong> after {frame.step} steps. Overlap {pct(frame.iou)}, total reward{" "}
        {signed(frame.total)}.
      </p>
    );
  if (frame.action === null)
    return (
      <p className="stepline" role="status">
        {prefix}
        {frame.source === "human"
          ? "New scan. Your box starts small in the middle: move it with the arrow keys, resize with + and −, press Enter when it covers the bright spot."
          : "New scan. The box starts by covering the whole image."}
      </p>
    );
  return (
    <p className="stepline" role="status">
      {prefix}Step {frame.step} of {frame.max_steps}: {actions[frame.action]}, reward {signed(frame.reward ?? 0, 1)}.
    </p>
  );
}

function AgentEyes({ episode, frame }: { episode: Episode | null; frame: VisionFrame | null }) {
  return (
    <section className="panel">
      <h2>What the agent sees</h2>
      <p className="panel__note">
        Two views, like an eye: a blurry 8×8 picture of the whole scan, and a sharp 12×12 close-up of whatever is
        inside its box. To see detail, it has to move the box there.
      </p>
      <div className="eyes">
        <figure className="eyes__view">
          <GrayCanvas pixels={episode?.context ?? null} side={8} className="eyes__canvas" label="Blurry whole-image view" />
          <figcaption>Whole scan, blurry</figcaption>
        </figure>
        <figure className="eyes__view">
          <GrayCanvas pixels={frame?.fovea ?? null} side={12} className="eyes__canvas" label="Sharp view inside the box" />
          <figcaption>Inside the box, sharp</figcaption>
        </figure>
      </div>
    </section>
  );
}

function ActionOdds({
  frame,
  actions,
  mode,
  modelAvailable,
}: {
  frame: VisionFrame | null;
  actions: string[];
  mode: Mode;
  modelAvailable: boolean;
}) {
  const probs = frame?.probs ?? null;
  const best = probs ? probs.indexOf(Math.max(...probs)) : -1;
  const uniform = probs ? Math.max(...probs) - Math.min(...probs) < 0.01 : false;
  const title = mode === "human" ? "What the trained AI would do" : "Next action odds";
  const note = frame?.done
    ? "Episode over. A new scan starts shortly."
    : mode === "random"
      ? `A random agent picks each of the ${actions.length} actions equally often.`
      : mode === "human" && !modelAvailable
        ? "Train a model to see its advice here."
        : "The actor's probabilities. The AI takes the most likely action.";
  return (
    <section className="panel">
      <h2>{title}</h2>
      <p className="panel__note">{note}</p>
      <ul className="odds">
        {actions.map((name, i) => {
          const p = probs?.[i] ?? 0;
          return (
            <li key={name} className={i === best && !uniform ? "odds__row odds__row--best" : "odds__row"}>
              <span className="odds__name odds__name--wide">{capital(name)}</span>
              <span className="odds__track">
                <span className="odds__bar" style={{ width: `${p * 100}%` }} />
              </span>
              <span className="odds__pct">{probs ? `${Math.round(p * 100)}%` : "–"}</span>
            </li>
          );
        })}
      </ul>
    </section>
  );
}

function Overlap({ frame, hit }: { frame: VisionFrame | null; hit: number }) {
  const iou = frame?.iou ?? 0;
  return (
    <section className="panel">
      <h2>Overlap with the anomaly</h2>
      <p className="panel__note">
        IoU: the shared area divided by the combined area of the box and the anomaly. &ldquo;Found it&rdquo; counts when
        it is at least {pct(hit)}. Every move that raises it earns +0.1; every other move costs 0.1.
      </p>
      <div className="iou">
        <span className="iou__track">
          <span className={iou >= hit ? "iou__bar iou__bar--hit" : "iou__bar"} style={{ width: `${iou * 100}%` }} />
          <span className="iou__mark" style={{ left: `${hit * 100}%` }} />
        </span>
        <strong className="iou__value">{frame ? pct(iou) : "–"}</strong>
      </div>
      {frame?.value !== null && frame?.value !== undefined && (
        <p className="panel__note">Critic&apos;s estimate of reward still to come: {signed(frame.value)}</p>
      )}
    </section>
  );
}

function HumanPad({
  actions,
  disabled,
  done,
  onAction,
  onReset,
}: {
  actions: string[];
  disabled: boolean;
  done: boolean;
  onAction: (a: number) => void;
  onReset: () => void;
}) {
  const keys = ["↑", "↓", "←", "→", "+", "−", "Enter"];
  return (
    <div className="pad">
      <div className="pad__grid" role="group" aria-label="Search box controls">
        {actions.map((a, i) => (
          <button
            key={a}
            type="button"
            className={i === 6 ? "btn pad__found" : "pad__btn"}
            disabled={disabled || done}
            onClick={() => onAction(i)}
          >
            {capital(a)} <span className="pad__key">{keys[i]}</span>
          </button>
        ))}
      </div>
      <button type="button" className={done ? "btn" : "btn btn--quiet"} onClick={onReset}>
        New scan
      </button>
    </div>
  );
}

// Measured on an Intel MacBook: 150 updates took about 4½ minutes and reached 87% found.
const LENGTHS = [
  { updates: 60, label: "Short", time: "about 2 minutes" },
  { updates: 150, label: "Standard", time: "about 4½ minutes" },
  { updates: 300, label: "Long", time: "about 9 minutes" },
];

function VisionTraining({
  state: v,
  send,
  connected,
}: {
  state: ReturnType<typeof useVisionSocket>;
  send: (c: VisionCommand) => void;
  connected: boolean;
}) {
  const [updates, setUpdates] = useState(150);
  const t = v.training;
  const last = v.history.at(-1);
  const secs = last ? last.elapsed / last.update : null;
  return (
    <div className="train">
      <section className="panel">
        <h2>Train the searcher with PPO</h2>
        {t.running ? (
          <>
            <p className="panel__note">
              16 scans are searched side by side. After each update the agent searches one new scan on the left, so you
              can see it improve.
            </p>
            <div className="progress" role="progressbar" aria-valuemin={0} aria-valuemax={t.updates} aria-valuenow={t.update} aria-label="Training progress">
              <span className="progress__bar" style={{ width: `${(t.update / Math.max(t.updates, 1)) * 100}%` }} />
            </div>
            <p className="train__status">
              {t.update === 0
                ? "Testing the untrained agent first…"
                : `Update ${t.update} of ${t.updates}. About ${Math.max(1, Math.round(((t.updates - t.update) * (secs ?? 1)) / 60))} min left.`}
            </p>
            <button type="button" className="btn btn--quiet" onClick={() => send({ type: "stop_train" })}>
              Stop training
            </button>
          </>
        ) : (
          <>
            <p className="panel__note">
              Starts from an untrained network. Each update the agent makes 2,048 moves across many scans, then adjusts
              its network.{v.modelAvailable && " When it finishes, the new model replaces the saved one."}
            </p>
            <fieldset className="lengths">
              <legend>How long to train</legend>
              {LENGTHS.map((l) => (
                <label key={l.updates} className="lengths__opt">
                  <input type="radio" name="vision-updates" checked={updates === l.updates} onChange={() => setUpdates(l.updates)} />
                  <span>
                    <strong>{l.label}</strong>: {l.updates} updates, {l.time}
                  </span>
                </label>
              ))}
            </fieldset>
            <button type="button" className="btn" disabled={!connected} onClick={() => send({ type: "train", updates })}>
              Start training
            </button>
          </>
        )}
      </section>

      {v.lastRun?.stopped && !t.running && (
        <p className="notice">Training stopped after {v.lastRun.updatesDone} updates. The saved model was not changed.</p>
      )}

      {(v.before || v.after) && (
        <section className="panel">
          <h2>Before and after</h2>
          <p className="panel__note">The agent searches fresh scans, always taking its most likely action.</p>
          <table className="compare">
            <thead>
              <tr>
                <th scope="col"></th>
                <th scope="col">Found it</th>
                <th scope="col">Wrong place</th>
                <th scope="col">Out of steps</th>
                <th scope="col">Avg steps</th>
              </tr>
            </thead>
            <tbody>
              {v.before && (
                <tr>
                  <th scope="row">Untrained</th>
                  <td>{pct(v.before.found)}</td>
                  <td>{pct(v.before.wrong)}</td>
                  <td>{pct(v.before.timeout)}</td>
                  <td>{v.before.steps.toFixed(1)}</td>
                </tr>
              )}
              <tr>
                <th scope="row">Trained</th>
                {v.after ? (
                  <>
                    <td className="compare__win">{pct(v.after.found)}</td>
                    <td>{pct(v.after.wrong)}</td>
                    <td>{pct(v.after.timeout)}</td>
                    <td>{v.after.steps.toFixed(1)}</td>
                  </>
                ) : (
                  <td colSpan={4} className="compare__pending">
                    {t.running ? "Measured when training finishes" : "Not measured"}
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

function VisionCharts({ history, total }: { history: VisionPoint[]; total: number }) {
  const [hover, setHover] = useState<number | null>(null);
  if (!history.length)
    return <p className="charts__empty">Charts of success rate, reward, entropy and KL appear once training starts.</p>;
  const common = { xMax: total, xLabel: "update", hover, onHover: setHover };
  const series = (f: (p: VisionPoint) => number) => history.map((p) => ({ x: p.update, y: f(p) }));
  return (
    <section className="charts" aria-label="Training progress charts">
      <div className="charts__grid">
        <LineChart
          {...common}
          title="Found the anomaly"
          description="Share of recent searches that ended with a correct “found it”."
          points={series((p) => p.found)}
          format={pct}
          domain={[0, 1]}
        />
        <LineChart
          {...common}
          title="Average reward"
          description="Per search. Rises as it moves toward the anomaly and stops in the right place."
          points={series((p) => p.avg_reward)}
          format={(x) => signed(x, 1)}
          domain="auto"
        />
        <LineChart
          {...common}
          title="Entropy"
          description="How random its actions are. Starts near 1.95 (7 equal choices) and falls."
          points={series((p) => p.entropy)}
          format={(x) => x.toFixed(2)}
          domain={[0, 2]}
        />
        <LineChart
          {...common}
          title="KL divergence"
          description="How much one update changed the policy. PPO keeps this small."
          points={series((p) => p.kl)}
          format={(x) => x.toFixed(3)}
          domain="auto"
        />
      </div>
    </section>
  );
}
