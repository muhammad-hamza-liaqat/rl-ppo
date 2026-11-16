"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Board } from "@/components/Board";
import { CriticValue, MoveOdds, Observation, StepLine } from "@/components/Thinking";
import { TrainingCharts, TrainingControls } from "@/components/TrainingPanel";
import type { Cell, ClientMessage, Frame } from "@/lib/protocol";
import { usePreviewPlayer } from "@/lib/usePreviewPlayer";
import { useTreasureSocket } from "@/lib/useTreasureSocket";

type Mode = "trained" | "random" | "human" | "train";

const TABS: { mode: Mode; label: string }[] = [
  { mode: "trained", label: "Watch the AI" },
  { mode: "random", label: "Watch random moves" },
  { mode: "human", label: "Play yourself" },
  { mode: "train", label: "Train" },
];

const KEYS: Record<string, number> = {
  ArrowUp: 0, w: 0, W: 0,
  ArrowDown: 1, s: 1, S: 1,
  ArrowLeft: 2, a: 2, A: 2,
  ArrowRight: 3, d: 3, D: 3,
};

/** Positions the agent has visited in the current game. */
function useTrail(frame: Frame | null): Cell[] {
  const ref = useRef<{ key: string; cells: Cell[] }>({ key: "", cells: [] });
  return useMemo(() => {
    if (!frame) return [];
    const key = `${frame.source}-${frame.episode}`;
    if (ref.current.key !== key || frame.step === 0) ref.current = { key, cells: [] };
    const cells = ref.current.cells;
    const last = cells.at(-1);
    const a = frame.board.agent;
    if (!last || last[0] !== a[0] || last[1] !== a[1]) cells.push(a);
    return [...cells];
  }, [frame]);
}

export default function Page() {
  const [mode, setMode] = useState<Mode>("trained");
  const [delayMs, setDelayMs] = useState(300);
  const [showAdvice, setShowAdvice] = useState(true);
  const delayRef = useRef(delayMs);
  delayRef.current = delayMs;

  // Tell the server what this tab wants to see.
  const startMode = useCallback((send: (m: ClientMessage) => void, m: Mode) => {
    if (m === "trained") send({ type: "watch", agent: "trained" });
    else if (m === "random") send({ type: "watch", agent: "random" });
    else if (m === "human") send({ type: "human_reset" });
    else send({ type: "stop" });
  }, []);

  // On (re)connect, send the speed first; the effect below then starts the current mode.
  const sock = useTreasureSocket((send) => send({ type: "speed", delay_ms: delayRef.current }));
  const { send, status, modelAvailable } = sock;
  const connected = status === "open";

  // Restart the stream when the mode changes or a newly trained model arrives.
  useEffect(() => {
    if (!connected) return;
    sock.clearError();
    if (mode === "trained" && !modelAvailable) send({ type: "stop" });
    else startMode(send, mode);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mode, connected, modelAvailable, sock.after]);

  useEffect(() => {
    if (connected) send({ type: "speed", delay_ms: delayMs });
  }, [delayMs, connected, send]);

  // Only show frames that belong to the current mode (old ones can still be in flight).
  const live = sock.frame && sock.frame.source === mode ? sock.frame : null;
  const preview = usePreviewPlayer(sock.preview, delayMs, mode === "train");
  const frame = mode === "train" ? preview.frame : live;
  const liveTrail = useTrail(live);
  const trail = mode === "train" ? preview.frames.map((f) => f.board.agent) : liveTrail;

  // Keyboard controls for "Play yourself".
  useEffect(() => {
    if (mode !== "human") return;
    const onKey = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLElement && e.target.closest("input, textarea, select")) return;
      if (e.key in KEYS) {
        e.preventDefault();
        if (live?.done) send({ type: "human_reset" });
        else send({ type: "human_move", action: KEYS[e.key] });
      } else if ((e.key === "Enter" || e.key === " ") && live?.done) {
        e.preventDefault();
        send({ type: "human_reset" });
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [mode, live?.done, send]);

  const trainingLabel =
    sock.training.running && mode !== "train"
      ? ` (${sock.training.update}/${sock.training.updates})`
      : "";

  return (
    <main className="page">
      <header className="masthead">
        <div>
          <h1>Treasure Grid</h1>
          <p className="lede">
            An agent learns by trial and error to reach the treasure and stay off the traps. Its brain is a
            small neural network trained with PPO.
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
          <pre>source env/bin/activate{"\n"}uvicorn server:app --port 8000</pre>
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
            <Board
              frame={frame}
              trail={trail}
              moveMs={delayMs}
              showArrows={mode !== "human" || showAdvice}
            />
            {mode === "trained" && !modelAvailable && connected && (
              <div className="board-empty">
                <p>There&apos;s no trained model yet. Train one first; it takes about 3 minutes.</p>
                <button type="button" className="btn" onClick={() => setMode("train")}>
                  Go to Train
                </button>
              </div>
            )}
            {mode === "train" && !frame && (
              <div className="board-empty">
                <p>
                  {sock.training.running
                    ? "Waiting for the first update…"
                    : "Start training to watch the agent play one game after every update."}
                </p>
              </div>
            )}
          </div>

          {mode === "train" && preview.update !== null ? (
            <p className="stepline">
              Update {preview.update}: the current policy plays one game.{" "}
              {frame?.done && frame.result === "win" ? "It found the treasure." : ""}
              {frame?.done && frame.result === "trap" ? "It hit a trap." : ""}
              {frame?.done && frame.result === "timeout" ? "It ran out of moves." : ""}
            </p>
          ) : (
            <StepLine frame={frame} />
          )}

          <div className="board-controls">
            {mode === "human" ? (
              <>
                <DPad
                  disabled={!live}
                  done={!!live?.done}
                  onMove={(a) => send({ type: "human_move", action: a })}
                  onReset={() => send({ type: "human_reset" })}
                />
                <label className="check">
                  <input
                    type="checkbox"
                    checked={showAdvice}
                    onChange={(e) => setShowAdvice(e.target.checked)}
                    disabled={!modelAvailable}
                  />
                  Show the trained AI&apos;s advice
                </label>
              </>
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
          </div>
        </div>

        <aside className="stage__side">
          {mode === "train" ? (
            <TrainingControls
              training={sock.training}
              history={sock.history}
              before={sock.before}
              after={sock.after}
              lastRun={sock.lastRun}
              modelAvailable={modelAvailable}
              connected={connected}
              rolloutSteps={sock.rolloutSteps}
              send={send}
            />
          ) : (
            <>
              <MoveOdds frame={frame} {...oddsCopy(mode, modelAvailable)} />
              {mode !== "random" && <CriticValue value={frame?.value ?? null} />}
              <Observation frame={frame} />
            </>
          )}
          {sock.error && sock.error.code !== "no_model" && (
            <p className="notice notice--error" role="alert">
              {sock.error.message}
            </p>
          )}
        </aside>
      </div>

      {mode === "train" && (
        <TrainingCharts history={sock.history} total={Math.max(sock.training.updates, 1)} />
      )}
    </main>
  );
}

function oddsCopy(mode: Mode, modelAvailable: boolean) {
  if (mode === "random")
    return { title: "Next move odds", note: "A random agent picks each direction 25% of the time." };
  if (mode === "human")
    return modelAvailable
      ? { title: "What the trained AI would do", note: "Its move probabilities for your current square." }
      : { title: "What the trained AI would do", note: "Train a model to see its advice here." };
  return { title: "Next move odds", note: "The actor's probabilities. The AI takes the most likely move." };
}

function DPad({
  disabled,
  done,
  onMove,
  onReset,
}: {
  disabled: boolean;
  done: boolean;
  onMove: (a: number) => void;
  onReset: () => void;
}) {
  return (
    <div className="dpad-wrap">
      <div className="dpad" role="group" aria-label="Move">
        <button type="button" className="dpad__up" disabled={disabled || done} onClick={() => onMove(0)} aria-label="Up">
          ↑
        </button>
        <button type="button" className="dpad__left" disabled={disabled || done} onClick={() => onMove(2)} aria-label="Left">
          ←
        </button>
        <button type="button" className="dpad__down" disabled={disabled || done} onClick={() => onMove(1)} aria-label="Down">
          ↓
        </button>
        <button type="button" className="dpad__right" disabled={disabled || done} onClick={() => onMove(3)} aria-label="Right">
          →
        </button>
      </div>
      <div className="dpad-help">
        <p>Use the arrow keys or W A S D.</p>
        <button type="button" className={done ? "btn" : "btn btn--quiet"} onClick={onReset}>
          New board
        </button>
      </div>
    </div>
  );
}
