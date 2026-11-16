"use client";

import { useEffect, useState } from "react";
import { LineChart } from "@/components/LineChart";
import { PaletteStrip } from "@/components/PaletteStrip";
import { signed } from "@/components/Thinking";
import { type RLHFCommand, type RLHFState, useRLHFSocket } from "@/lib/rlhf";

const LENGTHS = [
  { updates: 15, label: "Short", time: "about 15 seconds" },
  { updates: 30, label: "Standard", time: "about 30 seconds" },
  { updates: 60, label: "Long", time: "about a minute" },
];

export default function RLHFPage() {
  const r = useRLHFSocket();
  const { send, status } = r;
  const connected = status === "open";
  const busy = r.tuning.running;
  const picks = r.rm?.picks ?? 0;

  // Keyboard: ← left palette, → right palette, S no preference.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (busy || !r.pair) return;
      if (e.target instanceof HTMLElement && e.target.closest("input, select, textarea")) return;
      const pick = e.key === "ArrowLeft" ? "a" : e.key === "ArrowRight" ? "b" : e.key === "s" || e.key === "S" ? "skip" : null;
      if (pick) {
        e.preventDefault();
        send({ type: "choose", pick });
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [busy, r.pair, send]);

  return (
    <main className="page">
      <header className="masthead">
        <div>
          <h1>Teach it your taste</h1>
          <p className="lede">
            A small generator makes colour palettes. You pick the one you like more, a reward model learns your taste
            from those picks, then PPO tunes the generator to please the reward model. Chat assistants are fine-tuned
            with this same loop, called RLHF.
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

      <Compare state={r} send={send} connected={connected} />

      <div className="rlhf-grid">
        <RewardModelPanel state={r} />
        <TunePanel state={r} send={send} connected={connected} />
      </div>

      <TuneCharts state={r} />
      <BeforeAfter state={r} send={send} connected={connected} />

      <p className="rlhf-reset">
        <button
          type="button"
          className="btn btn--quiet"
          disabled={!connected}
          onClick={() => send({ type: "reset" })}
        >
          Start over
        </button>{" "}
        Clears your picks, the reward model and the tuned generator. {picks > 0 && `You have ${picks} picks.`}
      </p>
    </main>
  );
}

// ---------------------------------------------------------------------- //
function Compare({ state: r, send, connected }: { state: RLHFState; send: (c: RLHFCommand) => void; connected: boolean }) {
  const busy = r.tuning.running;
  const picks = r.rm?.picks ?? 0;
  const disabled = !connected || busy || !r.pair;
  return (
    <section className="compare-step" aria-labelledby="step1">
      <h2 id="step1" className="step-title">
        1. Pick the palette you like more
      </h2>
      <p className="panel__note">
        {busy
          ? "Picking is paused while PPO tunes the generator."
          : r.pair?.source === "tuned"
            ? `These come from the tuned generator (${r.tunedUpdates} updates). More picks help the reward model fix its mistakes.`
            : "These come from the original generator, which picks colours at random."}{" "}
        Use the arrow keys, or S for no preference.
      </p>

      <div className="duel">
        {(["a", "b"] as const).map((side) => (
          <button
            key={side}
            type="button"
            className="duel__option"
            disabled={disabled}
            onClick={() => send({ type: "choose", pick: side })}
          >
            {r.pair ? <PaletteStrip palette={r.pair[side]} size="l" /> : <span className="strip strip--l strip--empty" />}
            <span className="duel__label">{side === "a" ? "I prefer the left one (←)" : "I prefer the right one (→)"}</span>
          </button>
        ))}
      </div>

      <div className="duel__bar">
        <button type="button" className="btn btn--quiet" disabled={disabled} onClick={() => send({ type: "choose", pick: "skip" })}>
          No preference (S)
        </button>
        <span className="duel__count">
          {picks === 0
            ? `No picks yet. Tuning unlocks after ${r.minPicks}.`
            : picks < r.minPicks
              ? `${picks} picks. ${r.minPicks - picks} more to unlock tuning.`
              : `${picks} picks.`}
        </span>
        <button
          type="button"
          className="btn btn--quiet duel__sim"
          disabled={!connected || busy}
          onClick={() => send({ type: "simulate", n: 20 })}
          title={`A pretend person who ${r.simulatedDescription}.`}
        >
          Answer 20 for me
        </button>
      </div>
      {r.simulatedDescription && (
        <p className="duel__simnote">
          &ldquo;Answer 20 for me&rdquo; uses a pretend person who {r.simulatedDescription}. Use it to check whether the
          reward model can recover a taste you know.
        </p>
      )}
      {r.error && (
        <p className="notice notice--error" role="alert">
          {r.error}
        </p>
      )}
    </section>
  );
}

function RewardModelPanel({ state: r }: { state: RLHFState }) {
  const rm = r.rm;
  const w = rm?.weights ?? r.features.map(() => 0);
  const scale = Math.max(1, ...w.map(Math.abs));
  const hidden = rm?.hidden_taste ?? null;
  const hiddenScale = hidden ? Math.max(...hidden.map(Math.abs)) : 1;
  const learnedMax = Math.max(1e-6, ...w.map(Math.abs));

  return (
    <section className="panel" aria-labelledby="step2">
      <h2 id="step2" className="step-title">
        2. What the reward model learned
      </h2>
      <p className="panel__note">
        It scores a palette as a weighted sum of these 8 traits, fitted so that the palettes you picked score higher.
        Real reward models are large neural networks; this one is small enough to read.
      </p>

      {rm && rm.picks > 0 ? (
        <p className="rm__agree">
          It agrees with <strong>{Math.round((rm.agreement ?? 0) * 100)}%</strong> of your {rm.picks} picks.
        </p>
      ) : (
        <p className="rm__agree">Make a pick and the weights appear.</p>
      )}

      <ul className="weights">
        {r.features.map((f, i) => {
          const v = w[i] ?? 0;
          const pct = (Math.abs(v) / scale) * 50;
          const h = hidden ? (hidden[i] / hiddenScale) * (learnedMax / scale) * 50 : null;
          return (
            <li key={f.key} className="weights__row" title={f.description}>
              <span className="weights__label">
                {f.label}
                <span className="weights__desc">{f.description}</span>
              </span>
              <span className="weights__track">
                <span className="weights__zero" />
                <span
                  className={v >= 0 ? "weights__bar weights__bar--pos" : "weights__bar weights__bar--neg"}
                  style={v >= 0 ? { left: "50%", width: `${pct}%` } : { right: "50%", width: `${pct}%` }}
                />
                {h !== null && <span className="weights__hidden" style={{ left: `${50 + h}%` }} />}
              </span>
              <span className="weights__value">{signed(v)}</span>
            </li>
          );
        })}
      </ul>
      <p className="weights__legend">
        Bars to the right mean you like more of that trait; bars to the left mean less.
        {hidden && " The thin dark marks show the pretend person's hidden taste, scaled to match."}
      </p>

      {rm?.extremes && (
        <div className="extremes">
          <div>
            <span className="extremes__label">Its favourite of 400 random palettes</span>
            <PaletteStrip palette={rm.extremes.best} size="m" />
          </div>
          <div>
            <span className="extremes__label">Its least favourite</span>
            <PaletteStrip palette={rm.extremes.worst} size="m" />
          </div>
        </div>
      )}
    </section>
  );
}

function TunePanel({ state: r, send, connected }: { state: RLHFState; send: (c: RLHFCommand) => void; connected: boolean }) {
  const [beta, setBeta] = useState(0.1);
  const [updates, setUpdates] = useState(30);
  const picks = r.rm?.picks ?? 0;
  const ready = picks >= r.minPicks;
  const t = r.tuning;

  return (
    <section className="panel" aria-labelledby="step3">
      <h2 id="step3" className="step-title">
        3. Tune the generator with PPO
      </h2>
      <p className="panel__note">
        The generator picks 5 colours one at a time and gets the reward model&apos;s score at the end. PPO nudges it
        toward palettes that score well.
      </p>

      {t.running ? (
        <>
          <div className="progress" role="progressbar" aria-valuemin={0} aria-valuemax={t.updates} aria-valuenow={t.done} aria-label="Tuning progress">
            <span className="progress__bar" style={{ width: `${(t.done / Math.max(t.updates, 1)) * 100}%` }} />
          </div>
          <p className="train__status">
            {t.done === 0 ? "Starting…" : `Update ${t.done} of ${t.updates}. Some palettes it is making right now:`}
          </p>
          <div className="samples">
            {r.samples.map((p, i) => (
              <PaletteStrip key={i} palette={p} size="s" />
            ))}
          </div>
          <button type="button" className="btn btn--quiet" onClick={() => send({ type: "stop_tune" })}>
            Stop tuning
          </button>
        </>
      ) : (
        <>
          <label className="slider">
            <span className="slider__head">
              <span>Stay close to the original (KL penalty β)</span>
              <output>{beta.toFixed(2)}</output>
            </span>
            <input type="range" min={0} max={0.5} step={0.01} value={beta} onChange={(e) => setBeta(Number(e.target.value))} />
            <span className="slider__hint">
              At 0 it chases the reward model freely and often collapses onto a few palettes the reward model overrates.
              Higher keeps it varied and close to how it started.
            </span>
          </label>
          <fieldset className="lengths">
            <legend>How long to tune</legend>
            {LENGTHS.map((l) => (
              <label key={l.updates} className="lengths__opt">
                <input type="radio" name="tune-updates" checked={updates === l.updates} onChange={() => setUpdates(l.updates)} />
                <span>
                  <strong>{l.label}</strong>: {l.updates} updates, {l.time}
                </span>
              </label>
            ))}
          </fieldset>
          <button
            type="button"
            className="btn"
            disabled={!connected || !ready}
            onClick={() => send({ type: "tune", updates, beta })}
          >
            {r.tunedUpdates ? "Tune more" : "Start tuning"}
          </button>
          {!ready && <p className="panel__note">Make {r.minPicks - picks} more picks first.</p>}
          {r.lastRun?.stopped && <p className="panel__note">Tuning stopped early. The generator keeps what it learned so far.</p>}
        </>
      )}
    </section>
  );
}

function TuneCharts({ state: r }: { state: RLHFState }) {
  const [hover, setHover] = useState<number | null>(null);
  if (!r.history.length) return null;
  const common = {
    xMax: Math.max(r.history.at(-1)!.update, r.tuning.running ? r.history.at(-1)!.update - r.tuning.done + r.tuning.updates : 0, 2),
    xLabel: "update",
    hover,
    onHover: setHover,
  };
  return (
    <section className="charts" aria-label="Tuning charts">
      <div className="charts__grid charts__grid--3">
        <LineChart
          {...common}
          title="Reward model score"
          description="0 is an average random palette. Rises as it learns what you like."
          points={r.history.map((p) => ({ x: p.update, y: p.score }))}
          format={(v) => signed(v)}
          domain="auto"
        />
        <LineChart
          {...common}
          title="Distance from the original (KL)"
          description="How far it has drifted from the generator it started as."
          points={r.history.map((p) => ({ x: p.update, y: p.kl }))}
          format={(v) => v.toFixed(1)}
          domain="auto"
        />
        <LineChart
          {...common}
          title="Variety"
          description="Share of distinct palettes in each batch. A drop means it is collapsing."
          points={r.history.map((p) => ({ x: p.update, y: p.unique }))}
          format={(v) => `${Math.round(v * 100)}%`}
          domain={[0, 1]}
        />
      </div>
    </section>
  );
}

function BeforeAfter({ state: r, send, connected }: { state: RLHFState; send: (c: RLHFCommand) => void; connected: boolean }) {
  const g = r.gallery;
  if (!g) return null;
  const avg = (v: number | null) => (v === null ? "" : `Average score ${signed(v)}`);
  return (
    <section className="before-after" aria-labelledby="results">
      <div className="before-after__head">
        <h2 id="results" className="step-title">
          Original vs tuned
        </h2>
        <button type="button" className="btn btn--quiet" disabled={!connected || r.tuning.running} onClick={() => send({ type: "gallery" })}>
          Draw new samples
        </button>
      </div>
      <div className="gallery-row">
        <p className="gallery-row__title">
          Original generator <span className="gallery-row__avg">{avg(g.original_avg)}</span>
        </p>
        <div className="gallery">
          {g.original.map((p, i) => (
            <PaletteStrip key={i} palette={p} size="m" />
          ))}
        </div>
      </div>
      <div className="gallery-row">
        <p className="gallery-row__title">
          Tuned generator <span className="gallery-row__avg">{avg(g.tuned_avg)}</span>
        </p>
        {g.tuned.length ? (
          <div className="gallery">
            {g.tuned.map((p, i) => (
              <PaletteStrip key={i} palette={p} size="m" />
            ))}
          </div>
        ) : (
          <p className="panel__note">Not tuned yet. Make some picks, then start tuning.</p>
        )}
      </div>
      {g.tuned.length > 0 && <HackingNote state={r} />}
    </section>
  );
}

/** Explains reward hacking; with the simulated rater we know exactly which traits the reward model got wrong. */
function HackingNote({ state: r }: { state: RLHFState }) {
  const rm = r.rm;
  const hidden = rm?.hidden_taste;
  const wrong =
    rm && hidden
      ? r.features.filter((_, i) => Math.abs(hidden[i]) > 0.3 && Math.sign(hidden[i]) !== Math.sign(rm.weights[i]))
      : [];

  return (
    <div className="notice hacking">
      {wrong.length > 0 ? (
        <p>
          <strong>Reward hacking.</strong> The reward model has the wrong direction for{" "}
          {wrong.map((f) => f.label.toLowerCase()).join(" and ")}, so PPO pushed the generator toward exactly that. It
          scores well with the reward model but not with the pretend person.
        </p>
      ) : (
        <p>
          <strong>Watch for reward hacking.</strong> PPO optimises the reward model, not you. If the tuned palettes lean
          hard on something you don&apos;t actually like, the reward model learned it wrong and PPO exploited it.
        </p>
      )}
      <p>
        To fix it, do what real RLHF does: make more picks. New pairs now come from the tuned generator, so the reward
        model sees its mistakes. Then tune again.
      </p>
    </div>
  );
}
