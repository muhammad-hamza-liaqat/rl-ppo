import type { Frame } from "@/lib/protocol";
import { ACTIONS } from "@/lib/protocol";

const OBS_LABELS = [
  "Agent row",
  "Agent column",
  "Treasure row",
  "Treasure column",
  "Trap above",
  "Trap below",
  "Trap to the left",
  "Trap to the right",
];

function capital(s: string) {
  return s[0].toUpperCase() + s.slice(1);
}

export function signed(n: number, digits = 2) {
  const s = Math.abs(n).toFixed(digits);
  return n < 0 ? `−${s}` : `+${s}`;
}

/** The agent's move probabilities as four bars. */
export function MoveOdds({ frame, title, note }: { frame: Frame | null; title: string; note?: string }) {
  const probs = frame?.probs ?? null;
  const best = probs ? probs.indexOf(Math.max(...probs)) : -1;
  const uniform = probs ? Math.max(...probs) - Math.min(...probs) < 0.01 : false;

  return (
    <section className="panel">
      <h2>{title}</h2>
      {frame?.done ? (
        <p className="panel__note">Game over, so there&apos;s no next move. A new board starts shortly.</p>
      ) : (
        note && <p className="panel__note">{note}</p>
      )}
      <ul className="odds">
        {ACTIONS.map((name, i) => {
          const p = probs?.[i] ?? 0;
          return (
            <li key={name} className={i === best && !uniform ? "odds__row odds__row--best" : "odds__row"}>
              <span className="odds__name">{capital(name)}</span>
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

/** The critic's value estimate on a −1 … +1 scale. */
export function CriticValue({ value }: { value: number | null }) {
  const v = value ?? 0;
  const clamped = Math.max(-1, Math.min(1, v));
  const left = 50 + Math.min(0, clamped) * 50;
  const width = Math.abs(clamped) * 50;
  return (
    <section className="panel">
      <h2>How good the critic thinks this spot is</h2>
      <p className="panel__note">
        Expected reward from here: +1 means it expects to reach the treasure, −1 means it expects a trap.
      </p>
      <div className="critic">
        <span className="critic__end">−1</span>
        <span className="critic__track">
          <span className="critic__zero" />
          {value !== null && (
            <span
              className={v >= 0 ? "critic__bar critic__bar--pos" : "critic__bar critic__bar--neg"}
              style={{ left: `${left}%`, width: `${width}%` }}
            />
          )}
        </span>
        <span className="critic__end">+1</span>
        <strong className="critic__value">{value === null ? "–" : signed(value)}</strong>
      </div>
    </section>
  );
}

/** The 8 numbers the network actually receives. */
export function Observation({ frame }: { frame: Frame | null }) {
  const n = (frame?.board.size ?? 6) - 1;
  return (
    <section className="panel">
      <h2>What the agent sees</h2>
      <p className="panel__note">
        The network never sees the picture, only these 8 numbers. Positions are divided by {n} so they
        fall between 0 and 1.
      </p>
      <table className="obs">
        <tbody>
          {OBS_LABELS.map((label, i) => {
            const x = frame?.obs[i];
            let meaning = "";
            if (x !== undefined) {
              meaning = i < 4 ? `= ${Math.round(x * n)}` : x > 0.5 ? "yes" : "no";
            }
            return (
              <tr key={label} className={i >= 4 && x !== undefined && x > 0.5 ? "obs__row--hot" : ""}>
                <th scope="row">{label}</th>
                <td className="obs__raw">{x === undefined ? "–" : x.toFixed(2)}</td>
                <td className="obs__meaning">{meaning}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </section>
  );
}

const RESULT_TEXT = {
  win: "Found the treasure",
  trap: "Stepped on a trap",
  timeout: "Ran out of moves",
} as const;

/** One line under the board saying what just happened. */
export function StepLine({ frame }: { frame: Frame | null }) {
  if (!frame) return <p className="stepline">&nbsp;</p>;
  if (frame.done && frame.result) {
    return (
      <p className={`stepline stepline--${frame.result}`} role="status">
        <strong>{RESULT_TEXT[frame.result]}</strong> after {frame.step} moves. Total reward{" "}
        {signed(frame.total)}.
      </p>
    );
  }
  if (frame.action === null) {
    return (
      <p className="stepline" role="status">
        New board. Game {frame.episode}, move 0 of {frame.max_steps}.
      </p>
    );
  }
  return (
    <p className="stepline" role="status">
      Moved {ACTIONS[frame.action]}, reward {signed(frame.reward ?? 0)}. Total {signed(frame.total)} after{" "}
      {frame.step} of {frame.max_steps} moves.
    </p>
  );
}
