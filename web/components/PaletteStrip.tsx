import type { Palette } from "@/lib/rlhf";

/** Five colour blocks in a row. */
export function PaletteStrip({ palette, size = "m" }: { palette: Palette; size?: "s" | "m" | "l" }) {
  return (
    <span className={`strip strip--${size}`} aria-label={`Palette: ${palette.colors.join(", ")}`} role="img">
      {palette.colors.map((c, i) => (
        <span key={i} className="strip__chip" style={{ background: c }} />
      ))}
    </span>
  );
}
