import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Palette Taste",
  description: "RLHF in miniature: rate colour palettes, train a reward model, tune a generator with PPO.",
};

export default function RLHFLayout({ children }: { children: React.ReactNode }) {
  return children;
}
