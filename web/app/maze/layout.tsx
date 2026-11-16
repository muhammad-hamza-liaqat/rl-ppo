import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Q-learning Maze",
  description: "Watch tabular Q-learning fill in its table of values while it explores a maze.",
};

export default function MazeLayout({ children }: { children: React.ReactNode }) {
  return children;
}
