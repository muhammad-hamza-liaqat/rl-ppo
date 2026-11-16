import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Anomaly Search",
  description: "RL for image processing: an agent moves a search box over a scan until it finds the anomaly.",
};

export default function VisionLayout({ children }: { children: React.ReactNode }) {
  return children;
}
