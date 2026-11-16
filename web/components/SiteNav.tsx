"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const LINKS = [
  { href: "/", label: "Treasure Grid", algo: "PPO" },
  { href: "/maze", label: "Maze", algo: "Q-learning" },
  { href: "/rlhf", label: "Palette taste", algo: "RLHF" },
  { href: "/vision", label: "Anomaly search", algo: "Vision" },
];

export function SiteNav() {
  const path = usePathname();
  return (
    <nav className="sitenav" aria-label="Demos">
      <div className="sitenav__inner">
        {LINKS.map((l) => (
          <Link
            key={l.href}
            href={l.href}
            className="sitenav__link"
            aria-current={path === l.href ? "page" : undefined}
          >
            {l.label}
            <span className="sitenav__algo">{l.algo}</span>
          </Link>
        ))}
      </div>
    </nav>
  );
}
