import type { Metadata, Viewport } from "next";
import { Schibsted_Grotesk } from "next/font/google";
import { SiteNav } from "@/components/SiteNav";
import "./globals.css";

const sans = Schibsted_Grotesk({ subsets: ["latin"], variable: "--font-sans", display: "swap" });

export const metadata: Metadata = {
  title: "Treasure Grid",
  description: "Watch a PPO agent learn to find treasure and avoid traps, live.",
};

export const viewport: Viewport = {
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "#e8edf2" },
    { media: "(prefers-color-scheme: dark)", color: "#111d2b" },
  ],
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={sans.variable}>
      <body>
        <SiteNav />
        {children}
      </body>
    </html>
  );
}
