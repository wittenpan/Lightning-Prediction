import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({ variable: "--font-geist-sans", subsets: ["latin"] });
const geistMono = Geist_Mono({ variable: "--font-geist-mono", subsets: ["latin"] });

export const metadata: Metadata = {
  metadataBase: new URL("http://localhost:3000"),
  title: "StormSignal · Geospatial Prediction System",
  description: "A local H3 operations console for Kafka-streamed lightning prediction.",
  openGraph: {
    title: "StormSignal · Geospatial Prediction System",
    description: "Live H3 risk intelligence powered by MSK, Redis, and a gated XGBoost cascade.",
    images: [{ url: "/og.png", width: 1200, height: 630 }],
  },
  twitter: {
    card: "summary_large_image",
    title: "StormSignal · Geospatial Prediction System",
    description: "Live H3 risk intelligence powered by MSK, Redis, and a gated XGBoost cascade.",
    images: ["/og.png"],
  },
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body className={`${geistSans.variable} ${geistMono.variable}`}>{children}</body></html>;
}
