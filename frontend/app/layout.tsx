import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "HIT-UAV Thermal Detection",
  description: "Real-time YOLOv8 detection of people and vehicles in UAV infrared thermal imagery",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="bg-slate-950 text-slate-100 antialiased" suppressHydrationWarning>{children}</body>
    </html>
  );
}
