import "./globals.css";
import Navbar from "../components/Navbar";
import { ReactNode } from "react";

export const metadata = {
  title: "DisasterVision",
  description:
    "AI-powered satellite image analysis for landslide and flood prediction"
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>
        <Navbar />
        <main className="container mx-auto px-4 pt-20">{children}</main>
      </body>
    </html>
  );
}
