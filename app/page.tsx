"use client";

import { motion } from "framer-motion";
import Link from "next/link";

export default function HomePage() {
  return (
    <section className="flex flex-col items-center justify-center min-h-[60vh] text-center">
      <motion.h1
        initial={{ opacity: 0, y: -30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.7 }}
        className="text-5xl font-bold mb-4"
      >
        DisasterVision
      </motion.h1>
      <p className="max-w-xl text-lg mb-8">
        Predict landslides and floods using real‑time satellite imagery and
        weather data.
      </p>
      <Link
        href="/prediction"
        className="px-8 py-3 rounded-xl bg-blue-600 text-white hover:bg-blue-700 transition"
      >
        Try Prediction →
      </Link>
    </section>
  );
}
