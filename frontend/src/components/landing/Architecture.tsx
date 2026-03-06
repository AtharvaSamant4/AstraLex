"use client";

import { motion } from "framer-motion";
import { Search, Layers, BrainCircuit, MessageSquareText, ArrowDown } from "lucide-react";

const STEPS = [
  {
    icon: MessageSquareText,
    label: "User Query",
    desc: "Natural language legal question",
  },
  {
    icon: Search,
    label: "Hybrid Retrieval",
    desc: "FAISS dense search + BM25 sparse search → RRF fusion",
  },
  {
    icon: Layers,
    label: "Reranking",
    desc: "Cross-encoder reranking for precision",
  },
  {
    icon: BrainCircuit,
    label: "AI Response",
    desc: "Gemini LLM generates cited, verified answer",
  },
];

export default function Architecture() {
  return (
    <section className="py-28 px-6">
      <div className="mx-auto max-w-5xl">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center"
        >
          <h2 className="text-3xl font-bold sm:text-4xl">
            How the <span className="gradient-text">RAG Pipeline</span> Works
          </h2>
          <p className="mx-auto mt-4 max-w-2xl text-muted-foreground">
            A multi-stage retrieval-augmented generation pipeline that ensures
            every answer is grounded in authoritative Indian legal sources.
          </p>
        </motion.div>

        {/* Pipeline steps */}
        <div className="mt-16 flex flex-col items-center gap-2">
          {STEPS.map((step, i) => (
            <motion.div
              key={step.label}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: i * 0.12 }}
              className="w-full max-w-lg"
            >
              <div className="flex items-start gap-4 rounded-2xl border border-border bg-card p-5 transition hover:border-primary/40">
                <div className="flex-shrink-0 rounded-xl bg-primary/10 p-3 text-primary">
                  <step.icon className="h-6 w-6" />
                </div>
                <div>
                  <h3 className="font-semibold">{step.label}</h3>
                  <p className="mt-1 text-sm text-muted-foreground">{step.desc}</p>
                </div>
              </div>
              {i < STEPS.length - 1 && (
                <div className="flex justify-center py-1 text-muted-foreground/40">
                  <ArrowDown className="h-5 w-5" />
                </div>
              )}
            </motion.div>
          ))}
        </div>

        {/* Tiers */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="mt-20"
        >
          <h3 className="text-center text-xl font-bold mb-8">
            Adaptive 3-Tier Routing
          </h3>
          <div className="grid gap-4 sm:grid-cols-3">
            {[
              {
                tier: "Fast",
                color: "text-green-400",
                bg: "bg-green-400/10",
                desc: "Simple factual queries answered in one retrieval pass.",
              },
              {
                tier: "Standard",
                color: "text-yellow-400",
                bg: "bg-yellow-400/10",
                desc: "Moderate queries with query rewriting and multi-pass retrieval.",
              },
              {
                tier: "Deep Research",
                color: "text-red-400",
                bg: "bg-red-400/10",
                desc: "Complex multi-part queries with research planning and evidence graphs.",
              },
            ].map((t) => (
              <div
                key={t.tier}
                className="rounded-2xl border border-border bg-card p-5 text-center"
              >
                <div className={`text-xs font-bold uppercase tracking-wider ${t.color} ${t.bg} inline-block rounded-full px-3 py-1`}>
                  {t.tier}
                </div>
                <p className="mt-3 text-sm text-muted-foreground">{t.desc}</p>
              </div>
            ))}
          </div>
        </motion.div>
      </div>
    </section>
  );
}
