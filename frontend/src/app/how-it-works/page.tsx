"use client";

import Navbar from "@/components/landing/Navbar";
import Footer from "@/components/landing/Footer";
import { motion } from "framer-motion";
import {
  MessageSquareText,
  Search,
  Layers,
  BrainCircuit,
  ArrowDown,
  RefreshCw,
  History,
  Upload,
  Database,
} from "lucide-react";

const PIPELINE = [
  { icon: MessageSquareText, title: "User Query", desc: "You ask a natural language question about Indian law." },
  { icon: RefreshCw, title: "Query Rewriting", desc: "The system rewrites your query using conversation context for better retrieval." },
  { icon: Search, title: "Hybrid Retrieval", desc: "FAISS dense vector search + BM25 sparse keyword search, fused with Reciprocal Rank Fusion (RRF)." },
  { icon: Layers, title: "Cross-Encoder Reranking", desc: "A cross-encoder model (ms-marco-MiniLM-L-6-v2) re-scores and filters the most relevant legal passages." },
  { icon: BrainCircuit, title: "LLM Response Generation", desc: "Google Gemini generates a comprehensive, cited answer grounded in the retrieved legal texts." },
];

export default function HowItWorks() {
  return (
    <main className="min-h-screen">
      <Navbar />

      {/* Hero */}
      <section className="pt-32 pb-20 px-6 text-center">
        <motion.h1
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-4xl font-bold sm:text-5xl"
        >
          How <span className="gradient-text">AstraLex</span> Works
        </motion.h1>
        <motion.p
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="mx-auto mt-4 max-w-2xl text-muted-foreground text-lg"
        >
          A deep-research agentic RAG pipeline that retrieves, verifies, and
          synthesises information from authoritative Indian legal sources.
        </motion.p>
      </section>

      {/* Pipeline visualisation */}
      <section className="px-6 pb-20">
        <div className="mx-auto max-w-2xl flex flex-col items-center gap-2">
          {PIPELINE.map((step, i) => (
            <motion.div
              key={step.title}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: i * 0.1 }}
              className="w-full"
            >
              <div className="flex items-start gap-4 rounded-2xl border border-border bg-card p-6 transition hover:border-primary/40">
                <div className="flex-shrink-0 rounded-xl bg-primary/10 p-3 text-primary">
                  <step.icon className="h-6 w-6" />
                </div>
                <div>
                  <h3 className="font-semibold text-lg">{step.title}</h3>
                  <p className="mt-1 text-sm text-muted-foreground leading-relaxed">{step.desc}</p>
                </div>
              </div>
              {i < PIPELINE.length - 1 && (
                <div className="flex justify-center py-1 text-muted-foreground/30">
                  <ArrowDown className="h-5 w-5" />
                </div>
              )}
            </motion.div>
          ))}
        </div>
      </section>

      {/* Conversational Memory */}
      <section className="py-20 px-6 bg-card/30">
        <div className="mx-auto max-w-5xl grid gap-12 lg:grid-cols-2 items-center">
          <motion.div
            initial={{ opacity: 0, x: -30 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
          >
            <div className="inline-flex rounded-xl bg-primary/10 p-3 text-primary mb-4">
              <History className="h-6 w-6" />
            </div>
            <h2 className="text-2xl font-bold sm:text-3xl">Conversation Memory</h2>
            <p className="mt-4 text-muted-foreground leading-relaxed">
              Every message is persisted in PostgreSQL. When you ask a follow-up,
              the last 10 messages are hydrated into the pipeline&apos;s context window,
              enabling accurate query rewriting and contextual responses.
            </p>
          </motion.div>
          <motion.div
            initial={{ opacity: 0, x: 30 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            className="rounded-2xl border border-border bg-card p-6 space-y-3"
          >
            {[
              { role: "user", text: "What is Section 302 IPC?" },
              { role: "assistant", text: "Section 302 deals with punishment for murder — death or life imprisonment..." },
              { role: "user", text: "What about attempt to murder?" },
              { role: "assistant", text: "That falls under Section 307 IPC — up to 10 years imprisonment..." },
            ].map((msg, i) => (
              <div key={i} className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
                <div className={`rounded-xl px-4 py-2 text-sm max-w-[80%] ${msg.role === "user" ? "bg-primary text-primary-foreground" : "bg-muted text-foreground"}`}>
                  {msg.text}
                </div>
              </div>
            ))}
          </motion.div>
        </div>
      </section>

      {/* Document Upload */}
      <section className="py-20 px-6">
        <div className="mx-auto max-w-5xl grid gap-12 lg:grid-cols-2 items-center">
          <motion.div
            initial={{ opacity: 0, x: 30 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            className="order-2 lg:order-1 rounded-2xl border border-border bg-card p-6"
          >
            <div className="space-y-3">
              {["contract.pdf", "agreement.docx", "notes.txt"].map((f) => (
                <div key={f} className="flex items-center justify-between rounded-lg bg-muted px-4 py-3 text-sm">
                  <span>{f}</span>
                  <span className="text-xs text-green-400 font-medium">Ready</span>
                </div>
              ))}
            </div>
          </motion.div>
          <motion.div
            initial={{ opacity: 0, x: -30 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            className="order-1 lg:order-2"
          >
            <div className="inline-flex rounded-xl bg-primary/10 p-3 text-primary mb-4">
              <Upload className="h-6 w-6" />
            </div>
            <h2 className="text-2xl font-bold sm:text-3xl">Document Uploads</h2>
            <p className="mt-4 text-muted-foreground leading-relaxed">
              Upload PDF, DOCX, or TXT files. They&apos;re automatically chunked,
              embedded with the same model as the legal dataset, and stored in
              your personal FAISS index. The AI seamlessly searches both the
              legal knowledge base and your uploaded documents.
            </p>
          </motion.div>
        </div>
      </section>

      <Footer />
    </main>
  );
}
