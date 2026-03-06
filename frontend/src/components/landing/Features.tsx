"use client";

import { motion } from "framer-motion";
import {
  MessageSquare,
  Search,
  FileText,
  History,
  Upload,
  ShieldCheck,
} from "lucide-react";

const FEATURES = [
  {
    icon: MessageSquare,
    title: "Conversational AI",
    desc: "Chat naturally about Indian law — the AI understands context, follow-up questions, and nuanced legal queries.",
  },
  {
    icon: Search,
    title: "Legal Knowledge Retrieval",
    desc: "Hybrid search across IPC, CrPC, Constitution, Hindu Marriage Act, and more using FAISS + BM25 with reranking.",
  },
  {
    icon: FileText,
    title: "Document Analysis",
    desc: "Upload your own PDF, DOCX, or TXT documents and get AI-powered answers grounded in your content.",
  },
  {
    icon: History,
    title: "Persistent Chat History",
    desc: "All conversations are saved with full context — pick up exactly where you left off.",
  },
  {
    icon: Upload,
    title: "Document Uploads",
    desc: "Upload legal documents that get automatically chunked, embedded, and made searchable in your personal index.",
  },
  {
    icon: ShieldCheck,
    title: "Verified Responses",
    desc: "Every answer includes source citations and verification against the original legal texts.",
  },
];

const container = {
  hidden: {},
  visible: { transition: { staggerChildren: 0.08 } },
};

const item = {
  hidden: { opacity: 0, y: 30 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.5 } },
};

export default function Features() {
  return (
    <section className="py-28 px-6">
      <div className="mx-auto max-w-7xl">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center"
        >
          <h2 className="text-3xl font-bold sm:text-4xl">
            Everything You Need for{" "}
            <span className="gradient-text">Legal Research</span>
          </h2>
          <p className="mx-auto mt-4 max-w-2xl text-muted-foreground">
            A production-grade platform combining state-of-the-art retrieval
            with powerful language models.
          </p>
        </motion.div>

        <motion.div
          variants={container}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, margin: "-100px" }}
          className="mt-16 grid gap-6 sm:grid-cols-2 lg:grid-cols-3"
        >
          {FEATURES.map((f) => (
            <motion.div
              key={f.title}
              variants={item}
              className="group rounded-2xl border border-border bg-card p-6 transition-all hover:border-primary/40 hover:shadow-lg hover:shadow-primary/5"
            >
              <div className="mb-4 inline-flex rounded-xl bg-primary/10 p-3 text-primary transition-colors group-hover:bg-primary/20">
                <f.icon className="h-6 w-6" />
              </div>
              <h3 className="text-lg font-semibold">{f.title}</h3>
              <p className="mt-2 text-sm leading-relaxed text-muted-foreground">
                {f.desc}
              </p>
            </motion.div>
          ))}
        </motion.div>
      </div>
    </section>
  );
}
