"use client";

import Navbar from "@/components/landing/Navbar";
import Footer from "@/components/landing/Footer";
import { motion } from "framer-motion";
import {
  Landmark,
  Gavel,
  Scale,
  Users,
  Scroll,
  FileText,
  BookOpen,
  Database,
  Zap,
  Hash,
} from "lucide-react";

const DATASETS = [
  {
    icon: Landmark,
    title: "Constitution of India",
    category: "Constitutional Law",
    desc: "The supreme law of India including fundamental rights, directive principles, fundamental duties, and amendments. Covers the structure of the government and the relationship between citizens and the state.",
    stats: "395+ Articles • 25 Parts • 12 Schedules",
    color: "text-blue-400",
    bg: "bg-blue-400/10",
  },
  {
    icon: Gavel,
    title: "Indian Penal Code (IPC)",
    category: "Criminal Law",
    desc: "The comprehensive criminal code covering all substantive aspects of criminal law — offences against the state, body, property, and public order, along with their definitions and punishments.",
    stats: "511 Sections • 23 Chapters",
    color: "text-red-400",
    bg: "bg-red-400/10",
  },
  {
    icon: Scale,
    title: "Code of Criminal Procedure (CrPC)",
    category: "Procedural Law",
    desc: "The principal legislation on procedure for criminal trials, investigation, arrest, bail proceedings, trial processes, appeals, and more across Indian courts.",
    stats: "484 Sections • 56 Forms",
    color: "text-amber-400",
    bg: "bg-amber-400/10",
  },
  {
    icon: Users,
    title: "Hindu Marriage Act",
    category: "Family Law",
    desc: "Codified law governing marriage, divorce, judicial separation, restitution of conjugal rights, maintenance, and related matters for Hindus.",
    stats: "30 Sections",
    color: "text-pink-400",
    bg: "bg-pink-400/10",
  },
  {
    icon: Scroll,
    title: "Special Marriage Act",
    category: "Family Law",
    desc: "Legislation enabling inter-religious marriage, court marriage registration, and civil union provisions across communities regardless of faith.",
    stats: "50+ Sections",
    color: "text-green-400",
    bg: "bg-green-400/10",
  },
  {
    icon: FileText,
    title: "Dowry Prohibition Act",
    category: "Social Legislation",
    desc: "Laws prohibiting the giving or taking of dowry, including penalties for violations and provisions for protection of women.",
    stats: "Key Provisions",
    color: "text-orange-400",
    bg: "bg-orange-400/10",
  },
  {
    icon: BookOpen,
    title: "Domestic Violence Act",
    category: "Social Legislation",
    desc: "Protection of Women from Domestic Violence Act — covers physical, sexual, verbal, emotional, and economic abuse with provisions for protection orders.",
    stats: "37 Sections",
    color: "text-violet-400",
    bg: "bg-violet-400/10",
  },
];

const container = {
  hidden: {},
  visible: { transition: { staggerChildren: 0.08 } },
};
const item = {
  hidden: { opacity: 0, y: 20 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.45 } },
};

export default function DatasetsPage() {
  return (
    <main className="min-h-screen">
      <Navbar />

      {/* Hero */}
      <section className="pt-32 pb-16 px-6 text-center">
        <motion.h1
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-4xl font-bold sm:text-5xl"
        >
          Legal <span className="gradient-text">Knowledge Base</span>
        </motion.h1>
        <motion.p
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="mx-auto mt-4 max-w-2xl text-muted-foreground text-lg"
        >
          AstraLex is powered by 7 comprehensive Indian legal datasets, all
          structured, chunked, and embedded for high-precision retrieval.
        </motion.p>
      </section>

      {/* Stats bar */}
      <section className="px-6 pb-16">
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.2 }}
          className="mx-auto max-w-4xl grid grid-cols-3 gap-4"
        >
          {[
            { icon: Database, value: "7", label: "Legal Datasets" },
            { icon: Hash, value: "161+", label: "Knowledge Chunks" },
            { icon: Zap, value: "384-dim", label: "Vector Embeddings" },
          ].map((s) => (
            <div
              key={s.label}
              className="flex flex-col items-center rounded-2xl border border-border bg-card p-5 text-center"
            >
              <s.icon className="h-5 w-5 text-primary mb-2" />
              <div className="text-2xl font-bold gradient-text">{s.value}</div>
              <div className="text-xs text-muted-foreground">{s.label}</div>
            </div>
          ))}
        </motion.div>
      </section>

      {/* Dataset cards */}
      <section className="px-6 pb-28">
        <motion.div
          variants={container}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true }}
          className="mx-auto max-w-6xl grid gap-6 sm:grid-cols-2"
        >
          {DATASETS.map((d) => (
            <motion.div
              key={d.title}
              variants={item}
              className="group rounded-2xl border border-border bg-card p-6 transition hover:border-primary/40 hover:shadow-lg hover:shadow-primary/5"
            >
              <div className="flex items-start gap-4">
                <div className={`flex-shrink-0 rounded-xl ${d.bg} p-3 ${d.color}`}>
                  <d.icon className="h-6 w-6" />
                </div>
                <div className="flex-1">
                  <div className="flex items-center gap-3">
                    <h3 className="text-lg font-semibold">{d.title}</h3>
                    <span className="rounded-full bg-muted px-2.5 py-0.5 text-xs text-muted-foreground">
                      {d.category}
                    </span>
                  </div>
                  <p className="mt-2 text-sm text-muted-foreground leading-relaxed">
                    {d.desc}
                  </p>
                  <div className="mt-3 text-xs font-medium text-primary">
                    {d.stats}
                  </div>
                </div>
              </div>
            </motion.div>
          ))}
        </motion.div>
      </section>

      <Footer />
    </main>
  );
}
