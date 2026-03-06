"use client";

import { motion } from "framer-motion";
import { BookOpen, Landmark, Scale, FileText, Users, Gavel, Scroll } from "lucide-react";

const DATASETS = [
  {
    icon: Landmark,
    title: "Constitution of India",
    desc: "Fundamental rights, directive principles, and constitutional provisions.",
    items: "395+ Articles",
  },
  {
    icon: Gavel,
    title: "Indian Penal Code (IPC)",
    desc: "Complete criminal law — offences, punishments, and definitions.",
    items: "511 Sections",
  },
  {
    icon: Scale,
    title: "Code of Criminal Procedure (CrPC)",
    desc: "Procedural law for criminal trials, arrests, bail, and investigations.",
    items: "484 Sections",
  },
  {
    icon: Users,
    title: "Hindu Marriage Act",
    desc: "Marriage, divorce, maintenance, and matrimonial rights for Hindus.",
    items: "30 Sections",
  },
  {
    icon: Scroll,
    title: "Special Marriage Act",
    desc: "Inter-faith marriages and registration provisions.",
    items: "50+ Sections",
  },
  {
    icon: FileText,
    title: "Dowry Prohibition Act",
    desc: "Laws against dowry demand, giving, and related penalties.",
    items: "Key Provisions",
  },
  {
    icon: BookOpen,
    title: "Domestic Violence Act",
    desc: "Protection of women from domestic violence — rights and remedies.",
    items: "37 Sections",
  },
];

const container = {
  hidden: {},
  visible: { transition: { staggerChildren: 0.06 } },
};
const item = {
  hidden: { opacity: 0, y: 20 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.4 } },
};

export default function DatasetsSection() {
  return (
    <section className="py-28 px-6 bg-card/30">
      <div className="mx-auto max-w-7xl">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center"
        >
          <h2 className="text-3xl font-bold sm:text-4xl">
            Comprehensive{" "}
            <span className="gradient-text">Legal Knowledge Base</span>
          </h2>
          <p className="mx-auto mt-4 max-w-2xl text-muted-foreground">
            Structured datasets covering major Indian statutes — all indexed,
            chunked, and embedded for instant retrieval.
          </p>
        </motion.div>

        <motion.div
          variants={container}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, margin: "-80px" }}
          className="mt-16 grid gap-5 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4"
        >
          {DATASETS.map((d) => (
            <motion.div
              key={d.title}
              variants={item}
              className="group rounded-2xl border border-border bg-card p-5 transition hover:border-primary/40 hover:shadow-lg hover:shadow-primary/5"
            >
              <div className="mb-3 inline-flex rounded-xl bg-primary/10 p-3 text-primary group-hover:bg-primary/20">
                <d.icon className="h-5 w-5" />
              </div>
              <h3 className="font-semibold">{d.title}</h3>
              <p className="mt-1 text-sm text-muted-foreground">{d.desc}</p>
              <div className="mt-3 inline-block rounded-full bg-muted px-3 py-1 text-xs font-medium text-muted-foreground">
                {d.items}
              </div>
            </motion.div>
          ))}
        </motion.div>
      </div>
    </section>
  );
}
