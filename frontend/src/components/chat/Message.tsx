"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Copy, Check, User, Scale, BookOpen, ChevronDown, ChevronUp } from "lucide-react";
import type { Message as MessageType } from "@/types";

interface MessageProps {
  message: MessageType;
  isStreaming?: boolean;
}

export default function MessageBubble({ message, isStreaming }: MessageProps) {
  const isUser = message.role === "user";
  const [copied, setCopied] = useState(false);
  const [showSources, setShowSources] = useState(false);

  const handleCopy = async () => {
    await navigator.clipboard.writeText(message.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.25 }}
      className={`flex items-start gap-3 ${isUser ? "flex-row-reverse" : ""}`}
    >
      {/* Avatar */}
      <div
        className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-lg ${
          isUser ? "bg-indigo-500/20" : "bg-primary/10"
        }`}
      >
        {isUser ? (
          <User className="h-4 w-4 text-indigo-400" />
        ) : (
          <Scale className="h-4 w-4 text-primary" />
        )}
      </div>

      {/* Content */}
      <div className={`group min-w-0 max-w-[85%] ${isUser ? "text-right" : ""}`}>
        <div
          className={`rounded-2xl px-4 py-2.5 text-sm ${
            isUser
              ? "rounded-tr-md bg-indigo-500/15 text-foreground"
              : "rounded-tl-md bg-muted/50 text-foreground"
          }`}
        >
          {isUser ? (
            <p className="whitespace-pre-wrap text-left">{message.content}</p>
          ) : (
            <div className="prose-chat text-left">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {message.content}
              </ReactMarkdown>
              {isStreaming && (
                <span className="ml-0.5 inline-block h-4 w-1.5 animate-pulse rounded-sm bg-primary" />
              )}
            </div>
          )}
        </div>

        {/* Actions & metadata (assistant only) */}
        {!isUser && !isStreaming && message.content && (
          <div className="mt-1.5 flex items-center gap-2">
            <button
              onClick={handleCopy}
              className="flex items-center gap-1 rounded-md px-2 py-1 text-xs text-muted-foreground opacity-0 transition-opacity hover:bg-muted group-hover:opacity-100"
            >
              {copied ? (
                <Check className="h-3 w-3 text-green-400" />
              ) : (
                <Copy className="h-3 w-3" />
              )}
              {copied ? "Copied" : "Copy"}
            </button>

            {message.tier && (
              <span className="text-xs text-muted-foreground">
                {message.tier} tier
              </span>
            )}

            {message.complexity && (
              <span className="text-xs text-muted-foreground capitalize">
                • {message.complexity}
              </span>
            )}
          </div>
        )}

        {/* Sources */}
        {!isUser && message.sources && message.sources.length > 0 && !isStreaming && (
          <div className="mt-2">
            <button
              onClick={() => setShowSources((o) => !o)}
              className="flex items-center gap-1 text-xs text-primary hover:underline"
            >
              <BookOpen className="h-3 w-3" />
              {message.sources.length} source{message.sources.length !== 1 ? "s" : ""}
              {showSources ? <ChevronUp className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />}
            </button>

            {showSources && (
              <motion.ul
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: "auto", opacity: 1 }}
                className="mt-1.5 space-y-1 overflow-hidden"
              >
                {message.sources.map((src, i) => (
                  <li
                    key={i}
                    className="rounded-md bg-muted/30 px-3 py-1.5 text-xs text-muted-foreground"
                  >
                    {src}
                  </li>
                ))}
              </motion.ul>
            )}
          </div>
        )}

        {/* Streaming sources preview */}
        {isStreaming && message.sources && message.sources.length > 0 && (
          <div className="mt-2 flex flex-wrap gap-1">
            {message.sources.map((src, i) => (
              <span
                key={i}
                className="inline-flex items-center gap-1 rounded-full bg-primary/10 px-2 py-0.5 text-[10px] text-primary"
              >
                <BookOpen className="h-2.5 w-2.5" />
                {src}
              </span>
            ))}
          </div>
        )}
      </div>
    </motion.div>
  );
}
