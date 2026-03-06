"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { motion } from "framer-motion";
import { PanelLeft, Scale, Sparkles } from "lucide-react";
import { useSessionDetail, refreshSessionDetail } from "@/hooks/useChat";
import { useStreaming } from "@/hooks/useStreaming";
import type { Message as MessageType } from "@/types";
import MessageBubble from "./Message";
import MessageInput from "./MessageInput";

interface ChatWindowProps {
  sessionId: string | null;
  sidebarOpen: boolean;
  onToggleSidebar: () => void;
}

export default function ChatWindow({ sessionId, sidebarOpen, onToggleSidebar }: ChatWindowProps) {
  const { detail, isLoading } = useSessionDetail(sessionId);
  const streaming = useStreaming();
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const [optimisticMessages, setOptimisticMessages] = useState<MessageType[]>([]);

  const messages = detail?.messages || [];
  const allMessages = [...messages, ...optimisticMessages];

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [allMessages.length, streaming.tokens]);

  // When streaming ends and we have messageIds, refresh the session detail to get full messages
  useEffect(() => {
    if (!streaming.isStreaming && streaming.messageIds && sessionId) {
      setOptimisticMessages([]);
      refreshSessionDetail(sessionId);
      streaming.reset();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [streaming.isStreaming, streaming.messageIds, sessionId]);

  const handleSend = useCallback(
    async (question: string) => {
      if (!sessionId || streaming.isStreaming) return;

      // Add optimistic user message
      const tempUserMsg: MessageType = {
        id: Date.now(),
        session_id: sessionId,
        role: "user",
        content: question,
        sources: [],
        rewritten_query: null,
        complexity: null,
        tier: null,
        timings: {},
        created_at: new Date().toISOString(),
      };
      setOptimisticMessages([tempUserMsg]);

      // Start streaming
      streaming.startStream(sessionId, question);
    },
    [sessionId, streaming],
  );

  // Empty state
  if (!sessionId) {
    return (
      <div className="flex flex-1 flex-col">
        {/* Toolbar */}
        <div className="flex h-12 items-center border-b border-border px-4">
          {!sidebarOpen && (
            <button
              onClick={onToggleSidebar}
              className="mr-3 rounded-md p-1 text-muted-foreground hover:bg-muted transition-colors"
              aria-label="Open sidebar"
            >
              <PanelLeft className="h-5 w-5" />
            </button>
          )}
        </div>

        <div className="flex flex-1 items-center justify-center">
          <div className="text-center">
            <motion.div
              initial={{ scale: 0.8, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              transition={{ duration: 0.4 }}
              className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-2xl bg-primary/10"
            >
              <Scale className="h-8 w-8 text-primary" />
            </motion.div>
            <h2 className="text-xl font-semibold mb-2">Welcome to AstraLex</h2>
            <p className="mx-auto max-w-md text-sm text-muted-foreground">
              Start a new chat from the sidebar to ask questions about Indian law.
              Our AI will analyze your query using deep research capabilities.
            </p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-1 flex-col min-w-0">
      {/* Toolbar */}
      <div className="flex h-12 items-center justify-between border-b border-border px-4">
        <div className="flex items-center gap-2 min-w-0">
          {!sidebarOpen && (
            <button
              onClick={onToggleSidebar}
              className="mr-1 rounded-md p-1 text-muted-foreground hover:bg-muted transition-colors"
              aria-label="Open sidebar"
            >
              <PanelLeft className="h-5 w-5" />
            </button>
          )}
          <span className="truncate text-sm font-medium">
            {detail?.session?.title || "Untitled Chat"}
          </span>
        </div>
        {streaming.tier && (
          <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
            <Sparkles className="h-3.5 w-3.5" />
            <span className="capitalize">{streaming.tier} Tier</span>
          </div>
        )}
      </div>

      {/* Messages area */}
      <div className="flex-1 overflow-y-auto px-4 py-4 scrollbar-thin">
        {isLoading ? (
          <div className="flex items-center justify-center py-12">
            <div className="h-6 w-6 animate-spin rounded-full border-2 border-primary border-t-transparent" />
          </div>
        ) : allMessages.length === 0 && !streaming.isStreaming ? (
          <div className="flex flex-col items-center justify-center py-20">
            <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-xl bg-primary/10">
              <Sparkles className="h-6 w-6 text-primary" />
            </div>
            <p className="text-sm text-muted-foreground">
              Ask a question about Indian law to get started.
            </p>
          </div>
        ) : (
          <div className="mx-auto max-w-3xl space-y-4">
            {allMessages.map((msg) => (
              <MessageBubble key={msg.id} message={msg} />
            ))}

            {/* Streaming assistant response */}
            {streaming.isStreaming && streaming.tokens && (
              <MessageBubble
                message={{
                  id: -1,
                  session_id: sessionId,
                  role: "assistant",
                  content: streaming.tokens,
                  sources: streaming.sources,
                  rewritten_query: streaming.rewrittenQuery || null,
                  complexity: streaming.complexity || null,
                  tier: streaming.tier || null,
                  timings: {},
                  created_at: new Date().toISOString(),
                }}
                isStreaming
              />
            )}

            {/* Streaming "thinking" indicator */}
            {streaming.isStreaming && !streaming.tokens && (
              <div className="flex items-start gap-3">
                <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-primary/10">
                  <Scale className="h-4 w-4 text-primary" />
                </div>
                <div className="flex items-center gap-1.5 pt-2">
                  <span className="h-2 w-2 animate-bounce rounded-full bg-primary/60 [animation-delay:0ms]" />
                  <span className="h-2 w-2 animate-bounce rounded-full bg-primary/60 [animation-delay:150ms]" />
                  <span className="h-2 w-2 animate-bounce rounded-full bg-primary/60 [animation-delay:300ms]" />
                </div>
              </div>
            )}

            {streaming.error && (
              <div className="mx-auto max-w-md rounded-lg bg-destructive/10 px-4 py-3 text-sm text-destructive">
                {streaming.error}
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      {/* Input area */}
      <MessageInput
        onSend={handleSend}
        isStreaming={streaming.isStreaming}
        onStop={streaming.stopStream}
      />
    </div>
  );
}
