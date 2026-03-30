"use client";

import { useEffect, useRef, useCallback } from "react";
import { motion } from "framer-motion";
import { PanelLeft, Scale, Sparkles } from "lucide-react";
import { useSessionDetail, refreshSessionDetail } from "@/hooks/useChat";
import { useMessageStore } from "@/hooks/useMessageStore";
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
  const { messages: allMessages, streamingAsstId, dispatch } = useMessageStore();
  const onStreamComplete = useCallback(() => {
    if (sessionId) refreshSessionDetail(sessionId);
  }, [sessionId]);
  const streaming = useStreaming(dispatch, onStreamComplete);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const scrollTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const prevSessionRef = useRef<string | null>(null);
  /** Ref-based send lock — prevents double sends from rapid clicks/enters
   *  that pass the `streaming.isStreaming` state guard before React re-renders. */
  const sendLockRef = useRef(false);

  // ── Sync server messages into the store (merge, never reset) ──
  // Use `detail?.messages` directly as the dependency — SWR keeps the
  // reference stable between re-renders.  `|| []` must NOT appear here
  // because it would create a new array on every render when detail is
  // undefined, causing an infinite dispatch→re-render→dispatch loop.
  const serverMessages = detail?.messages;
  useEffect(() => {
    if (!serverMessages) return;
    const withSeq = serverMessages.map((m, i) => ({
      ...m,
      seq: typeof m.seq === "number" ? m.seq : i + 1,
    }));
    dispatch({ type: "SYNC_SERVER_MESSAGES", serverMessages: withSeq });
  }, [serverMessages, dispatch]);

  // ── Session switching — abort & cleanup ────────────────────────
  useEffect(() => {
    if (prevSessionRef.current !== sessionId) {
      streaming.abortActiveStream();
      streaming.reset();
      dispatch({ type: "SWITCH_SESSION" });
      prevSessionRef.current = sessionId;
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionId]);

  // ── Cleanup on unmount ─────────────────────────────────────────
  useEffect(() => {
    return () => {
      streaming.abortActiveStream();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ── Auto-scroll — only when a new message arrives or streaming ends ─
  // Tracking `allMessages.length` caused scrolls on every token because
  // `deriveMessages()` returns a new array each dispatch.  Instead, track
  // only discrete events: the message count snapshot and streaming status.
  const msgCountRef = useRef(0);
  useEffect(() => {
    const countChanged = allMessages.length !== msgCountRef.current;
    const streamJustEnded = !streaming.isStreaming;
    msgCountRef.current = allMessages.length;

    if (!countChanged && !streamJustEnded) return;

    if (scrollTimerRef.current) clearTimeout(scrollTimerRef.current);
    scrollTimerRef.current = setTimeout(() => {
      messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }, 80);
    return () => {
      if (scrollTimerRef.current) clearTimeout(scrollTimerRef.current);
    };
  }, [allMessages.length, streaming.isStreaming]);

  // Server refresh is now handled by the onStreamComplete callback
  // passed directly to useStreaming, which fires immediately after
  // FINALIZE_STREAM — no effect timing issues.

  // ── Send handler ───────────────────────────────────────────────
  const handleSend = useCallback(
    (question: string) => {
      if (!sessionId || streaming.isStreaming || sendLockRef.current) return;
      sendLockRef.current = true;

      const ts = Date.now();
      const userTempId = `opt_user_${ts}`;
      const asstTempId = `opt_asst_${ts}`;

      // Add optimistic messages into the store
      dispatch({
        type: "SEND",
        sessionId,
        question,
        userTempId,
        asstTempId,
      });

      // Start streaming — pass temp IDs so the stream can dispatch updates
      streaming.startStream(sessionId, question, userTempId, asstTempId);
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [sessionId, streaming.isStreaming, streaming.startStream, dispatch],
  );

  // Release the send lock when streaming ends
  useEffect(() => {
    if (!streaming.isStreaming) {
      sendLockRef.current = false;
    }
  }, [streaming.isStreaming]);

  // ── Empty state (no session selected) ──────────────────────────
  if (!sessionId) {
    return (
      <div className="flex flex-1 flex-col">
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

  // ── Derive streaming tier from the assistant placeholder ───────
  const streamingMsg = streamingAsstId
    ? allMessages.find((m) => m.id === streamingAsstId)
    : null;
  const displayTier = streamingMsg?.tier;

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
        {displayTier && (
          <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
            <Sparkles className="h-3.5 w-3.5" />
            <span className="capitalize">{displayTier} Tier</span>
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
            {allMessages.map((msg) => {
              const isStreamTarget = msg.id === streamingAsstId;
              // Hide empty assistant placeholder — thinking dots shown separately
              if (isStreamTarget && !msg.content) return null;
              return (
                <MessageBubble
                  key={msg.id}
                  message={msg}
                  isStreaming={isStreamTarget && streaming.isStreaming}
                />
              );
            })}

            {/* Thinking indicator (before first token arrives) */}
            {streaming.isStreaming && streamingMsg && !streamingMsg.content && (
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
        sessionId={sessionId}
      />
    </div>
  );
}
