"use client";

import { useCallback, useRef, useState } from "react";
import { apiStream } from "@/lib/api";
import type { StoreAction } from "@/hooks/useMessageStore";

/* ── Public state (no token accumulation — that lives in the store) ── */

interface StreamState {
  isStreaming: boolean;
  error: string | null;
  /** Which session this stream belongs to. */
  sessionId: string | null;
}

const INITIAL: StreamState = {
  isStreaming: false,
  error: null,
  sessionId: null,
};

/* ── Hook ─────────────────────────────────────────────────────── */

/**
 * useStreaming — manages the SSE connection lifecycle.
 *
 * Token data is dispatched directly into the message store via the
 * provided `dispatch` function, eliminating the intermediate state
 * that previously caused double-renders and flicker.
 *
 * Stale stream protection: a monotonic generation ref scoped to the
 * hook instance (not module-level) ensures aborted streams cannot
 * dispatch after a new stream has started.
 */
export function useStreaming(
  dispatch: React.Dispatch<StoreAction>,
  onStreamComplete?: () => void,
) {
  const [state, setState] = useState<StreamState>(INITIAL);
  const abortRef = useRef<(() => void) | null>(null);
  const genRef = useRef(0);

  /** Abort any in-flight stream without resetting state. */
  const abortActiveStream = useCallback(() => {
    abortRef.current?.();
    abortRef.current = null;
  }, []);

  const startStream = useCallback(
    async (
      sessionId: string,
      question: string,
      userTempId: string,
      asstTempId: string,
    ) => {
      // Abort previous stream
      abortActiveStream();

      const gen = ++genRef.current;

      setState({ isStreaming: true, error: null, sessionId });

      const { reader, abort } = apiStream(
        `/chat/sessions/${sessionId}/stream`,
        { question },
      );
      abortRef.current = abort;

      const decoder = new TextDecoder();
      let buffer = "";
      let collectedTokens = "";
      /** Current assistant ID — updated when server echoes real IDs at START. */
      let currentAsstId: string | number = asstTempId;
      let currentUserTempId: string | number = userTempId;
      /** Track if we've received message IDs yet. */
      let receivedMessageIds = false;

      /** Guard: only dispatch if this stream is still the active one. */
      const isActive = () => genRef.current === gen;

      try {
        // eslint-disable-next-line no-constant-condition
        while (true) {
          const { done, value } = await reader.read();
          if (done || !isActive()) break;

          buffer += decoder.decode(value, { stream: true });
          // SSE events are delimited by \n\n — split on full boundaries
          const events = buffer.split("\n\n");
          buffer = events.pop() || "";

          for (const event of events) {
            if (!isActive()) break;

            const lines = event.split("\n");
            let pendingEvt: string | null = null;

            for (const line of lines) {
              if (!isActive()) break;

              const cleaned = line.replace(/\r$/, "");
              if (!cleaned) continue;

              if (cleaned.startsWith("event: ")) {
                pendingEvt = cleaned.slice(7).trim();
                continue;
              }

              if (cleaned.startsWith("data: ")) {
                const payload = cleaned.slice(6);

                if (pendingEvt) {
                  switch (pendingEvt) {
                    case "message_ids":
                      // Message IDs now arrive at START (before any tokens)
                      try {
                        const ids = JSON.parse(payload) as {
                          user: number;
                          assistant: number;
                        };
                        dispatch({
                          type: "REPLACE_TEMP_IDS",
                          userTempId: currentUserTempId,
                          asstTempId: currentAsstId,
                          realUserId: ids.user,
                          realAsstId: ids.assistant,
                        });
                        currentAsstId = ids.assistant;
                        currentUserTempId = ids.user;
                        receivedMessageIds = true;
                      } catch {
                        /* ignore malformed */
                      }
                      break;
                    case "sources":
                      dispatch({
                        type: "STREAM_META",
                        asstTempId: currentAsstId,
                        sources: payload.split("; ").filter(Boolean),
                      });
                      break;
                    case "rewritten":
                      dispatch({
                        type: "STREAM_META",
                        asstTempId: currentAsstId,
                        rewrittenQuery: payload,
                      });
                      break;
                    case "complexity":
                      dispatch({
                        type: "STREAM_META",
                        asstTempId: currentAsstId,
                        complexity: payload,
                      });
                      break;
                    case "tier":
                      dispatch({
                        type: "STREAM_META",
                        asstTempId: currentAsstId,
                        tier: payload,
                      });
                      break;
                    case "error":
                      setState((s) => ({ ...s, error: payload }));
                      break;
                    case "retry":
                      // Backend is retrying generation
                      dispatch({
                        type: "STREAM_RETRY",
                        asstTempId: currentAsstId,
                      });
                      collectedTokens = ""; // Reset accumulated tokens
                      break;
                  }
                  pendingEvt = null;
                  continue;
                }

                // Default (unnamed) data — token stream (deltas from backend)
                if (payload === "[DONE]") continue;
                collectedTokens += payload;
                dispatch({
                  type: "STREAM_TOKEN",
                  asstTempId: currentAsstId,
                  content: collectedTokens,
                });
              }
            }
          }
        }
      } catch (err: unknown) {
        if (!isActive()) return;
        if (err instanceof DOMException && err.name === "AbortError") {
          // user cancelled — not an error
        } else {
          setState((s) => ({
            ...s,
            error: err instanceof Error ? err.message : "Stream failed",
          }));
        }
      } finally {
        if (!isActive()) return;

        // Finalize with the full accumulated text.
        dispatch({
          type: "FINALIZE_STREAM",
          asstTempId: currentAsstId,
          content: collectedTokens,
        });

        onStreamComplete?.();

        setState((s) => ({ ...s, isStreaming: false }));
      }
    },
    [abortActiveStream, dispatch, onStreamComplete],
  );

  const stopStream = useCallback(() => {
    abortActiveStream();
  }, [abortActiveStream]);

  const reset = useCallback(() => {
    setState(INITIAL);
  }, []);

  return { ...state, startStream, stopStream, reset, abortActiveStream };
}
