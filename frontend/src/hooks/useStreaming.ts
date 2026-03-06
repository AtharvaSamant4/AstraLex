"use client";

import { useCallback, useRef, useState } from "react";
import { apiStream } from "@/lib/api";

interface StreamState {
  isStreaming: boolean;
  tokens: string;
  sources: string[];
  rewrittenQuery: string;
  complexity: string;
  tier: string;
  messageIds: { user: number; assistant: number } | null;
  error: string | null;
}

const INITIAL: StreamState = {
  isStreaming: false,
  tokens: "",
  sources: [],
  rewrittenQuery: "",
  complexity: "",
  tier: "",
  messageIds: null,
  error: null,
};

export function useStreaming() {
  const [state, setState] = useState<StreamState>(INITIAL);
  const abortRef = useRef<(() => void) | null>(null);

  const startStream = useCallback(
    async (sessionId: string, question: string) => {
      setState({ ...INITIAL, isStreaming: true });

      const { reader, abort } = apiStream(
        `/chat/sessions/${sessionId}/stream`,
        { question },
      );
      abortRef.current = abort;

      const decoder = new TextDecoder();
      let buffer = "";
      let collectedTokens = "";

      try {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n");
          buffer = lines.pop() || "";

          for (const line of lines) {
            if (line.startsWith("data: ")) {
              const payload = line.slice(6);
              if (payload === "[DONE]") continue;
              collectedTokens += payload;
              setState((s) => ({ ...s, tokens: collectedTokens }));
            } else if (line.startsWith("event: ")) {
              const eventName = line.slice(7).trim();
              // Next line should be the data
              const dataLineIdx = lines.indexOf(line) + 1;
              if (dataLineIdx < lines.length && lines[dataLineIdx].startsWith("data: ")) {
                const eventData = lines[dataLineIdx].slice(6);
                switch (eventName) {
                  case "sources":
                    setState((s) => ({
                      ...s,
                      sources: eventData.split("; ").filter(Boolean),
                    }));
                    break;
                  case "rewritten":
                    setState((s) => ({ ...s, rewrittenQuery: eventData }));
                    break;
                  case "complexity":
                    setState((s) => ({ ...s, complexity: eventData }));
                    break;
                  case "message_ids":
                    try {
                      setState((s) => ({
                        ...s,
                        messageIds: JSON.parse(eventData),
                      }));
                    } catch {}
                    break;
                  case "error":
                    setState((s) => ({ ...s, error: eventData }));
                    break;
                }
              }
            }
          }
        }
      } catch (err: unknown) {
        if (err instanceof DOMException && err.name === "AbortError") {
          // user cancelled
        } else {
          setState((s) => ({
            ...s,
            error: err instanceof Error ? err.message : "Stream failed",
          }));
        }
      } finally {
        setState((s) => ({ ...s, isStreaming: false }));
      }
    },
    [],
  );

  const stopStream = useCallback(() => {
    abortRef.current?.();
  }, []);

  const reset = useCallback(() => {
    setState(INITIAL);
  }, []);

  return { ...state, startStream, stopStream, reset };
}
