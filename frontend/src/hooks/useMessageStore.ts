"use client";

/**
 * useMessageStore — Deterministic message state via useReducer.
 *
 * All mutations flow through a single reducer, eliminating race conditions
 * between effects. The store is the single source of truth for the merged
 * (server + optimistic) message list.
 *
 * Architectural rules enforced:
 *  1. Functional state updates only (reducer is pure).
 *  2. Stable message identity via `id` field.
 *  3. Optimistic → server ID replacement (REPLACE_TEMP_IDS).
 *  4. Server data is merged, never reset (SYNC_SERVER_MESSAGES).
 *  5. Single assistant placeholder (SEND creates both user + assistant).
 *  6. Streaming tokens mutate only the assistant placeholder (STREAM_TOKEN).
 *  7. Deduplication by `message.id` via Map.
 *  8. Deterministic ordering by `seq` (monotonic counter, not Date strings).
 */

import { useReducer, useCallback } from "react";
import type { Message } from "@/types";

/* ── Sequence counter ─────────────────────────────────────────── */
let globalSeq = 0;
function nextSeq(): number {
  return ++globalSeq;
}

/* ── Action types ─────────────────────────────────────────────── */

interface SendAction {
  type: "SEND";
  sessionId: string;
  question: string;
  userTempId: string;
  asstTempId: string;
}

interface StreamTokenAction {
  type: "STREAM_TOKEN";
  asstTempId: string | number;
  content: string;
}

interface StreamMetaAction {
  type: "STREAM_META";
  asstTempId: string | number;
  sources?: string[];
  rewrittenQuery?: string | null;
  complexity?: string | null;
  tier?: string | null;
}

interface ReplaceTempIdsAction {
  type: "REPLACE_TEMP_IDS";
  userTempId: string | number;
  asstTempId: string | number;
  realUserId: number;
  realAsstId: number;
}

interface FinalizeStreamAction {
  type: "FINALIZE_STREAM";
  asstTempId: string | number;
  content: string;
}

interface StreamRetryAction {
  type: "STREAM_RETRY";
  asstTempId: string | number;
}

interface RemoveAsstPlaceholderAction {
  type: "REMOVE_ASST_PLACEHOLDER";
  asstTempId: string | number;
}

interface SyncServerMessagesAction {
  type: "SYNC_SERVER_MESSAGES";
  serverMessages: Message[];
}

interface SwitchSessionAction {
  type: "SWITCH_SESSION";
}

type StoreAction =
  | SendAction
  | StreamTokenAction
  | StreamMetaAction
  | ReplaceTempIdsAction
  | FinalizeStreamAction
  | RemoveAsstPlaceholderAction
  | StreamRetryAction
  | SyncServerMessagesAction
  | SwitchSessionAction;

/* ── State ────────────────────────────────────────────────────── */

interface StoreState {
  /** Optimistic messages not yet confirmed by server. Keyed by id. */
  optimistic: Map<string | number, Message>;
  /** Latest server messages snapshot (used for dedup). */
  serverIds: Set<string | number>;
  /** Merged, deduplicated, ordered list (derived on every dispatch). */
  messages: Message[];
  /** Maps from current (possibly replaced) assistant temp ID back. */
  asstTempId: string | number | null;
}

/* ── Helpers ──────────────────────────────────────────────────── */

/** Merge server + optimistic into a deduplicated, seq-sorted array. */
function deriveMessages(
  serverMessages: Message[],
  optimistic: Map<string | number, Message>,
): Message[] {
  const merged = new Map<string | number, Message>();
  // Server provides canonical metadata, but optimistic content takes
  // precedence when both exist — prevents the visible message from
  // mutating when SWR refreshes after streaming finishes.
  for (const m of serverMessages) {
    const opt = optimistic.get(m.id);
    if (opt && opt.content) {
      merged.set(m.id, { ...m, content: opt.content });
    } else {
      merged.set(m.id, m);
    }
  }
  // Optimistic fills gaps
  for (const [id, m] of optimistic) {
    if (!merged.has(id)) merged.set(id, m);
  }
  return Array.from(merged.values()).sort((a, b) => a.seq - b.seq);
}

/* ── Reducer ──────────────────────────────────────────────────── */

function reducer(state: StoreState, action: StoreAction): StoreState {
  switch (action.type) {
    /* ─ User hits send ──────────────────────────────────────────── */
    case "SEND": {
      const userSeq = nextSeq();
      const asstSeq = nextSeq();
      const userMsg: Message = {
        id: action.userTempId,
        session_id: action.sessionId,
        role: "user",
        content: action.question,
        sources: [],
        rewritten_query: null,
        complexity: null,
        tier: null,
        timings: {},
        created_at: new Date().toISOString(),
        seq: userSeq,
      };
      const asstMsg: Message = {
        id: action.asstTempId,
        session_id: action.sessionId,
        role: "assistant",
        content: "",
        sources: [],
        rewritten_query: null,
        complexity: null,
        tier: null,
        timings: {},
        created_at: new Date().toISOString(),
        seq: asstSeq,
      };
      const next = new Map(state.optimistic);
      next.set(userMsg.id, userMsg);
      next.set(asstMsg.id, asstMsg);
      return {
        ...state,
        optimistic: next,
        asstTempId: action.asstTempId,
        messages: deriveMessages(
          serverMsgsFromIds(state),
          next,
        ),
      };
    }

    /* ─ Streaming token arrives ─────────────────────────────────── */
    case "STREAM_TOKEN": {
      const target = state.optimistic.get(action.asstTempId);
      if (!target) return state; // stale
      const updated = { ...target, content: action.content };
      const next = new Map(state.optimistic);
      next.set(action.asstTempId, updated);
      return {
        ...state,
        optimistic: next,
        messages: deriveMessages(serverMsgsFromIds(state), next),
      };
    }

    /* ─ Streaming metadata (sources, tier, etc.) ────────────────── */
    case "STREAM_META": {
      const target = state.optimistic.get(action.asstTempId);
      if (!target) return state;
      const updated = { ...target };
      if (action.sources !== undefined) updated.sources = action.sources;
      if (action.rewrittenQuery !== undefined) updated.rewritten_query = action.rewrittenQuery;
      if (action.complexity !== undefined) updated.complexity = action.complexity;
      if (action.tier !== undefined) updated.tier = action.tier;
      const next = new Map(state.optimistic);
      next.set(action.asstTempId, updated);
      return {
        ...state,
        optimistic: next,
        messages: deriveMessages(serverMsgsFromIds(state), next),
      };
    }

    /* ─ Server echoes real IDs ──────────────────────────────────── */
    case "REPLACE_TEMP_IDS": {
      const next = new Map(state.optimistic);
      const user = next.get(action.userTempId);
      if (user) {
        next.delete(action.userTempId);
        next.set(action.realUserId, { ...user, id: action.realUserId });
      }
      const asst = next.get(action.asstTempId);
      if (asst) {
        next.delete(action.asstTempId);
        next.set(action.realAsstId, { ...asst, id: action.realAsstId });
      }
      return {
        ...state,
        optimistic: next,
        asstTempId: action.realAsstId,
        messages: deriveMessages(serverMsgsFromIds(state), next),
      };
    }

    /* ─ Stream finished successfully ────────────────────────────── */
    case "FINALIZE_STREAM": {
      const target = state.optimistic.get(action.asstTempId);
      if (!target) return { ...state, asstTempId: null };
      // The reducer already holds the correct content from the last
      // STREAM_TOKEN.  Only overwrite if the action carries non-empty
      // content (legacy callers); otherwise preserve what's already there.
      const updated: Message = {
        ...target,
        content: action.content || target.content,
      };
      const next = new Map(state.optimistic);
      next.set(action.asstTempId, updated);
      return {
        ...state,
        optimistic: next,
        messages: deriveMessages(serverMsgsFromIds(state), next),
      };
    }

    /* ─ Backend is retrying generation — reset accumulated tokens ── */
    case "STREAM_RETRY": {
      const target = state.optimistic.get(action.asstTempId);
      if (!target) return state;
      const updated: Message = {
        ...target,
        content: "",
        sources: [],
        rewritten_query: null,
        complexity: null,
        tier: null,
      };
      const next = new Map(state.optimistic);
      next.set(action.asstTempId, updated);
      return {
        ...state,
        optimistic: next,
        messages: deriveMessages(serverMsgsFromIds(state), next),
      };
    }

    /* ─ Stream produced nothing — remove placeholder ────────────── */
    case "REMOVE_ASST_PLACEHOLDER": {
      const next = new Map(state.optimistic);
      next.delete(action.asstTempId);
      return {
        ...state,
        optimistic: next,
        asstTempId: null,
        messages: deriveMessages(serverMsgsFromIds(state), next),
      };
    }

    /* ─ SWR server data arrived — merge & GC optimistic ─────────── */
    case "SYNC_SERVER_MESSAGES": {
      // While streaming is active, skip the sync entirely to prevent
      // server data from overwriting the optimistic assistant placeholder
      // mid-stream (causes visible content mutation).
      if (state.asstTempId !== null) {
        return state;
      }

      const sIds = new Set(action.serverMessages.map((m) => m.id));
      // Bump globalSeq past the highest server seq so the next
      // optimistic SEND produces values that sort AFTER all existing
      // server messages.  Without this, a fresh page load starts
      // globalSeq at 0 and new optimistic messages sort BEFORE the
      // server messages (seq 1-N), causing a brief reorder flash.
      for (const m of action.serverMessages) {
        if (m.seq > globalSeq) globalSeq = m.seq;
      }
      // GC optimistic entries that server now owns
      const nextOpt = new Map<string | number, Message>();
      for (const [id, m] of state.optimistic) {
        if (!sIds.has(id)) nextOpt.set(id, m);
      }
      return {
        ...state,
        optimistic: nextOpt,
        serverIds: sIds,
        messages: deriveMessages(action.serverMessages, nextOpt),
      };
    }

    /* ─ Session changed — full reset ────────────────────────────── */
    case "SWITCH_SESSION": {
      return INITIAL_STATE;
    }
  }
}

/** Reconstruct server message array from the current derived list minus optimistic.
 *  This is only used internally when server data hasn't changed but optimistic has. */
function serverMsgsFromIds(state: StoreState): Message[] {
  return state.messages.filter((m) => state.serverIds.has(m.id));
}

const INITIAL_STATE: StoreState = {
  optimistic: new Map(),
  serverIds: new Set(),
  messages: [],
  asstTempId: null,
};

/* ── Hook ─────────────────────────────────────────────────────── */

export function useMessageStore() {
  const [state, dispatch] = useReducer(reducer, INITIAL_STATE);
  return {
    /** Merged, deduplicated, ordered messages. */
    messages: state.messages,
    /** Currently-streaming assistant message ID (for cursor indicator). */
    streamingAsstId: state.asstTempId,
    /** Dispatch a store action. */
    dispatch: useCallback(dispatch, [dispatch]),
  };
}

export type { StoreAction };
