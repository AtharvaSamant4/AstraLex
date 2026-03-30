"use client";

import useSWR, { mutate } from "swr";
import { apiGet, apiUpload, apiDelete } from "@/lib/api";
import type { Document, DocumentStatus, UploadResponse } from "@/types";

const DOCS_KEY = "/documents";

function fetcher<T>(url: string) {
  return apiGet<T>(url);
}

export function useDocuments() {
  const { data, error, isLoading } = useSWR<{ documents: Document[] }>(
    DOCS_KEY,
    fetcher,
  );
  return { documents: data?.documents || [], error, isLoading };
}

/**
 * Returns true when a document's status allows retrieval.
 * Everything before "ready" is not searchable.
 */
export function isDocumentReady(doc: Document): boolean {
  return doc.status === "ready";
}

/**
 * Readable label for any document status.
 */
export function statusLabel(status: DocumentStatus): string {
  switch (status) {
    case "uploading":
      return "Uploading…";
    case "processing":
      return "Indexing…";
    case "ready":
      return "Ready";
    case "failed":
      return "Failed";
  }
}

/* ── Polling FSM ──────────────────────────────────────────────── */

/** Active poll timers keyed by document ID — prevents duplicate polls. */
const activePolls = new Map<number, ReturnType<typeof setInterval>>();

/**
 * Poll until a document transitions out of a non-terminal state
 * (uploading | processing) into a terminal state (ready | failed).
 * Each poll writes directly into SWR cache to keep UI reactive.
 */
function pollUntilTerminal(docId: number) {
  if (activePolls.has(docId)) return; // already polling

  const INTERVAL = 2_000;
  const MAX_DURATION = 120_000;

  const timer = setInterval(async () => {
    try {
      const data = await apiGet<{ documents: Document[] }>(DOCS_KEY);
      mutate(DOCS_KEY, data, { revalidate: false });
      const doc = data.documents.find((d) => d.id === docId);
      if (!doc || doc.status === "ready" || doc.status === "failed") {
        clearInterval(timer);
        activePolls.delete(docId);
      }
    } catch {
      clearInterval(timer);
      activePolls.delete(docId);
    }
  }, INTERVAL);

  activePolls.set(docId, timer);

  // Safety cut-off
  setTimeout(() => {
    if (activePolls.has(docId)) {
      clearInterval(activePolls.get(docId)!);
      activePolls.delete(docId);
    }
  }, MAX_DURATION);
}

/* ── Mutations ────────────────────────────────────────────────── */

export async function uploadDocument(
  file: File,
  title?: string,
  sessionId?: string,
): Promise<UploadResponse> {
  const extras: Record<string, string> = {};
  if (title) extras.title = title;
  if (sessionId) extras.session_id = sessionId;

  const result = await apiUpload<UploadResponse>(
    "/documents/upload",
    file,
    extras,
  );
  // Immediately refresh to show the new entry (status: processing from server)
  mutate(DOCS_KEY);
  // Poll until terminal
  pollUntilTerminal(result.document_id);
  return result;
}

export async function deleteDocument(docId: number): Promise<void> {
  // Stop any active poll for this doc
  if (activePolls.has(docId)) {
    clearInterval(activePolls.get(docId)!);
    activePolls.delete(docId);
  }
  await apiDelete(`/documents/${docId}`);
  mutate(DOCS_KEY);
}

export function refreshDocuments() {
  mutate(DOCS_KEY);
}
