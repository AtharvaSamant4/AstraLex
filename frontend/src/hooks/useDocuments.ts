"use client";

import useSWR, { mutate } from "swr";
import { apiGet, apiUpload, apiDelete } from "@/lib/api";
import type { Document, UploadResponse } from "@/types";

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

export async function uploadDocument(
  file: File,
  title?: string,
): Promise<UploadResponse> {
  const extras: Record<string, string> = {};
  if (title) extras.title = title;
  const result = await apiUpload<UploadResponse>(
    "/documents/upload",
    file,
    extras,
  );
  // Refresh after a delay to allow background processing
  setTimeout(() => mutate(DOCS_KEY), 3000);
  mutate(DOCS_KEY);
  return result;
}

export async function deleteDocument(docId: number): Promise<void> {
  await apiDelete(`/documents/${docId}`);
  mutate(DOCS_KEY);
}

export function refreshDocuments() {
  mutate(DOCS_KEY);
}
