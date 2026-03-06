"use client";

import useSWR, { mutate } from "swr";
import { apiGet, apiPost, apiDelete, apiPatch } from "@/lib/api";
import type { Session, SessionDetail } from "@/types";

const SESSIONS_KEY = "/chat/sessions";

function fetcher<T>(url: string) {
  return apiGet<T>(url);
}

export function useSessions() {
  const { data, error, isLoading } = useSWR<{ sessions: Session[] }>(
    SESSIONS_KEY,
    fetcher,
  );
  return { sessions: data?.sessions || [], error, isLoading };
}

export function useSessionDetail(sessionId: string | null) {
  const { data, error, isLoading } = useSWR<SessionDetail>(
    sessionId ? `${SESSIONS_KEY}/${sessionId}` : null,
    fetcher,
  );
  return { detail: data, error, isLoading };
}

export async function createSession(title?: string): Promise<Session> {
  const session = await apiPost<Session>(SESSIONS_KEY, title ? { title } : {});
  mutate(SESSIONS_KEY);
  return session;
}

export async function deleteSession(sessionId: string): Promise<void> {
  await apiDelete(`${SESSIONS_KEY}/${sessionId}`);
  mutate(SESSIONS_KEY);
}

export async function renameSession(sessionId: string, title: string): Promise<void> {
  await apiPatch(`${SESSIONS_KEY}/${sessionId}`, { title });
  mutate(SESSIONS_KEY);
}

export function refreshSessions() {
  mutate(SESSIONS_KEY);
}

export function refreshSessionDetail(sessionId: string) {
  mutate(`${SESSIONS_KEY}/${sessionId}`);
}
