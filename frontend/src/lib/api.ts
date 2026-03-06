/* ── API client — handles base URL, JWT headers, JSON parsing ──── */

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

function getToken(): string | null {
  if (typeof window === "undefined") return null;
  return localStorage.getItem("token");
}

function authHeaders(): Record<string, string> {
  const token = getToken();
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (token) headers["Authorization"] = `Bearer ${token}`;
  return headers;
}

async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    const msg = body.detail || JSON.stringify(body);
    throw new Error(msg);
  }
  return res.json() as Promise<T>;
}

/* ── Generic HTTP helpers ─────────────────────────────────────── */

export async function apiGet<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, { headers: authHeaders() });
  return handleResponse<T>(res);
}

export async function apiPost<T>(path: string, body?: unknown): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: authHeaders(),
    body: body ? JSON.stringify(body) : undefined,
  });
  return handleResponse<T>(res);
}

export async function apiPatch<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "PATCH",
    headers: authHeaders(),
    body: JSON.stringify(body),
  });
  return handleResponse<T>(res);
}

export async function apiDelete<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "DELETE",
    headers: authHeaders(),
  });
  return handleResponse<T>(res);
}

/* ── Multipart upload (for documents) ──────────────────────────── */

export async function apiUpload<T>(path: string, file: File, extras?: Record<string, string>): Promise<T> {
  const form = new FormData();
  form.append("file", file);
  const token = getToken();
  const headers: Record<string, string> = {};
  if (token) headers["Authorization"] = `Bearer ${token}`;

  const url = new URL(`${API_BASE}${path}`);
  if (extras) {
    Object.entries(extras).forEach(([k, v]) => url.searchParams.set(k, v));
  }

  const res = await fetch(url.toString(), {
    method: "POST",
    headers,
    body: form,
  });
  return handleResponse<T>(res);
}

/* ── SSE stream helper ─────────────────────────────────────────── */

export function apiStream(path: string, body: unknown): {
  reader: ReadableStreamDefaultReader<Uint8Array>;
  abort: () => void;
} {
  const controller = new AbortController();
  const token = getToken();
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (token) headers["Authorization"] = `Bearer ${token}`;

  const fetchPromise = fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers,
    body: JSON.stringify(body),
    signal: controller.signal,
  });

  // We return a proxy reader that resolves the fetch lazily
  let readerRef: ReadableStreamDefaultReader<Uint8Array> | null = null;
  const lazyReader: ReadableStreamDefaultReader<Uint8Array> = {
    read: async () => {
      if (!readerRef) {
        const res = await fetchPromise;
        if (!res.body) throw new Error("No response body");
        readerRef = res.body.getReader();
      }
      return readerRef.read();
    },
    releaseLock: () => readerRef?.releaseLock(),
    cancel: async (reason) => {
      readerRef?.cancel(reason);
      controller.abort();
    },
    closed: Promise.resolve(undefined),
  };

  return { reader: lazyReader, abort: () => controller.abort() };
}

export { API_BASE };
