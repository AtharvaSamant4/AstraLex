/* ── Shared type definitions ────────────────────────────────────── */

export interface AuthResponse {
  token: string;
  user_id: number;
  email: string;
}

export interface Session {
  id: string;
  user_id: number;
  title: string | null;
  created_at: string;
  updated_at: string;
}

export interface Message {
  id: number | string;
  session_id: string;
  role: "user" | "assistant";
  content: string;
  sources: string[];
  rewritten_query: string | null;
  complexity: string | null;
  tier: string | null;
  timings: Record<string, number>;
  created_at: string;
  /** Sequence number for deterministic ordering (monotonic within session). */
  seq: number;
}

export interface ChatMessageResponse {
  answer: string;
  sources: string[];
  rewritten_query: string;
  verified: boolean;
  timings: Record<string, number>;
  research_plan: Record<string, unknown> | null;
  evidence_graph_stats: Record<string, number> | null;
  complexity: string;
  retrieval_iterations: number;
  follow_up_queries: string[];
  tier: string;
  user_message_id: number;
  assistant_message_id: number;
}

export interface SessionDetail {
  session: Session;
  messages: Message[];
  documents: Document[];
}

/** Document processing FSM: uploading → processing → ready | failed */
export type DocumentStatus = "uploading" | "processing" | "ready" | "failed";

export interface Document {
  id: number;
  user_id: number;
  filename: string;
  file_type: string;
  title: string;
  total_chunks: number;
  status: DocumentStatus;
  created_at: string;
  session_id: string | null;
}

export interface SystemMetrics {
  total_queries: number;
  average_latency_ms: number;
  p95_latency_ms: number;
  hallucination_rate: number;
  total_feedback: number;
  negative_feedback: number;
  retrieval_success_rate: number;
  active_users: number;
  queries_per_user: number;
  total_documents_uploaded: number;
  total_document_chunks: number;
  tier_breakdown: Record<string, number>;
  latency_trend_24h: Array<{
    hour: string;
    queries: number;
    avg_latency_ms: number;
  }>;
}

export interface UploadResponse {
  document_id: number;
  filename: string;
  status: string;
  session_id: string | null;
  message: string;
}
