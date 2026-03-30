"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { Send, Square, Paperclip, X, FileText, Loader2 } from "lucide-react";
import { uploadDocument } from "@/hooks/useDocuments";

const ALLOWED_EXTENSIONS = ["pdf", "docx", "txt"];
const MAX_FILE_SIZE = 20 * 1024 * 1024; // 20 MB

interface AttachedFile {
  file: File;
  status: "pending" | "uploading" | "done" | "error";
  error?: string;
}

interface MessageInputProps {
  onSend: (message: string) => void;
  isStreaming: boolean;
  onStop: () => void;
  sessionId: string | null;
}

export default function MessageInput({ onSend, isStreaming, onStop, sessionId }: MessageInputProps) {
  const [value, setValue] = useState("");
  const [files, setFiles] = useState<AttachedFile[]>([]);
  const [isDragging, setIsDragging] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Clear files when session changes
  useEffect(() => {
    setFiles([]);
  }, [sessionId]);

  // Auto-resize textarea
  useEffect(() => {
    const textarea = textareaRef.current;
    if (!textarea) return;
    textarea.style.height = "auto";
    textarea.style.height = Math.min(textarea.scrollHeight, 200) + "px";
  }, [value]);

  const validateAndAddFiles = useCallback((incoming: FileList | File[]) => {
    const newFiles: AttachedFile[] = [];
    for (const file of Array.from(incoming)) {
      const ext = file.name.split(".").pop()?.toLowerCase() || "";
      if (!ALLOWED_EXTENSIONS.includes(ext)) {
        newFiles.push({ file, status: "error", error: `Unsupported type: .${ext}` });
      } else if (file.size > MAX_FILE_SIZE) {
        newFiles.push({ file, status: "error", error: "File too large (max 20 MB)" });
      } else {
        newFiles.push({ file, status: "pending" });
      }
    }
    setFiles((prev) => [...prev, ...newFiles]);
  }, []);

  const handleFileSelect = () => {
    fileInputRef.current?.click();
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.length) {
      validateAndAddFiles(e.target.files);
      e.target.value = "";
    }
  };

  const removeFile = (index: number) => {
    setFiles((prev) => prev.filter((_, i) => i !== index));
  };

  // Drag-and-drop handlers
  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };
  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  };
  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files?.length) {
      validateAndAddFiles(e.dataTransfer.files);
    }
  };

  // Use a ref to avoid stale closures when accessing files inside async loops
  const filesRef = useRef(files);
  filesRef.current = files;

  const uploadPendingFiles = useCallback(async () => {
    if (!sessionId) return;
    const snapshot = filesRef.current;
    const pendingIndexes = snapshot
      .map((f, i) => (f.status === "pending" ? i : -1))
      .filter((i) => i >= 0);
    if (pendingIndexes.length === 0) return;

    // Mark all pending as uploading
    setFiles((prev) =>
      prev.map((f, i) =>
        pendingIndexes.includes(i) ? { ...f, status: "uploading" as const } : f,
      ),
    );

    for (const idx of pendingIndexes) {
      const fileToUpload = snapshot[idx].file;
      try {
        await uploadDocument(fileToUpload, undefined, sessionId);
        setFiles((prev) =>
          prev.map((f, i) => (i === idx ? { ...f, status: "done" as const } : f)),
        );
      } catch {
        setFiles((prev) =>
          prev.map((f, i) =>
            i === idx ? { ...f, status: "error" as const, error: "Upload failed" } : f,
          ),
        );
      }
    }
  }, [sessionId]);

  const handleSubmit = useCallback(async () => {
    const trimmed = value.trim();
    if (isStreaming) return;

    // Upload pending files first
    await uploadPendingFiles();

    if (!trimmed) return;
    onSend(trimmed);
    setValue("");
    // Clear all files after sending (done + error + any remaining)
    setFiles([]);
  }, [value, isStreaming, uploadPendingFiles, onSend]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const hasUploading = files.some((f) => f.status === "uploading");

  return (
    <div
      className="border-t border-border px-4 py-3"
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      <div className="mx-auto max-w-3xl">
        <div
          className={`rounded-xl border px-3 py-2 transition-colors focus-within:border-primary ${
            isDragging
              ? "border-primary bg-primary/5"
              : "border-border bg-muted/30"
          }`}
        >
          {/* File chips */}
          {files.length > 0 && (
            <div className="mb-2 flex flex-wrap gap-1.5">
              {files.map((f, i) => (
                <div
                  key={`${f.file.name}-${i}`}
                  className={`flex items-center gap-1.5 rounded-lg px-2.5 py-1 text-xs ${
                    f.status === "error"
                      ? "bg-destructive/10 text-destructive"
                      : f.status === "done"
                        ? "bg-green-500/10 text-green-400"
                        : "bg-muted text-muted-foreground"
                  }`}
                >
                  {f.status === "uploading" ? (
                    <Loader2 className="h-3 w-3 animate-spin" />
                  ) : (
                    <FileText className="h-3 w-3" />
                  )}
                  <span className="max-w-[140px] truncate">{f.file.name}</span>
                  {f.error && (
                    <span className="text-[10px]">({f.error})</span>
                  )}
                  <button
                    onClick={() => removeFile(i)}
                    className="ml-0.5 rounded-full p-0.5 hover:bg-foreground/10"
                    aria-label={`Remove ${f.file.name}`}
                  >
                    <X className="h-3 w-3" />
                  </button>
                </div>
              ))}
            </div>
          )}

          {/* Input row */}
          <div className="flex items-end gap-2">
            {/* File upload button */}
            <button
              type="button"
              onClick={handleFileSelect}
              disabled={!sessionId}
              className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg text-muted-foreground transition hover:bg-muted hover:text-foreground disabled:opacity-30 disabled:cursor-not-allowed"
              aria-label="Attach file"
              title={sessionId ? "Attach a document (PDF, DOCX, TXT)" : "Start a chat first to attach files"}
            >
              <Paperclip className="h-4 w-4" />
            </button>

            <input
              ref={fileInputRef}
              type="file"
              accept=".pdf,.docx,.txt"
              multiple
              onChange={handleFileChange}
              className="hidden"
            />

            <textarea
              ref={textareaRef}
              value={value}
              onChange={(e) => setValue(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask about Indian law…"
              rows={1}
              className="max-h-[200px] flex-1 resize-none bg-transparent text-sm outline-none placeholder:text-muted-foreground"
            />

            {isStreaming ? (
              <button
                onClick={onStop}
                className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-destructive text-destructive-foreground transition hover:brightness-110"
                aria-label="Stop streaming"
              >
                <Square className="h-3.5 w-3.5" />
              </button>
            ) : (
              <button
                onClick={handleSubmit}
                disabled={hasUploading || (!value.trim() && files.filter((f) => f.status === "pending").length === 0)}
                className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-primary text-primary-foreground transition hover:brightness-110 disabled:opacity-30 disabled:cursor-not-allowed"
                aria-label="Send message"
              >
                <Send className="h-3.5 w-3.5" />
              </button>
            )}
          </div>
        </div>

        {/* Drag overlay hint */}
        {isDragging && (
          <div className="mt-1 text-center text-xs text-primary">
            Drop files here
          </div>
        )}

        <p className="mt-1.5 text-center text-[10px] text-muted-foreground">
          AstraLex may produce inaccurate information. Always verify legal advice with a qualified professional.
        </p>
      </div>
    </div>
  );
}
