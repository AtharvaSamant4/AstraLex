"use client";

import { useState, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Plus,
  MessageSquare,
  Trash2,
  Pencil,
  Check,
  X,
  LogOut,
  Scale,
  ChevronLeft,
} from "lucide-react";
import { useAuth } from "@/context/AuthContext";
import { useSessions, createSession, deleteSession, renameSession } from "@/hooks/useChat";

interface SidebarProps {
  activeSessionId: string | null;
  onSelectSession: (id: string) => void;
  onNewChat: (id: string) => void;
  isOpen: boolean;
  onToggle: () => void;
}

export default function Sidebar({
  activeSessionId,
  onSelectSession,
  onNewChat,
  isOpen,
  onToggle,
}: SidebarProps) {
  const { email, logout } = useAuth();
  const { sessions, isLoading } = useSessions();
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editTitle, setEditTitle] = useState("");
  const editInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (editingId && editInputRef.current) {
      editInputRef.current.focus();
      editInputRef.current.select();
    }
  }, [editingId]);

  const handleNew = async () => {
    try {
      const session = await createSession();
      onNewChat(session.id);
    } catch {
      /* silent */
    }
  };

  const handleDelete = async (id: string) => {
    try {
      await deleteSession(id);
      if (activeSessionId === id) onSelectSession("");
    } catch {
      /* silent */
    }
  };

  const handleRenameStart = (id: string, currentTitle: string | null) => {
    setEditingId(id);
    setEditTitle(currentTitle || "");
  };

  const handleRenameConfirm = async () => {
    if (editingId && editTitle.trim()) {
      try {
        await renameSession(editingId, editTitle.trim());
      } catch {
        /* silent */
      }
    }
    setEditingId(null);
  };

  const handleRenameCancel = () => {
    setEditingId(null);
  };

  const sortedSessions = [...sessions].sort(
    (a, b) => new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime(),
  );

  return (
    <>
      <AnimatePresence mode="wait">
        {isOpen && (
          <motion.aside
            key="sidebar"
            initial={{ width: 0, opacity: 0 }}
            animate={{ width: 280, opacity: 1 }}
            exit={{ width: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="relative flex h-full flex-col border-r border-border bg-card overflow-hidden"
          >
            {/* Header */}
            <div className="flex items-center justify-between px-4 py-3 border-b border-border">
              <div className="flex items-center gap-2">
                <Scale className="h-5 w-5 text-primary" />
                <span className="font-semibold text-sm gradient-text">AstraLex</span>
              </div>
              <button
                onClick={onToggle}
                className="rounded-md p-1 text-muted-foreground hover:bg-muted transition-colors"
                aria-label="Close sidebar"
              >
                <ChevronLeft className="h-4 w-4" />
              </button>
            </div>

            {/* New Chat */}
            <div className="px-3 py-3">
              <button
                onClick={handleNew}
                className="flex w-full items-center gap-2 rounded-lg border border-border px-3 py-2 text-sm font-medium hover:bg-muted transition-colors"
              >
                <Plus className="h-4 w-4" />
                New Chat
              </button>
            </div>

            {/* Sessions */}
            <div className="flex-1 overflow-y-auto px-3 scrollbar-thin">
              {isLoading ? (
                <div className="space-y-2 py-2">
                  {[...Array(4)].map((_, i) => (
                    <div key={i} className="h-9 animate-pulse rounded-lg bg-muted" />
                  ))}
                </div>
              ) : sortedSessions.length === 0 ? (
                <p className="px-3 py-6 text-center text-xs text-muted-foreground">
                  No conversations yet. Start one!
                </p>
              ) : (
                <div className="space-y-0.5 py-1">
                  {sortedSessions.map((session) => {
                    const isActive = session.id === activeSessionId;
                    const isEditing = session.id === editingId;

                    return (
                      <div
                        key={session.id}
                        className={`group flex items-center gap-1 rounded-lg px-2 py-2 text-sm transition-colors ${
                          isActive
                            ? "bg-primary/10 text-primary"
                            : "text-muted-foreground hover:bg-muted hover:text-foreground"
                        }`}
                      >
                        <MessageSquare className="h-4 w-4 shrink-0" />

                        {isEditing ? (
                          <div className="flex flex-1 items-center gap-1 min-w-0">
                            <input
                              ref={editInputRef}
                              value={editTitle}
                              onChange={(e) => setEditTitle(e.target.value)}
                              onKeyDown={(e) => {
                                if (e.key === "Enter") handleRenameConfirm();
                                if (e.key === "Escape") handleRenameCancel();
                              }}
                              className="min-w-0 flex-1 rounded border border-border bg-background px-1.5 py-0.5 text-xs focus:outline-none focus:border-primary"
                            />
                            <button
                              onClick={handleRenameConfirm}
                              className="p-0.5 text-green-400 hover:text-green-300"
                            >
                              <Check className="h-3.5 w-3.5" />
                            </button>
                            <button
                              onClick={handleRenameCancel}
                              className="p-0.5 text-red-400 hover:text-red-300"
                            >
                              <X className="h-3.5 w-3.5" />
                            </button>
                          </div>
                        ) : (
                          <>
                            <button
                              onClick={() => onSelectSession(session.id)}
                              className="flex-1 truncate text-left text-xs"
                            >
                              {session.title || "Untitled Chat"}
                            </button>
                            <div className="hidden shrink-0 items-center gap-0.5 group-hover:flex">
                              <button
                                onClick={(e) => {
                                  e.stopPropagation();
                                  handleRenameStart(session.id, session.title);
                                }}
                                className="rounded p-0.5 hover:bg-accent"
                              >
                                <Pencil className="h-3 w-3" />
                              </button>
                              <button
                                onClick={(e) => {
                                  e.stopPropagation();
                                  handleDelete(session.id);
                                }}
                                className="rounded p-0.5 hover:bg-destructive/20 text-destructive"
                              >
                                <Trash2 className="h-3 w-3" />
                              </button>
                            </div>
                          </>
                        )}
                      </div>
                    );
                  })}
                </div>
              )}
            </div>

            {/* User footer */}
            <div className="border-t border-border px-4 py-3">
              <div className="flex items-center justify-between">
                <div className="min-w-0">
                  <p className="truncate text-xs font-medium">
                    {email || "User"}
                  </p>
                </div>
                <button
                  onClick={logout}
                  className="rounded-md p-1.5 text-muted-foreground hover:bg-destructive/10 hover:text-destructive transition-colors"
                  aria-label="Sign out"
                >
                  <LogOut className="h-4 w-4" />
                </button>
              </div>
            </div>
          </motion.aside>
        )}
      </AnimatePresence>
    </>
  );
}
