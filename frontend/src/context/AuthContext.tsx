"use client";

import React, { createContext, useContext, useEffect, useState, useCallback } from "react";
import { apiPost } from "@/lib/api";
import type { AuthResponse } from "@/types";

interface AuthState {
  token: string | null;
  userId: number | null;
  email: string | null;
  isAuthenticated: boolean;
  isLoading: boolean;
}

interface AuthContextValue extends AuthState {
  login: (email: string, password: string) => Promise<void>;
  signup: (email: string, password: string) => Promise<void>;
  logout: () => void;
}

const AuthContext = createContext<AuthContextValue | null>(null);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [state, setState] = useState<AuthState>({
    token: null,
    userId: null,
    email: null,
    isAuthenticated: false,
    isLoading: true,
  });

  // Hydrate from localStorage on mount
  useEffect(() => {
    const token = localStorage.getItem("token");
    const userId = localStorage.getItem("userId");
    const email = localStorage.getItem("email");
    if (token && userId) {
      setState({
        token,
        userId: Number(userId),
        email,
        isAuthenticated: true,
        isLoading: false,
      });
    } else {
      setState((s) => ({ ...s, isLoading: false }));
    }
  }, []);

  const setAuth = useCallback((data: AuthResponse) => {
    localStorage.setItem("token", data.token);
    localStorage.setItem("userId", String(data.user_id));
    localStorage.setItem("email", data.email);
    setState({
      token: data.token,
      userId: data.user_id,
      email: data.email,
      isAuthenticated: true,
      isLoading: false,
    });
  }, []);

  const login = useCallback(
    async (email: string, password: string) => {
      const data = await apiPost<AuthResponse>("/auth/login", { email, password });
      setAuth(data);
    },
    [setAuth],
  );

  const signup = useCallback(
    async (email: string, password: string) => {
      const data = await apiPost<AuthResponse>("/auth/signup", { email, password });
      setAuth(data);
    },
    [setAuth],
  );

  const logout = useCallback(() => {
    localStorage.removeItem("token");
    localStorage.removeItem("userId");
    localStorage.removeItem("email");
    setState({
      token: null,
      userId: null,
      email: null,
      isAuthenticated: false,
      isLoading: false,
    });
  }, []);

  return (
    <AuthContext.Provider value={{ ...state, login, signup, logout }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used inside <AuthProvider>");
  return ctx;
}
