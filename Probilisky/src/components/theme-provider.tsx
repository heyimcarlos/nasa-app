// components/theme-provider.tsx
"use client";

import React from "react";

type Theme = "light" | "dark" | "system";

type ThemeContextType = {
  theme: Theme;
  setTheme: (t: Theme) => void;
};

const ThemeContext = React.createContext<ThemeContextType | undefined>(undefined);

function getSystemTheme(): "light" | "dark" {
  if (typeof window === "undefined") return "light";
  return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

function applyTheme(theme: Theme) {
  const resolved = theme === "system" ? getSystemTheme() : theme;
  const root = document.documentElement;
  root.setAttribute("data-theme", resolved);
}

export function ThemeProvider({
  children,
  defaultTheme = "system",
  enableSystem = true,
}: {
  children: React.ReactNode;
  defaultTheme?: Theme;
  enableSystem?: boolean;
}) {
  const [theme, setThemeState] = React.useState<Theme>(() => {
    if (typeof window === "undefined") return defaultTheme;
    const stored = localStorage.getItem("theme") as Theme | null;
    return stored ?? defaultTheme;
  });

  React.useEffect(() => {
    applyTheme(theme);
    localStorage.setItem("theme", theme);
  }, [theme]);

  React.useEffect(() => {
    if (!enableSystem) return;
    const m = window.matchMedia("(prefers-color-scheme: dark)");
    const handler = () => {
      const stored = (localStorage.getItem("theme") as Theme | null) ?? "system";
      if (stored === "system") applyTheme("system");
    };
    m.addEventListener?.("change", handler);
    return () => m.removeEventListener?.("change", handler);
  }, [enableSystem]);

  const setTheme = React.useCallback((t: Theme) => setThemeState(t), []);
  const value = React.useMemo(() => ({ theme, setTheme }), [theme, setTheme]);

  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>;
}

export function useTheme(): ThemeContextType {
  const ctx = React.useContext(ThemeContext);
  if (!ctx) throw new Error("useTheme must be used within ThemeProvider");
  return ctx;
}
