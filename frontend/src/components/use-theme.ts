import React from "react";

type Theme = "light" | "dark" | "system";

type ThemeContextType = {
  theme: Theme;
  setTheme: (t: Theme) => void;
};

const ThemeContext = React.createContext<ThemeContextType | undefined>(
  undefined
);

export function useTheme(): ThemeContextType {
  const ctx = React.useContext(ThemeContext);
  if (!ctx) throw new Error("useTheme must be used within ThemeProvider");
  return ctx;
}

export { ThemeContext };
export type { Theme, ThemeContextType };
