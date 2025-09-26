import { create } from "zustand";

interface ChartOptionsState {
  model: string;
  setModel: (model: string) => void;
  currency: string;
  setCurrency: (currency: string) => void;
}

export const useChartOptionsStore = create<ChartOptionsState>((set) => ({
  model: "lstm_attention",
  setModel: (model) => set({ model }),
  currency: "usd",
  setCurrency: (currency) => set({ currency }),
}));
