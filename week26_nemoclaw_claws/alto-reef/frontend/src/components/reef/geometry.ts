import type { Landmark, SandboxStatus } from "@/api/types";
export const VIEW = { w: 1200, h: 720 };
/** Fixed enclosure slots on the island (floor centre in viewBox units). */
export const SLOTS: { x: number; y: number }[] = [
  { x: 330, y: 290 }, { x: 790, y: 250 }, { x: 300, y: 520 }, { x: 820, y: 520 }, { x: 560, y: 585 }, { x: 560, y: 160 },
];
export const CENTER = { x: 590, y: 400 };
export const LANDMARK_POS: Record<string, { x: number; y: number }> = {
  gateway: { x: 130, y: 210 }, vllm: { x: 860, y: 78 }, nat: { x: 985, y: 165 }, phoenix: { x: 470, y: 62 }, otel: { x: 1120, y: 120 },
};
export const STATUS_COLOR: Record<SandboxStatus, string> = {
  running: "#3fd0c9", degraded: "#f2b64c", error: "#ff5c8a", stopped: "#8fb0d0", unknown: "#8fb0d0",
};
export const landmarkColor = (l: Landmark) => (l.ok === null ? "#8fb0d0" : l.ok ? "#3fd0c9" : "#ff5c8a");
/** Isometric floor diamond for an enclosure centred at (x,y). */
export const FLOOR = { hw: 95, hh: 46, wall: 38 };
export function floorPoints(x: number, y: number) {
  const { hw, hh } = FLOOR;
  return `${x},${y - hh} ${x + hw},${y} ${x},${y + hh} ${x - hw},${y}`;
}
/** Positions for n members spread across the floor. */
export function memberSpots(x: number, y: number, n: number) {
  if (n === 0) return [] as { x: number; y: number; labelAbove: boolean }[];
  const spread = Math.min(FLOOR.hw * 1.5, 52 * (n - 1));
  return Array.from({ length: n }, (_, i) => ({ x: x - spread / 2 + (n === 1 ? 0 : (spread * i) / (n - 1)), y: y + (i % 2 ? 10 : -8), labelAbove: i % 2 === 1 }));
}
