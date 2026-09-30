import type { Landmark } from "@/api/types";
import { landmarkColor } from "./geometry";

export function LandmarkGlyph({ l, x, y, onHover }: { l: Landmark; x: number; y: number; onHover: (l: Landmark | null, x: number, y: number) => void }) {
  const c = landmarkColor(l);
  const common = { onPointerEnter: () => onHover(l, x, y), onPointerLeave: () => onHover(null, 0, 0), onFocus: () => onHover(l, x, y), onBlur: () => onHover(null, 0, 0) };
  return (
    <g transform={`translate(${x},${y})`} tabIndex={0} aria-label={`${l.label}: ${l.ok ? "ok" : l.ok === false ? "down" : "unknown"}`} className="cursor-help focus:outline-none" {...common}>
      {l.kind === "lighthouse" && (<g>
        <ellipse cx={0} cy={26} rx={34} ry={12} fill="#7f8c9a" /><ellipse cx={0} cy={22} rx={26} ry={9} fill="#98a6b3" />
        <path d="M-10,22 L-6,-30 L6,-30 L10,22 Z" fill="#e9f0f6" /><rect x={-10} y={-2} width={20} height={7} fill="#ff5c8a" opacity={0.85} />
        <rect x={-7} y={-40} width={14} height={11} fill="#16304f" stroke="#e9f0f6" strokeWidth={1} />
        <circle cx={0} cy={-35} r={4} fill={c} className={l.ok ? "ring-pulse" : undefined} />
        <path d="M-7,-40 L0,-48 L7,-40 Z" fill="#e9f0f6" />
      </g>)}
      {l.kind === "reactor" && (<g>
        <path d="M-34,14 L-20,-14 L8,-22 L34,-6 L28,18 L-6,26 Z" fill="#5b6b7c" /><path d="M-20,-14 L8,-22 L34,-6 L10,4 Z" fill="#7a8b9c" />
        <circle cx={6} cy={0} r={9} fill={c} opacity={0.9} className={l.ok ? "ring-pulse" : undefined} />
        <text x={6} y={4} textAnchor="middle" fontSize={9} fontWeight={700} fill="#0a1729">GPU</text>
      </g>)}
      {l.kind === "nautilus" && (<g>
        <circle cx={0} cy={0} r={20} fill="#e6d3b3" stroke="#b39a73" strokeWidth={1} />
        <path d="M-14,0 a14,14 0 1 1 14,-14 a8,8 0 1 1 8,8 a4,4 0 1 1 -4,4" stroke="#8b7250" strokeWidth={1.6} fill="none" />
        <circle cx={14} cy={12} r={5} fill={c} />
      </g>)}
      {l.kind === "buoy" && (<g>
        <ellipse cx={0} cy={12} rx={16} ry={5} fill="#0a1729" opacity={0.25} />
        <path d="M-9,10 L-6,-6 L6,-6 L9,10 Z" fill="#e9f0f6" /><rect x={-7} y={-2} width={14} height={5} fill="#4aa3ff" />
        <line x1={0} y1={-6} x2={0} y2={-18} stroke="#e9f0f6" strokeWidth={2} /><circle cx={0} cy={-20} r={4} fill={c} className={l.ok ? "ring-pulse" : undefined} />
      </g>)}
      <text x={0} y={46} textAnchor="middle" fontSize={10} fontWeight={600} fill="#e6f1ff" style={{ paintOrder: "stroke", stroke: "#0a1729", strokeWidth: 3 }}>{l.label}</text>
    </g>
  );
}
