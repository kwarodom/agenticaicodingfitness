import type { Sandbox } from "@/api/types";
import { FLOOR, STATUS_COLOR, floorPoints } from "./geometry";
import { clsx } from "clsx";

export function Enclosure({ s, x, y, dropTarget, onOpen, onDrop, children }:
  { s: Sandbox; x: number; y: number; dropTarget: boolean; onOpen: () => void; onDrop: () => void; children?: React.ReactNode }) {
  const c = STATUS_COLOR[s.status]; const { hw, hh, wall } = FLOOR;
  const glass = "rgba(160,220,255,0.16)"; const edge = "rgba(200,235,255,0.55)";
  return (
    <g data-enclosure={s.id} transform={`translate(${x},${y})`} role="button" tabIndex={0}
       aria-label={`${s.name}, ${s.status}, ${s.members.length} claws`} className="cursor-pointer focus:outline-none"
       onClick={(e) => { e.stopPropagation(); dropTarget ? onDrop() : onOpen(); }}
       onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); dropTarget ? onDrop() : onOpen(); } }}>
      {/* status ring */}
      <ellipse cx={0} cy={0} rx={hw + 22} ry={hh + 16} fill="none" stroke={c} strokeWidth={dropTarget ? 3 : 2}
               opacity={dropTarget ? 1 : 0.8} className={clsx((s.status === "degraded" || dropTarget) && "ring-pulse")} strokeDasharray={dropTarget ? "6 4" : undefined} />
      {/* floor */}
      <polygon points={floorPoints(0, 0)} fill="#d9e6f2" stroke={edge} strokeWidth={1} />
      <polygon points={`0,${-hh+6} ${hw-12},0 0,${hh-6} ${-hw+12},0`} fill="#c3d6e8" opacity={0.7} />
      {/* back glass walls (drawn behind occupants) */}
      <polygon points={`${-hw},0 0,${-hh} 0,${-hh - wall} ${-hw},${-wall}`} fill={glass} stroke={edge} strokeWidth={0.8} />
      <polygon points={`0,${-hh} ${hw},0 ${hw},${-wall} 0,${-hh - wall}`} fill={glass} stroke={edge} strokeWidth={0.8} />
      {children}
      {/* front glass walls (translucent, over occupants) */}
      <polygon points={`${-hw},0 0,${hh} 0,${hh - wall} ${-hw},${-wall}`} fill="rgba(160,220,255,0.08)" stroke={edge} strokeWidth={0.8} />
      <polygon points={`0,${hh} ${hw},0 ${hw},${-wall} 0,${hh - wall}`} fill="rgba(160,220,255,0.08)" stroke={edge} strokeWidth={0.8} />
      {/* name plate */}
      <g transform={`translate(0,${hh + 30})`}>
        <rect x={-64} y={-11} width={128} height={22} rx={6} fill="#10213a" stroke={c} strokeWidth={1} opacity={0.95} />
        <circle cx={-52} cy={0} r={3.5} fill={c} />
        <text x={-44} y={4} fontSize={11} fontWeight={600} fill="#e6f1ff">{s.name.length > 16 ? s.name.slice(0, 15) + "…" : s.name}</text>
        <text x={58} y={4} fontSize={10} textAnchor="end" fill="#8fb0d0" className="mono">{s.members.length}</text>
      </g>
    </g>
  );
}
