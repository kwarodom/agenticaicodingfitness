import type { Profile } from "@/api/types";
import { clsx } from "clsx";

/** Simple silhouettes per harness: crab (OpenClaw), lobster (Hermes), octopus (Deep Agents), nautilus (NAT). */
export function AvatarShape({ p, r = 13 }: { p: Profile; r?: number }) {
  const c = p.color; const dark = "#0a1729";
  switch (p.harness) {
    case "hermes": // lobster: long body + tail fan
      return (<g>
        <ellipse cx={0} cy={0} rx={r * 1.35} ry={r * 0.6} fill={c} />
        <path d={`M${r*1.2},0 l${r*0.9},-${r*0.55} v${r*1.1} z`} fill={c} opacity={0.85} />
        <circle cx={-r * 1.5} cy={-r * 0.45} r={r * 0.42} fill={c} /><circle cx={-r * 1.5} cy={r * 0.45} r={r * 0.42} fill={c} />
        <circle cx={-r * 0.8} cy={-r * 0.22} r={r * 0.14} fill={dark} /><circle cx={-r * 0.8} cy={r * 0.22} r={r * 0.14} fill={dark} />
      </g>);
    case "deepagents": // octopus: dome + tentacles
      return (<g>
        <path d={`M${-r},${r*0.2} a${r},${r} 0 0 1 ${2*r},0 z`} fill={c} />
        {[-0.8,-0.4,0,0.4,0.8].map((k,i)=><path key={i} d={`M${k*r},${r*0.2} q${k*r*0.3},${r*0.6} ${k*r*0.5+ (i%2?4:-4)},${r*0.9}`} stroke={c} strokeWidth={r*0.28} fill="none" strokeLinecap="round" />)}
        <circle cx={-r * 0.35} cy={-r * 0.25} r={r * 0.16} fill={dark} /><circle cx={r * 0.35} cy={-r * 0.25} r={r * 0.16} fill={dark} />
      </g>);
    case "nat": // nautilus: spiral shell + small head
      return (<g>
        <circle cx={0} cy={0} r={r} fill={c} />
        <path d={`M0,0 m-${r*0.7},0 a${r*0.7},${r*0.7} 0 1 1 ${r*0.7},-${r*0.7} a${r*0.42},${r*0.42} 0 1 1 ${r*0.42},${r*0.42} a${r*0.2},${r*0.2} 0 1 1 -${r*0.2},${r*0.2}`} stroke={dark} strokeWidth={1.6} fill="none" opacity={0.7} />
        <ellipse cx={r * 1.05} cy={r * 0.25} rx={r * 0.5} ry={r * 0.36} fill={c} opacity={0.9} /><circle cx={r * 1.2} cy={r * 0.15} r={r * 0.12} fill={dark} />
      </g>);
    default: // crab
      return (<g>
        <ellipse cx={0} cy={0} rx={r} ry={r * 0.68} fill={c} />
        <circle cx={-r * 1.15} cy={-r * 0.35} r={r * 0.36} fill={c} /><circle cx={r * 1.15} cy={-r * 0.35} r={r * 0.36} fill={c} />
        {[-0.75,-0.35,0.35,0.75].map((k,i)=><line key={i} x1={k*r} y1={r*0.4} x2={k*r*1.35} y2={r*0.95} stroke={c} strokeWidth={2} strokeLinecap="round" />)}
        <circle cx={-r * 0.32} cy={-r * 0.55} r={r * 0.17} fill="#fff" /><circle cx={r * 0.32} cy={-r * 0.55} r={r * 0.17} fill="#fff" />
        <circle cx={-r * 0.32} cy={-r * 0.55} r={r * 0.08} fill={dark} /><circle cx={r * 0.32} cy={-r * 0.55} r={r * 0.08} fill={dark} />
      </g>);
  }
}

export function Avatar({ p, x, y, selected, dragging, onSelect, onPointerDown, labelAbove = false }:
  { p: Profile; x: number; y: number; selected: boolean; dragging: boolean; onSelect: () => void; onPointerDown: (e: React.PointerEvent) => void; labelAbove?: boolean }) {
  return (
    <g transform={`translate(${x},${y})`} data-profile={p.id} role="button" tabIndex={0} aria-label={`${p.name}, ${p.archetype}, ${p.harness}`}
       className={clsx("cursor-grab focus:outline-none", dragging && "cursor-grabbing")}
       onClick={(e) => { e.stopPropagation(); onSelect(); }} onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); onSelect(); } }}
       onPointerDown={onPointerDown} style={{ opacity: dragging ? 0.35 : 1 }}>
      {selected && <ellipse cx={0} cy={8} rx={26} ry={12} fill="none" stroke="#fff" strokeWidth={1.5} strokeDasharray="4 3" className="ring-pulse" />}
      <ellipse cx={0} cy={10} rx={14} ry={4} fill="#000" opacity={0.18} />
      <g className="bob"><AvatarShape p={p} /></g>
      <text x={0} y={labelAbove ? -22 : 27} textAnchor="middle" fontSize={11} fontWeight={600} fill="#e6f1ff" style={{ paintOrder: "stroke", stroke: "#0a1729", strokeWidth: 3 }}>{p.name}</text>
    </g>
  );
}
