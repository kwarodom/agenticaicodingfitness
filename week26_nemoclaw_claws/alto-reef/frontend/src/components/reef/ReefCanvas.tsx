import { useCallback, useRef, useState } from "react";
import type { Landmark, Profile, Sandbox } from "@/api/types";
import { Avatar } from "./Avatar";
import { Enclosure } from "./Enclosure";
import { LandmarkGlyph } from "./Landmarks";
import { CENTER, LANDMARK_POS, SLOTS, VIEW, memberSpots } from "./geometry";

interface Props {
  sandboxes: Sandbox[]; profiles: Profile[]; landmarks: Landmark[];
  selectedProfile: string | null; onSelectProfile: (id: string | null) => void;
  onOpenSandbox: (id: string) => void; onAssign: (sandboxId: string, profileId: string) => void;
  onLandmarkHover: (l: Landmark | null, clientX: number, clientY: number) => void;
}

const PALMS = [{ x: 220, y: 150, s: 1 }, { x: 720, y: 140, s: .9 }, { x: 150, y: 470, s: .8 }, { x: 1050, y: 480, s: 1.1 }, { x: 660, y: 690, s: .8 }];
const CORAL = [{ x: 440, y: 200 }, { x: 700, y: 640 }, { x: 900, y: 420 }, { x: 250, y: 620 }, { x: 560, y: 270 }];

export function ReefCanvas({ sandboxes, profiles, landmarks, selectedProfile, onSelectProfile, onOpenSandbox, onAssign, onLandmarkHover }: Props) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [drag, setDrag] = useState<{ id: string; x: number; y: number } | null>(null);
  const [hoverEnclosure, setHoverEnclosure] = useState<string | null>(null);
  const byId = Object.fromEntries(profiles.map(p => [p.id, p]));

  const toView = (e: { clientX: number; clientY: number }) => {
    const svg = svgRef.current!; const r = svg.getBoundingClientRect();
    return { x: ((e.clientX - r.left) / r.width) * VIEW.w, y: ((e.clientY - r.top) / r.height) * VIEW.h };
  };
  const enclosureAt = (clientX: number, clientY: number) => {
    const el = document.elementFromPoint(clientX, clientY) as Element | null;
    return el?.closest("[data-enclosure]")?.getAttribute("data-enclosure") ?? null;
  };
  const startDrag = useCallback((id: string) => (e: React.PointerEvent) => {
    if (e.button !== 0) return; e.preventDefault();
    const start = toView(e); let moved = false; const startClient = { x: e.clientX, y: e.clientY };
    const move = (ev: PointerEvent) => {
      if (!moved && Math.hypot(ev.clientX - startClient.x, ev.clientY - startClient.y) < 4) return;
      moved = true; const v = toView(ev); setDrag({ id, x: v.x, y: v.y }); setHoverEnclosure(enclosureAt(ev.clientX, ev.clientY));
    };
    const up = (ev: PointerEvent) => {
      window.removeEventListener("pointermove", move); window.removeEventListener("pointerup", up);
      if (moved) { const target = enclosureAt(ev.clientX, ev.clientY); if (target) onAssign(target, id); }
      setDrag(null); setHoverEnclosure(null);
    };
    void start; window.addEventListener("pointermove", move); window.addEventListener("pointerup", up);
  }, [onAssign]);

  const unassigned = profiles.filter(p => !p.sandboxId || !sandboxes.some(s => s.id === p.sandboxId));
  const centreSpots = memberSpots(CENTER.x, CENTER.y + 4, unassigned.length);

  return (
    <svg ref={svgRef} viewBox={`0 0 ${VIEW.w} ${VIEW.h}`} className="w-full h-full select-none" role="application" aria-label="Reef map of sandboxes and claws"
         onClick={() => onSelectProfile(null)}>
      <defs>
        <radialGradient id="water" cx="50%" cy="45%" r="75%"><stop offset="0%" stopColor="#2aa7d6" /><stop offset="70%" stopColor="#1478b5" /><stop offset="100%" stopColor="#0d4f86" /></radialGradient>
        <radialGradient id="sand" cx="50%" cy="50%" r="60%"><stop offset="0%" stopColor="#f3e7cf" /><stop offset="100%" stopColor="#e2cfa7" /></radialGradient>
        <filter id="soft" x="-20%" y="-20%" width="140%" height="140%"><feGaussianBlur stdDeviation="6" /></filter>
      </defs>
      <rect width={VIEW.w} height={VIEW.h} fill="url(#water)" />
      {/* wave lines */}
      {[80, 160, 640, 700].map((y, i) => <path key={i} d={`M0,${y} q60,-8 120,0 t120,0 t120,0 t120,0 t120,0 t120,0 t120,0 t120,0 t120,0 t120,0`} stroke="#8fe0ff" strokeWidth={1} fill="none" opacity={0.25} />)}
      {/* shallows + island */}
      <path d="M170,330 C150,190 370,90 600,100 C880,110 1120,220 1090,410 C1060,600 830,690 560,680 C330,670 190,560 170,330 Z" fill="#5fd0ea" opacity={0.55} filter="url(#soft)" />
      <path d="M200,330 C185,205 385,120 600,128 C860,138 1090,235 1060,405 C1032,580 815,665 560,655 C350,646 215,545 200,330 Z" fill="url(#sand)" />
      <ellipse cx={650} cy={260} rx={40} ry={18} fill="#4cc1e0" opacity={0.75} />{/* tide pool */}
      {/* central platform */}
      <g transform={`translate(${CENTER.x},${CENTER.y})`}>
        <ellipse cx={0} cy={6} rx={92} ry={46} fill="#5b6b7c" /><ellipse cx={0} cy={0} rx={86} ry={40} fill="#7a8b9c" />
        <ellipse cx={0} cy={-2} rx={70} ry={30} fill="#8fa0b0" opacity={0.7} />
        <g transform="translate(0,-46)"><rect x={-26} y={-16} width={52} height={32} rx={16} fill="#0a1729" /><path d="M-13,0 l7,-8 h12 l7,8 -7,8 h-12 z" fill="#76b900" /></g>
        <text x={0} y={52} textAnchor="middle" fontSize={10} fill="#0a1729" fontWeight={600} opacity={0.75}>Unassigned claws</text>
      </g>
      {/* decorations */}
      {CORAL.map((c, i) => <g key={i} transform={`translate(${c.x},${c.y})`} opacity={0.8}>
        <path d="M0,0 l-4,-14 l-6,-6 M0,0 l2,-16 l6,-4 M0,0 l6,-10" stroke={i % 2 ? "#ff5c8a" : "#3fd0c9"} strokeWidth={2.4} fill="none" strokeLinecap="round" /></g>)}
      {PALMS.map((p, i) => <g key={i} transform={`translate(${p.x},${p.y}) scale(${p.s})`}>
        <path d="M0,0 q3,-24 -2,-48" stroke="#8b6b3e" strokeWidth={4} fill="none" strokeLinecap="round" />
        {[0, 60, 120, 180, 240, 300].map(a => <path key={a} d="M-2,-48 q14,-10 26,-2" stroke="#2e9c5a" strokeWidth={5} fill="none" strokeLinecap="round" transform={`rotate(${a} -2 -48)`} />)}
      </g>)}
      {/* landmarks */}
      {landmarks.map(l => LANDMARK_POS[l.id] && <LandmarkGlyph key={l.id} l={l} x={LANDMARK_POS[l.id].x} y={LANDMARK_POS[l.id].y}
        onHover={(lm) => { const pos = LANDMARK_POS[l.id]; const r = svgRef.current!.getBoundingClientRect();
          onLandmarkHover(lm, r.left + (pos.x / VIEW.w) * r.width, r.top + (pos.y / VIEW.h) * r.height); }} />)}
      {/* enclosures with members */}
      {sandboxes.map((s) => {
        const slot = SLOTS[s.slot % SLOTS.length];
        const members = s.members.map(id => byId[id]).filter(Boolean);
        const spots = memberSpots(0, 0, members.length);
        const dropTarget = drag ? hoverEnclosure === s.id : !!selectedProfile;
        return (
          <Enclosure key={s.id} s={s} x={slot.x} y={slot.y} dropTarget={dropTarget}
                     onOpen={() => onOpenSandbox(s.id)} onDrop={() => selectedProfile && onAssign(s.id, selectedProfile)}>
            {members.map((p, i) => <Avatar key={p.id} p={p} x={spots[i].x} y={spots[i].y} labelAbove={spots[i].labelAbove} selected={selectedProfile === p.id} dragging={drag?.id === p.id}
                                          onSelect={() => onSelectProfile(selectedProfile === p.id ? null : p.id)} onPointerDown={startDrag(p.id)} />)}
          </Enclosure>
        );
      })}
      {/* unassigned on the platform */}
      {unassigned.map((p, i) => <Avatar key={p.id} p={p} x={centreSpots[i].x} y={centreSpots[i].y} labelAbove={centreSpots[i].labelAbove} selected={selectedProfile === p.id} dragging={drag?.id === p.id}
                                       onSelect={() => onSelectProfile(selectedProfile === p.id ? null : p.id)} onPointerDown={startDrag(p.id)} />)}
      {/* drag ghost */}
      {drag && byId[drag.id] && <g transform={`translate(${drag.x},${drag.y})`} pointerEvents="none" opacity={0.95}>
        <Avatar p={byId[drag.id]} x={0} y={0} selected={false} dragging={false} onSelect={() => {}} onPointerDown={() => {}} /></g>}
    </svg>
  );
}
