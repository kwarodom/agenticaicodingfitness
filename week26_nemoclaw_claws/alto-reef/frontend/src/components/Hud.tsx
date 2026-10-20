import { useState } from "react";
import { clsx } from "clsx";
import { Activity, Blocks, Hammer, LayoutGrid, MonitorPlay, RefreshCw, RotateCcw, X } from "lucide-react";
import type { Health, Landmark, Profile, ReefEvent, Sandbox } from "@/api/types";
import { STRINGS, type Lang } from "@/i18n/strings";
import { AvatarShape } from "./reef/Avatar";

export const chipFor = (ok: boolean | null | undefined) => ok === true ? "chip chip-teal" : ok === false ? "chip chip-magenta" : "chip chip-grey";
export const ago = (t?: number | null) => { if (!t) return "—"; const s = Math.max(0, Math.round(Date.now() / 1000 - t)); return s < 60 ? `${s}s` : s < 3600 ? `${Math.round(s / 60)}m` : `${Math.round(s / 3600)}h`; };

export function Toolbar({ lang, setLang, kiosk, setKiosk, onRefresh, onReset, onManager, onBuild }:
  { lang: Lang; setLang: (l: Lang) => void; kiosk: boolean; setKiosk: (k: boolean) => void; onRefresh: () => void; onReset: () => void; onManager: () => void; onBuild: () => void }) {
  const t = STRINGS[lang];
  return (
    <div className="flex flex-wrap items-center gap-1.5 justify-end">
      <button className="btn" onClick={onRefresh} data-testid="btn-refresh"><RefreshCw size={13} />{t.refresh}</button>
      {!kiosk && <button className="btn btn-danger" onClick={onReset} data-testid="btn-reset"><RotateCcw size={13} />{t.resetBooth}</button>}
      <button className={clsx("btn", kiosk && "btn-primary")} onClick={() => setKiosk(!kiosk)} data-testid="btn-kiosk"><MonitorPlay size={13} />{t.kiosk}</button>
      <button className="btn" onClick={onManager} data-testid="btn-manager"><LayoutGrid size={13} />{t.manager}</button>
      {!kiosk && <button className="btn btn-primary" onClick={onBuild} data-testid="btn-build"><Hammer size={13} />{t.buildClaw}</button>}
      <button className="btn mono" onClick={() => setLang(lang === "en" ? "th" : "en")} data-testid="btn-lang">{lang.toUpperCase()}</button>
    </div>
  );
}

export function HealthTiles({ h, lang }: { h: Health | undefined; lang: Lang }) {
  const t = STRINGS[lang];
  const gw = h?.gateway; const inf = h?.inference;
  return (
    <div className="grid grid-cols-2 gap-2">
      <div className="panel px-3 py-2"><div className="label">{t.gateway}</div>
        <div className="flex items-center gap-2 mt-1"><span className={chipFor(gw?.ok)}>{gw ? (gw.ok ? t.healthy : t.unreachable) : "…"}</span><span className="text-[10px] text-muted mono truncate">{gw?.detail}</span></div></div>
      <div className="panel px-3 py-2"><div className="label">{t.inference}</div>
        <div className="flex items-center gap-2 mt-1"><span className={clsx("chip", inf?.model ? "chip-nv" : "chip-grey")}>{inf?.provider ?? "—"}</span><span className="text-[11px] mono truncate" title={inf?.model ?? ""}>{inf?.model ?? "no model set"}</span></div></div>
    </div>
  );
}

export function StatusBanner({ h, lang }: { h: Health | undefined; lang: Lang }) {
  const t = STRINGS[lang];
  if (!h) return <div className="panel px-3 py-1.5 text-[12px] text-muted">Probing runner…</div>;
  const msgs: { cls: string; text: string }[] = [];
  if (h.mode === "mock") msgs.push({ cls: "border-amber/70 text-amber", text: t.bannerMock });
  if (!h.gateway?.ok) msgs.push({ cls: "border-magenta/70 text-magenta", text: t.bannerGatewayDown });
  const vllm = h.landmarks.find(l => l.id === "vllm"); if (vllm && vllm.ok === false) msgs.push({ cls: "border-magenta/70 text-magenta", text: t.bannerInferenceDown });
  if (!msgs.length) return null;
  return <div className="flex flex-col gap-1">{msgs.map((m, i) => <div key={i} className={clsx("panel px-3 py-1.5 text-[12px] font-medium", m.cls)} role="status">{m.text}</div>)}</div>;
}

export function LandmarkTip({ l, x, y }: { l: Landmark; x: number; y: number }) {
  return (
    <div className="pointer-events-none fixed z-40 panel px-3 py-2 text-[12px] -translate-x-1/2 -translate-y-full" style={{ left: x, top: y - 56 }}>
      <div className="flex items-center gap-2"><span className={chipFor(l.ok)}>{l.ok ? "OK" : l.ok === false ? "DOWN" : "?"}</span><span className="font-semibold">{l.label}</span></div>
      <div className="mono text-muted mt-1 max-w-[260px] break-words">{l.detail || "—"}</div>
      <div className="text-[10px] text-muted mt-1">probed {ago(l.probedAt)} ago{l.url ? ` · ${l.url}` : ""}</div>
    </div>
  );
}

export function ProfileCard({ p, sandbox, lang, onClose, onRemove, kiosk }: { p: Profile; sandbox?: Sandbox; lang: Lang; onClose: () => void; onRemove: () => void; kiosk: boolean }) {
  const t = STRINGS[lang];
  return (
    <div className="panel p-3 w-[300px]" data-testid="profile-card">
      <div className="flex items-start gap-3">
        <svg width={44} height={44} viewBox="-22 -22 44 44"><AvatarShape p={p} r={12} /></svg>
        <div className="flex-1 min-w-0">
          <div className="flex items-center justify-between"><div className="font-semibold text-[14px]">{p.name}</div><button className="text-muted hover:text-ink" onClick={onClose} aria-label="close"><X size={14} /></button></div>
          <div className="text-[11px] text-muted">{p.archetype.replace("_", " ")} · {p.harness} · <span className="mono">{p.tier}</span></div>
          <div className="text-[12px] mt-1">{p.persona}</div>
        </div>
      </div>
      <div className="label mt-3">skills</div>
      <div className="flex flex-wrap gap-1 mt-1">{p.skills.map(s => <span key={s} className="chip chip-blue mono">{s}</span>)}</div>
      <div className="label mt-3">sandbox</div>
      <div className="flex items-center justify-between mt-1 text-[12px]">
        <span>{sandbox ? sandbox.name : <span className="text-muted">{t.noSandbox}</span>}</span>
        {sandbox && !kiosk && <button className="btn btn-danger !py-0.5" onClick={onRemove}>{t.remove}</button>}
      </div>
      {!sandbox && <div className="text-[11px] text-muted mt-2">{t.dropHint}</div>}
    </div>
  );
}

export function CommsStream({ events, live, lang, open, setOpen }: { events: ReefEvent[]; live: "sse"|"poll"|"down"; lang: Lang; open: boolean; setOpen: (b: boolean) => void }) {
  const t = STRINGS[lang]; const [tab, setTab] = useState<"chat"|"activity"|"history">("chat");
  const shown = events.filter(e => tab === "chat" ? (e.kind === "chat" || e.kind === "system") : e.kind === tab);
  const lvl = (l: ReefEvent["level"]) => l === "error" ? "text-magenta" : l === "warn" ? "text-amber" : l === "ok" ? "text-teal" : "text-ink";
  if (!open) return <button className="btn" onClick={() => setOpen(true)} data-testid="btn-comms"><Activity size={13} />{t.comms} <span className="mono text-muted">{events.length}</span></button>;
  return (
    <div className="panel flex flex-col w-[340px] max-h-[55vh]" data-testid="comms">
      <div className="flex items-center justify-between px-3 py-2 border-b border-line">
        <div><div className="label">{t.comms}</div><div className="text-[10px] text-muted mono">{events.length} {t.entries} · <span className={live === "sse" ? "text-teal" : live === "poll" ? "text-amber" : "text-magenta"}>{live === "sse" ? t.sseLive : live === "poll" ? t.ssePoll : t.sseDown}</span></div></div>
        <button className="text-muted hover:text-ink" onClick={() => setOpen(false)} aria-label="close"><X size={14} /></button>
      </div>
      <div className="flex gap-1 px-2 pt-2">{(["chat","activity","history"] as const).map(k => <button key={k} className={clsx("chip", tab === k ? "chip-teal" : "chip-grey")} onClick={() => setTab(k)}>{t[k]} <span className="mono opacity-70">{events.filter(e => k === "chat" ? (e.kind === "chat" || e.kind === "system") : e.kind === k).length}</span></button>)}</div>
      <div className="overflow-y-auto px-3 py-2 space-y-2 text-[12px]">
        {shown.length === 0 && <div className="text-muted">—</div>}
        {shown.slice().reverse().map(e => (
          <div key={e.id} className="border-b border-line/60 pb-1.5">
            <div className="flex items-center gap-2 text-[10px] text-muted mono"><span>{new Date(e.at * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}</span><span className="text-ink font-semibold">{e.actor}</span>{e.kind !== "chat" && <span className="chip chip-grey !py-0">{e.kind}</span>}</div>
            <div className={clsx("mt-0.5 break-words", lvl(e.level), e.kind === "activity" && "mono text-[11px]")}>{e.text}</div>
          </div>))}
      </div>
    </div>
  );
}

export function AskBar({ lang, selected, onAsk, busy }: { lang: Lang; selected: Profile | null; onAsk: (text: string) => void; busy: boolean }) {
  const t = STRINGS[lang]; const [text, setText] = useState("");
  return (
    <form className="panel flex items-center gap-2 px-3 py-2 w-[min(640px,92vw)]" onSubmit={(e) => { e.preventDefault(); if (text.trim()) { onAsk(text.trim()); setText(""); } }}>
      <Blocks size={14} className="text-muted" />
      <div className="text-[11px] text-muted whitespace-nowrap">{t.selected}: <span className="text-ink">{selected ? selected.name : t.none}</span></div>
      <input className="flex-1 bg-transparent outline-none text-[13px] placeholder:text-muted" placeholder={t.askPlaceholder} value={text} onChange={e => setText(e.target.value)} data-testid="ask-input" />
      <button className="btn btn-primary" disabled={busy || !text.trim()} data-testid="ask-submit">{t.ask}</button>
    </form>
  );
}
