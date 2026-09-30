import { useMemo, useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { useLocation } from "wouter";
import { api, queryClient } from "@/api/client";
import { useReefEvents } from "@/api/events";
import type { Health, Landmark, Profile, Sandbox } from "@/api/types";
import { ReefCanvas } from "@/components/reef/ReefCanvas";
import { AskBar, CommsStream, HealthTiles, LandmarkTip, ProfileCard, StatusBanner, Toolbar } from "@/components/Hud";
import { STRINGS, type Lang } from "@/i18n/strings";

export default function ReefPage({ lang, setLang }: { lang: Lang; setLang: (l: Lang) => void }) {
  const t = STRINGS[lang]; const [, navigate] = useLocation();
  const health = useQuery<Health>({ queryKey: ["/api/health"], refetchInterval: 10_000 });
  const sandboxes = useQuery<Sandbox[]>({ queryKey: ["/api/sandboxes"], refetchInterval: 10_000 });
  const profiles = useQuery<Profile[]>({ queryKey: ["/api/profiles"] });
  const { events, live } = useReefEvents();
  const [selected, setSelected] = useState<string | null>(null);
  const [kiosk, setKiosk] = useState(false);
  const [commsOpen, setCommsOpen] = useState(true);
  const [tip, setTip] = useState<{ l: Landmark; x: number; y: number } | null>(null);
  const [confirmReset, setConfirmReset] = useState(false);

  const invalidate = () => { queryClient.invalidateQueries({ queryKey: ["/api/sandboxes"] }); queryClient.invalidateQueries({ queryKey: ["/api/profiles"] }); };
  const assign = useMutation({ mutationFn: ({ sid, pid }: { sid: string; pid: string }) => api(`/api/sandboxes/${sid}/members`, { method: "POST", body: JSON.stringify({ profileId: pid }) }), onSuccess: () => { invalidate(); setSelected(null); } });
  const remove = useMutation({ mutationFn: ({ sid, pid }: { sid: string; pid: string }) => api(`/api/sandboxes/${sid}/members/${pid}`, { method: "DELETE" }), onSuccess: invalidate });
  const ask = useMutation({ mutationFn: (text: string) => api(`/api/ask`, { method: "POST", body: JSON.stringify({ text, profileIds: selected ? [selected] : [] }) }) });

  const selectedProfile = useMemo(() => profiles.data?.find(p => p.id === selected) ?? null, [profiles.data, selected]);
  const selectedSandbox = selectedProfile?.sandboxId ? sandboxes.data?.find(s => s.id === selectedProfile.sandboxId) : undefined;
  const error = health.error || sandboxes.error || profiles.error;

  return (
    <div className="relative h-full w-full overflow-hidden bg-bg">
      {/* map */}
      <div className="absolute inset-0">
        {sandboxes.data && profiles.data && health.data ? (
          <ReefCanvas sandboxes={sandboxes.data} profiles={profiles.data} landmarks={health.data.landmarks} selectedProfile={selected}
                      onSelectProfile={setSelected} onOpenSandbox={(id) => navigate(`/sandbox/${id}`)}
                      onAssign={(sid, pid) => { if (!kiosk) assign.mutate({ sid, pid }); }}
                      onLandmarkHover={(l, x, y) => setTip(l ? { l, x, y } : null)} />
        ) : (
          <div className="h-full w-full flex items-center justify-center text-muted text-[13px]">
            {error ? <div className="panel px-4 py-3 text-magenta">Runner unreachable: {(error as Error).message}</div> : "Loading reef…"}
          </div>
        )}
      </div>
      {health.data?.mode === "mock" && (
        <div className="pointer-events-none absolute inset-0 z-10 flex items-center justify-center overflow-hidden" aria-hidden>
          <div className="rotate-[-24deg] text-[110px] font-bold tracking-[0.2em] text-white/10 whitespace-nowrap">MOCK DATA</div>
        </div>)}
      {tip && <LandmarkTip l={tip.l} x={tip.x} y={tip.y} />}

      {/* top-left: title + health */}
      <div className="absolute left-3 top-3 z-20 flex flex-col gap-2 w-[min(520px,60vw)]">
        <div className="panel px-3 py-2 flex items-center gap-3">
          <div><div className="font-semibold text-[15px] leading-tight">{t.title}</div><div className="text-[11px] text-muted">{t.subtitle}</div></div>
          <div className="ml-auto flex items-center gap-1.5">
            <span className={health.data?.mode === "real" ? "chip chip-teal" : "chip chip-amber"}>{health.data?.mode ?? "…"}</span>
            {sandboxes.data && <span className="chip chip-grey mono">{sandboxes.data.length} sandboxes</span>}
            {profiles.data && <span className="chip chip-grey mono">{profiles.data.length} {t.claws}</span>}
          </div>
        </div>
        <HealthTiles h={health.data} lang={lang} />
        <StatusBanner h={health.data} lang={lang} />
      </div>

      {/* top-right toolbar */}
      <div className="absolute right-3 top-3 z-20">
        <Toolbar lang={lang} setLang={setLang} kiosk={kiosk} setKiosk={setKiosk} onRefresh={() => { health.refetch(); invalidate(); }}
                 onReset={() => setConfirmReset(true)} onManager={() => navigate("/manager")} onBuild={() => navigate("/build")} />
      </div>

      {/* right: comms */}
      <div className="absolute right-3 bottom-20 z-20">
        <CommsStream events={events} live={live} lang={lang} open={commsOpen} setOpen={setCommsOpen} />
      </div>

      {/* left-bottom: selected profile card */}
      {selectedProfile && (
        <div className="absolute left-3 bottom-20 z-20">
          <ProfileCard p={selectedProfile} sandbox={selectedSandbox} lang={lang} kiosk={kiosk} onClose={() => setSelected(null)}
                       onRemove={() => selectedSandbox && remove.mutate({ sid: selectedSandbox.id, pid: selectedProfile.id })} />
          {selectedSandbox && <button className="btn mt-2" onClick={() => navigate(`/sandbox/${selectedSandbox.id}`)}>{t.openWorkbench}: {selectedSandbox.name}</button>}
        </div>)}

      {/* bottom: ask bar */}
      <div className="absolute bottom-3 left-1/2 -translate-x-1/2 z-20">
        <AskBar lang={lang} selected={selectedProfile} busy={ask.isPending} onAsk={(text) => ask.mutate(text)} />
      </div>

      {/* reset confirmation: lists exactly what would run */}
      {confirmReset && (
        <div className="absolute inset-0 z-30 bg-black/60 flex items-center justify-center" onClick={() => setConfirmReset(false)}>
          <div className="panel p-4 w-[520px]" onClick={e => e.stopPropagation()}>
            <div className="label">{t.resetBooth}</div>
            <div className="text-[13px] mt-1">This would destroy and recreate the lab sandboxes. The runner would execute exactly:</div>
            <pre className="mono text-[11px] mt-2 bg-bg/70 rounded-md p-2 overflow-x-auto">{(sandboxes.data ?? []).map(s => `openshell sandbox delete ${s.id}`).join("\n")}\n# then re-onboard from the lab catalog (M4)</pre>
            <div className="text-[11px] text-amber mt-2">Not wired yet: sandbox lifecycle mutations arrive in milestone M1. Nothing will be executed.</div>
            <div className="flex justify-end gap-2 mt-3"><button className="btn" onClick={() => setConfirmReset(false)}>Cancel</button><button className="btn btn-danger" disabled>Confirm (M1)</button></div>
          </div>
        </div>)}
    </div>
  );
}
