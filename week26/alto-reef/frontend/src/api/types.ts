export type Harness = "openclaw" | "hermes" | "deepagents" | "nat";
export type Archetype = "researcher"|"analyst"|"critic"|"planner"|"writer"|"coder"|"lead"|"ops_engineer";
export type Tier = "restricted"|"balanced"|"open"|"personal";
export type SandboxStatus = "running"|"degraded"|"error"|"stopped"|"unknown";

export interface Profile { id: string; name: string; harness: Harness; archetype: Archetype; color: string;
  accessories: Record<string,string>; skills: string[]; tier: Tier; systemPrompt: string; sandboxId: string|null; persona: string; }
export interface Sandbox { id: string; name: string; blueprint: string; status: SandboxStatus; modelHandle: string|null; provider: string|null;
  presets: string[]; policyRevision: number|null; forwards: {remotePort:number; localPort:number}[]; members: string[];
  lastRunSummary: string|null; lastRunStatus: string|null; slot: number; probedAt: number|null; source: string; }
export interface Landmark { id: string; label: string; kind: "lighthouse"|"reactor"|"nautilus"|"buoy"; ok: boolean|null; detail: string; probedAt: number|null; url: string|null; }
export interface Health { mode: "mock"|"real"; gateway: {ok:boolean; detail:string; at:number}; inference: {provider?:string|null; model?:string|null; ok?:boolean; at:number}; landmarks: Landmark[]; at: number; }
export interface ReefEvent { id: number; at: number; kind: "chat"|"activity"|"history"|"system"; sandboxId: string|null; profileId: string|null; actor: string; text: string; level: "info"|"warn"|"error"|"ok"; }
