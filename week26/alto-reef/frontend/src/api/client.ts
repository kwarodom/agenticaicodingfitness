import { QueryClient } from "@tanstack/react-query";
// __PORT_4455__ is rewritten at deploy time to proxy the runner; locally Vite proxies /api.
export const API_BASE = "__PORT_4455__".startsWith("__") ? "" : "__PORT_4455__";
const TOKEN = (import.meta as any).env?.VITE_REEF_TOKEN as string | undefined;

export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...(TOKEN ? { Authorization: `Bearer ${TOKEN}` } : {}), ...(init?.headers || {}) },
  });
  if (!res.ok) throw new Error(`${res.status}: ${(await res.text()) || res.statusText}`);
  return res.json() as Promise<T>;
}
export const queryClient = new QueryClient({
  defaultOptions: { queries: { queryFn: ({ queryKey }) => api(queryKey.join("/")), retry: false, refetchOnWindowFocus: false, staleTime: 5_000 } },
});
