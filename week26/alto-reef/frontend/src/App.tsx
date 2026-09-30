import { useState } from "react";
import { Route, Router, Switch } from "wouter";
import { useHashLocation } from "wouter/use-hash-location";
import { QueryClientProvider } from "@tanstack/react-query";
import { queryClient } from "@/api/client";
import ReefPage from "@/pages/Reef";
import Placeholder from "@/pages/Placeholder";
import type { Lang } from "@/i18n/strings";

export default function App() {
  const [lang, setLang] = useState<Lang>("en");
  return (
    <QueryClientProvider client={queryClient}>
      <Router hook={useHashLocation}>
        <Switch>
          <Route path="/" component={() => <ReefPage lang={lang} setLang={setLang} />} />
          <Route path="/manager" component={() => <Placeholder title="Sandbox Manager" lang={lang} milestone="M2" />} />
          <Route path="/build" component={() => <Placeholder title="Build a Claw" lang={lang} milestone="M3" />} />
          <Route path="/sandbox/:id">{(p) => <Placeholder title={`Workbench — ${p.id}`} lang={lang} milestone="M2" />}</Route>
          <Route component={() => <Placeholder title="Not found" lang={lang} milestone="404" />} />
        </Switch>
      </Router>
    </QueryClientProvider>
  );
}
