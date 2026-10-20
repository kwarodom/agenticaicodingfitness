/* Jev UI judge — end-to-end test of the Jev Lab Runner, with Jev as the fast semantic judge.
 *
 * Load inside the running app (browser console, or a Claude-in-Chrome javascript call):
 *     const J = await import("/static/uitest/jev_ui_judge.js");
 *     J.start({labMode: "live"});        // runs in the background
 *     J.status();                        // poll: {phase, done, total, flags}
 *     J.report();                        // final: only what needs a human look
 *
 * Division of labour (the course's own rule — code checks facts, Jev judges meaning):
 *   code → exit codes, element counts, JS errors, placeholder tags, raw-markdown regex
 *   Jev  → does a lab's real output tell the same story as the tutorial's "Expected output"?
 *          does any output claim a real-world action was executed?
 *          is a rendered section visibly broken? how clear is it for a beginner?
 * All Jev calls go through the runner's own /api/jev proxy, so the key stays server-side.
 */
const MODEL = "jev-1.13.0";
const S = {phase: "idle", done: 0, total: 0, calls: 0, tokens: 0, cost: 0, started: 0,
           sections: [], labs: [], inline: [], starters: [], errors: []};
const sleep = ms => new Promise(r => setTimeout(r, ms));
const clip = (t, n) => (t.length <= n ? t : t.slice(0, Math.floor(n * 0.45)) + "\n…[middle trimmed]…\n" + t.slice(-Math.floor(n * 0.5)));

window.addEventListener("error", e => S.errors.push(String(e.message || e)));
window.addEventListener("unhandledrejection", e => S.errors.push("promise: " + String(e.reason && e.reason.message || e.reason)));

async function jev(state, questions) {
  const r = await fetch("/api/jev", {method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify({state, questions, mode: "live", model: MODEL})});
  const d = await r.json();
  if (!r.ok) throw new Error(typeof d.detail === "string" ? d.detail : JSON.stringify(d.detail));
  S.calls++; S.tokens += (d.usage || {}).input_tokens || 0; S.cost += d._cost_usd || 0;
  return d.answers;
}

/* ── expected-output blocks: a ```bash block naming the file, then **Expected output** + fence ── */
function expectedFor(fi) {
  const map = {};
  const md = COURSE[fi].sections.map(s => s.md || "").join("\n");
  const lines = md.split("\n");
  let lastCmd = "", i = 0;
  while (i < lines.length) {
    const t = lines[i].trim();
    if (t.startsWith("```bash") || t === "```sh") {
      const buf = []; i++;
      while (i < lines.length && !lines[i].trim().startsWith("```")) buf.push(lines[i++]);
      lastCmd = buf.join("\n"); i++; continue;
    }
    if (/expected output/i.test(t)) {
      let j = i + 1;
      while (j < lines.length && !lines[j].trim()) j++;
      if (j < lines.length && lines[j].trim().startsWith("```")) {
        const buf = []; j++;
        while (j < lines.length && !lines[j].trim().startsWith("```")) buf.push(lines[j++]);
        const files = lastCmd.match(/(labs|exercises(\/solutions)?)\/[a-z0-9_]+\.py/g) || [];
        files.forEach(f => { if (!map[f]) map[f] = buf.join("\n"); });
        i = j + 1; continue;
      }
    }
    i++;
  }
  return map;
}

/* ── 1 · every rendered section: code checks + one batched Jev call per module (fan-out) ── */
const CLARITY = ["Confusing: jargon or steps without explanation",
                 "Followable with effort: some unexplained terms or jumps",
                 "Clear: plain language and concrete steps",
                 "Very clear: plain language, concrete steps, and an example, command or checkpoint"];

async function auditSections() {
  S.phase = "sections";
  for (let fi = 0; fi < COURSE.length; fi++) {
    const texts = {}, meta = {};
    COURSE[fi].sections.forEach((s, si) => {
      gotoSec(fi, si);
      const c = document.getElementById("content");
      const md = c.querySelector(".md");
      const code = {
        pwarn: c.querySelectorAll(".pwarn").length,
        rawMd: !!md && /(^|\s)\*\*\S|\]\(http|^#{2,}\s/m.test([...md.childNodes].filter(n => !n.matches || !n.matches(".codewrap,.jevblk,pre")).map(n => n.textContent).join("\n")),
        jevBlocks: c.querySelectorAll(".jevblk").length,
        cards: c.querySelectorAll(".labcard").length,
      };
      if (md) {                                             // judge the prose, not code/JSON
        const clone = md.cloneNode(true);
        clone.querySelectorAll(".codewrap,.jevblk,pre").forEach(n => n.replaceWith(document.createTextNode(" [code block] ")));
        texts["s" + si] = clip(clone.innerText.replace(/\n{3,}/g, "\n\n").trim(), 3500);
      }
      meta["s" + si] = {fi, si, id: s.id, title: s.title, kind: s.kind, code};
    });
    const qs = {};
    Object.keys(texts).forEach(k => {
      if (texts[k].length < 40) return;
      qs["render_" + k] = {type: "noul", instructions:
        `Does \`sections.${k}\` show visibly broken formatting that a reader would notice as a bug — literal markdown symbols such as ** or ## or [text](url), stray HTML tags, or garbled characters? The marker [code block] and ordinary punctuation, arrows, emoji, tables and inline names are fine.`};
      if (!["diagrams", "labs", "next", "troubleshooting"].includes(meta[k].kind))
        qs["clarity_" + k] = {type: "score", instructions:
          `How easy is \`sections.${k}\` to follow for a beginner programmer learning to call an AI API?`, criteria: CLARITY};
    });
    let ans = {};
    try { ans = await jev({course_module: COURSE[fi].title, sections: texts}, qs); }
    catch (e) { S.errors.push(`sections ${COURSE[fi].folder}: ${e.message}`); }
    Object.entries(meta).forEach(([k, m]) => {
      S.sections.push({...m, folder: COURSE[fi].folder,
        broken: ans["render_" + k] ? ans["render_" + k].noul : null,
        clarity: ans["clarity_" + k] ? +ans["clarity_" + k].score.toFixed(2) : null});
    });
    S.done++;
  }
}

/* ── 2 · run every lab + solution through the real ▶ Run path, then Jev compares to Expected ── */
async function auditLabs(labMode) {
  S.phase = "labs"; setMode(labMode);
  for (let fi = 0; fi < COURSE.length; fi++) {
    const c = COURSE[fi], exp = expectedFor(fi);
    const jobs = [...(c.labs || []).map(l => ["labs", l.file]),
                  ...(c.exercises || []).filter(e => e.solution).map(e => ["exercises", e.solution])];
    for (const [kind, rel] of jobs) {
      gotoSec(fi, c.sections.findIndex(s => s.kind === kind));
      const t0 = performance.now();
      await runFile(fi, rel);
      const secs = (performance.now() - t0) / 1000;
      const k = fileKey(fi, rel);
      const out = (document.getElementById("term-" + fid(fi, rel)) || {}).innerText || "";
      const foot = (document.getElementById("foot-" + fid(fi, rel)) || {}).innerText || "";
      const exitOk = /exit 0/.test(foot);
      const rec = {folder: c.folder, file: rel, exitOk, foot, secs: +secs.toFixed(1),
                   placeholder: /PLACEHOLDER/.test(out), traceback: /Traceback \(most recent/.test(out),
                   hasExpected: !!exp[rel]};
      try {
        const qs = {
          unexpected_problem: {type: "noul", instructions:
            "Does `output` show an unexpected problem — a Python traceback, an HTTP or network error, a missing file, or an answer labelled PLACEHOLDER — as opposed to intentional teaching examples of wrong or uncertain answers?"},
          claims_action: {type: "noul", instructions:
            "Does `output` claim that a real-world action was actually executed (an email sent or deleted, a setpoint or BACnet value written, a candidate rejected, a payment made, a work order filed)? Proposals, recommendations and lines saying execute: False do not count."},
        };
        if (exp[rel]) qs.matches_expected = {type: "choice", instructions:
          "Compare `output` (a real run) with `expected` (a trimmed sample printed in the tutorial). Ignore differences in latency, cost, token counts, dates, small probability shifts of about ±0.1, column alignment and lines the sample omitted with …. Do they tell the learner the same story — same steps, same routes, same pass/fail verdicts?",
          criteria: {same_story: "Same steps and the same conclusions",
                     minor_drift: "Same overall story, but some labels, rankings or verdicts differ noticeably",
                     contradicts: "Different conclusions: other routes, verdicts or pass/fail results, or missing steps"}};
        const a = await jev({file: rel, output: clip(out, 9000), expected: exp[rel] || ""}, qs);
        rec.problem = a.unexpected_problem.noul; rec.action = a.claims_action.noul;
        if (a.matches_expected) { rec.match = a.matches_expected.choice; rec.matchConf = a.matches_expected.confidence; }
      } catch (e) { rec.judgeError = e.message; }
      S.labs.push(rec); S.done++;
    }
  }
}

/* ── 3 · every inline ⚡ Ask Jev block: click it for real, then count what rendered (code only) ── */
async function auditInline() {
  S.phase = "inline"; setMode("live");
  for (let i = 0; i < FLAT.length; i++) {
    const f = FLAT[i]; gotoSec(f.fi, f.si);
    for (let n = 0; n < JB.length; n++) {
      let spec; try { spec = JSON.parse(JB[n].orig); } catch (e) { S.inline.push({key: JB[n].key, ok: false, why: "invalid JSON in tutorial"}); continue; }
      await askJev(n);
      const o = document.getElementById("jb-out-" + n);
      const got = o.querySelectorAll(".jans").length, want = Object.keys(spec.questions).length;
      S.inline.push({key: JB[n].key, ok: got === want && !o.querySelector(".jwarn.bad") && /LIVE/.test(o.innerText),
                     got, want, err: (o.querySelector(".jwarn.bad") || {}).innerText || ""});
      S.calls++;
    }
    for (let n = 0; n < (typeof LB !== "undefined" ? LB.length : 0); n++) {   // ```llm blocks → the chosen LLM
      await askLlm(n);
      const o = document.getElementById("lb-out-" + n);
      const txt = (o.querySelector(".llmtext") || {}).innerText || "";
      S.inline.push({key: LB[n].key, kind: "llm", ok: txt.trim().length > 0 && !o.querySelector(".jwarn.bad") && /LIVE/.test(o.innerText),
                     chars: txt.length, err: (o.querySelector(".jwarn.bad,.jwarn") || {}).innerText || ""});
    }
    S.done = i;
  }
}

/* ── 4 · unfinished exercise starters must stop at the free checker (code only) ── */
async function auditStarters() {
  S.phase = "starters"; setMode("dry");
  for (let fi = 0; fi < COURSE.length; fi++) {
    for (const e of COURSE[fi].exercises || []) {
      gotoSec(fi, COURSE[fi].sections.findIndex(s => s.kind === "exercises"));
      await runFile(fi, e.file);
      const out = document.getElementById("term-" + fid(fi, e.file)).innerText;
      const foot = document.getElementById("foot-" + fid(fi, e.file)).innerText;
      S.starters.push({file: `${COURSE[fi].folder}/${e.file}`, ok: /not done yet/.test(foot) && !/◆ jev/.test(out) && !/Traceback/.test(out)});
    }
  }
  setMode("live");
}

export function start({labMode = "live", only = null, lang = "en"} = {}) {
  if (S.phase !== "idle" && S.phase !== "done") return "already running";
  Object.assign(S, {phase: "starting", done: 0, calls: 0, tokens: 0, cost: 0, started: Date.now(),
                    sections: [], labs: [], inline: [], starters: [], errors: []});
  (async () => {
    try {
      if (typeof setLang === "function" && lang !== LANG) await setLang(lang);   // audit the Thai course with lang:"th"
      const want = p => !only || only.includes(p);
      if (want("sections")) await auditSections();
      if (want("labs")) await auditLabs(labMode);
      if (want("inline")) await auditInline();
      if (want("starters")) await auditStarters();
      S.phase = "done";
    } catch (e) { S.errors.push("harness: " + (e.stack || e)); S.phase = "done"; }
  })();
  return "started";
}

export function status() {
  return {phase: S.phase, done: S.done, secs: Math.round((Date.now() - S.started) / 1000),
          jevCalls: S.calls, cost: +S.cost.toFixed(5), labsRun: S.labs.length, errors: S.errors.length};
}

/* Only what needs a human (or Claude) to look — everything else passed. */
export function report({clarityBelow = 1.6, brokenAbove = 0.5, problemAbove = 0.5} = {}) {
  const sec = S.sections;
  const clar = sec.filter(s => s.clarity != null);
  const byModule = {};
  clar.forEach(s => { (byModule[s.folder] = byModule[s.folder] || []).push(s.clarity); });
  return {
    phase: S.phase, seconds: Math.round((Date.now() - S.started) / 1000),
    jev: {calls: S.calls, input_tokens: S.tokens, cost_usd: +S.cost.toFixed(5)},
    js_errors: S.errors,
    sections: {
      total: sec.length,
      code_flags: sec.filter(s => s.code.pwarn || s.code.rawMd).map(s => `${s.folder}/${s.id}: pwarn=${s.code.pwarn} rawMd=${s.code.rawMd}`),
      jev_render_flags: sec.filter(s => s.broken != null && s.broken > brokenAbove).map(s => `${s.folder}/${s.id} p(broken)=${s.broken}`),
      low_clarity: clar.filter(s => s.clarity < clarityBelow).map(s => `${s.folder}/${s.id} "${s.title}" clarity=${s.clarity}`),
      clarity_by_module: Object.fromEntries(Object.entries(byModule).map(([k, v]) => [k, +(v.reduce((a, b) => a + b, 0) / v.length).toFixed(2)])),
    },
    labs: {
      total: S.labs.length,
      exit_failures: S.labs.filter(l => !l.exitOk).map(l => `${l.folder}/${l.file}: ${l.foot}`),
      code_flags: S.labs.filter(l => l.placeholder || l.traceback).map(l => `${l.folder}/${l.file} placeholder=${l.placeholder} traceback=${l.traceback}`),
      jev_problem: S.labs.filter(l => l.problem > problemAbove).map(l => `${l.folder}/${l.file} p=${l.problem}`),
      jev_claims_action: S.labs.filter(l => l.action > 0.5).map(l => `${l.folder}/${l.file} p=${l.action}`),
      expected_mismatch: S.labs.filter(l => l.match && l.match !== "same_story").map(l => `${l.folder}/${l.file}: ${l.match} (conf ${l.matchConf})`),
      no_expected_block: S.labs.filter(l => !l.hasExpected).map(l => `${l.folder}/${l.file}`),
      judge_errors: S.labs.filter(l => l.judgeError).map(l => `${l.folder}/${l.file}: ${l.judgeError}`),
      slowest: [...S.labs].sort((a, b) => b.secs - a.secs).slice(0, 3).map(l => `${l.folder}/${l.file} ${l.secs}s`),
    },
    inline: {total: S.inline.length, failures: S.inline.filter(x => !x.ok)},
    starters: {total: S.starters.length, failures: S.starters.filter(x => !x.ok)},
  };
}

export function raw() { return S; }
