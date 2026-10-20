/**
 * Step 4 of Exercise 3 — batch processing pattern with custom_id correlation.
 *
 * Demonstrates: when extracting many documents, tag each with a custom_id so
 * failures can be identified and resubmitted independently. This is the shape
 * of Anthropic's Message Batches API. Pi is provider-agnostic so we
 * demonstrate the PATTERN sequentially — in production you'd POST the same
 * shape as a true batch and get 50% off.
 *
 * Maps to slide 7 ("Two ways to call the API: sync vs batch").
 *
 * Why custom_id is load-bearing:
 *   - Batch results may not return in submission order
 *   - When 5 of 100 fail, you resubmit ONLY the 5 (identified by custom_id)
 *   - Never reprocess the 95 successes
 *
 * Pattern in this file: glob the source paths → load each as { custom_id,
 * content } → ask the model to process each in turn, preserving custom_ids.
 */

import { glob, readFile } from "node:fs/promises";
import { resolve } from "node:path";
import type { ExtensionAPI } from "@mariozechner/pi-coding-agent";

/**
 * Register the /batch-extract command.
 * @param pi            the extension API
 * @param extensionDir  used as the fallback search root when the user's
 *                      glob doesn't match anything in cwd (so the bundled
 *                      samples just work from any working directory).
 */
export function registerBatchCommand(pi: ExtensionAPI, extensionDir: string): void {
  pi.registerCommand("batch-extract", {
    description: "Batch-extract multiple documents. Usage: /batch-extract <glob>",
    handler: async (args, ctx) => {
      const pattern = args.trim() || "samples/*.txt";

      const collect = async (cwd: string): Promise<string[]> => {
        const out: string[] = [];
        for await (const p of glob(pattern, { cwd })) {
          out.push(typeof p === "string" ? p : p.toString());
        }
        return out;
      };

      // Try cwd first (so the user can batch their own docs); fall back to
      // the extension dir (so bundled samples work regardless of where pi
      // was invoked from).
      let baseDir = ctx.cwd;
      let paths = await collect(baseDir);
      if (paths.length === 0) {
        baseDir = extensionDir;
        paths = await collect(baseDir);
      }

      if (paths.length === 0) {
        ctx.ui.notify(`No files matched ${pattern}`, "warning");
        return;
      }

      ctx.ui.notify(
        `Batch: ${paths.length} documents from ${baseDir}. ` +
          `custom_ids = filenames. Sequential demo (real Batch API would be parallel + 50% cheaper).`,
        "info",
      );

      // Build the batch as { custom_id, content } pairs. In the real
      // Anthropic Batch API this would become the requests[] array.
      const docs: Array<{ custom_id: string; content: string }> = [];
      for (const path of paths) {
        try {
          docs.push({
            custom_id: path,
            content: await readFile(resolve(baseDir, path), "utf8"),
          });
        } catch (err) {
          // Failure tracked by custom_id — exactly what Anthropic's Batch API
          // gives you for free. In production you'd resubmit just these.
          ctx.ui.notify(`Failed to load [${path}]: ${(err as Error).message}`, "error");
        }
      }

      const batchPrompt =
        `Extract each of the ${docs.length} documents below using extract_invoice. ` +
        `Process them one at a time. The custom_id for each extraction MUST be ` +
        `the source path provided. Treat failures (validation errors after retry) ` +
        `as terminal for that custom_id and continue with the next document.\n\n` +
        docs
          .map(
            (d) =>
              `=== custom_id=${d.custom_id} ===\n${d.content}\n=== end ${d.custom_id} ===`,
          )
          .join("\n\n");

      pi.sendUserMessage(batchPrompt);
    },
  });
}
