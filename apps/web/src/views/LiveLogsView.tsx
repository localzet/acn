import { EmptyState } from "../components/EmptyState";
import { Panel } from "../components/Panel";
import type { LiveLogEntry } from "../types/dashboard";

export function LiveLogsView({ logs }: { logs: LiveLogEntry[] }) {
  return (
    <Panel title="Live Logs" description="SSE or WebSocket stream from the FastAPI backend.">
      {logs.length === 0 ? (
        <EmptyState label="No live log events received yet." />
      ) : (
        <div className="max-h-[640px] space-y-2 overflow-y-auto font-mono text-xs">
          {logs.map((log) => (
            <div
              key={log.id}
              className="grid gap-2 rounded-md border border-slate-200 p-3 dark:border-slate-800 md:grid-cols-[160px_90px_1fr]"
            >
              <span className="text-slate-500">{new Date(log.createdAt).toLocaleString()}</span>
              <span className={levelClass(log.level)}>{log.level}</span>
              <span>
                {log.source}: {log.message}
              </span>
            </div>
          ))}
        </div>
      )}
    </Panel>
  );
}

function levelClass(level: LiveLogEntry["level"]) {
  if (level === "error") return "text-red-600 dark:text-red-300";
  if (level === "warning") return "text-amber-600 dark:text-amber-300";
  if (level === "debug") return "text-slate-500";
  return "text-emerald-600 dark:text-emerald-300";
}
