import {
  Activity,
  AlertTriangle,
  Bell,
  Blocks,
  GitBranch,
  GitCommit,
  History,
  LineChart,
  Moon,
  Radio,
  RotateCcw,
  ShieldCheck,
  Sun,
} from "lucide-react";
import { useMemo, useState } from "react";

import { useDashboardData } from "./hooks/useDashboardData";
import { OverrideConsole } from "./views/OverrideConsole";
import { BranchGraphView } from "./views/BranchGraphView";
import { CommitGraphView } from "./views/CommitGraphView";
import { ControllerDecisionsView } from "./views/ControllerDecisionsView";
import { ExperimentInspectorView } from "./views/ExperimentInspectorView";
import { LiveLogsView } from "./views/LiveLogsView";
import { MetricsTimelineView } from "./views/MetricsTimelineView";
import { RollbackHistoryView } from "./views/RollbackHistoryView";

const navItems = [
  { id: "commits", label: "Commits", icon: GitCommit },
  { id: "branches", label: "Branches", icon: GitBranch },
  { id: "metrics", label: "Metrics", icon: LineChart },
  { id: "experiment", label: "Experiment", icon: Blocks },
  { id: "decisions", label: "Decisions", icon: ShieldCheck },
  { id: "rollback", label: "Rollback", icon: RotateCcw },
  { id: "logs", label: "Logs", icon: Radio },
  { id: "override", label: "Override", icon: Bell },
] as const;

type ViewId = (typeof navItems)[number]["id"];

export function App() {
  const [activeView, setActiveView] = useState<ViewId>("commits");
  const [darkMode, setDarkMode] = useState(true);
  const { snapshot, connection, error, refresh, submitOverride } = useDashboardData();

  const activeExperiment = snapshot.experiments[0];
  const stats = useMemo(
    () => [
      { label: "Experiments", value: snapshot.experiments.length },
      { label: "Commits", value: snapshot.commitGraph.nodes.length },
      { label: "Branches", value: snapshot.branchGraph.nodes.length },
      { label: "Decisions", value: snapshot.controllerDecisions.length },
    ],
    [snapshot],
  );

  return (
    <div className={darkMode ? "dark" : ""}>
      <main className="min-h-screen bg-slate-100 text-slate-950 transition-colors dark:bg-slate-950 dark:text-slate-100">
        <aside className="fixed inset-x-0 top-0 z-20 border-b border-slate-200 bg-white/95 backdrop-blur dark:border-slate-800 dark:bg-slate-950/95 lg:inset-x-auto lg:inset-y-0 lg:w-64 lg:border-b-0 lg:border-r">
          <div className="flex h-16 items-center justify-between px-4 lg:h-auto lg:flex-col lg:items-start lg:gap-6 lg:px-5 lg:py-6">
            <div>
              <p className="text-xs font-semibold uppercase tracking-normal text-emerald-600 dark:text-emerald-400">
                Adaptive Core Network
              </p>
              <h1 className="mt-1 text-lg font-semibold">Control Surface</h1>
            </div>
            <button
              className="inline-flex size-9 items-center justify-center rounded-md border border-slate-300 text-slate-700 hover:bg-slate-100 dark:border-slate-700 dark:text-slate-200 dark:hover:bg-slate-900"
              type="button"
              aria-label="Toggle dark mode"
              onClick={() => setDarkMode((value) => !value)}
            >
              {darkMode ? <Sun size={18} /> : <Moon size={18} />}
            </button>
          </div>
          <nav className="flex gap-1 overflow-x-auto px-3 pb-3 lg:flex-col lg:overflow-visible lg:px-3">
            {navItems.map((item) => {
              const Icon = item.icon;
              const selected = item.id === activeView;
              return (
                <button
                  key={item.id}
                  className={`inline-flex min-w-fit items-center gap-2 rounded-md px-3 py-2 text-sm font-medium transition ${
                    selected
                      ? "bg-slate-900 text-white dark:bg-slate-100 dark:text-slate-950"
                      : "text-slate-600 hover:bg-slate-100 dark:text-slate-300 dark:hover:bg-slate-900"
                  }`}
                  type="button"
                  onClick={() => setActiveView(item.id)}
                >
                  <Icon size={16} />
                  {item.label}
                </button>
              );
            })}
          </nav>
        </aside>

        <section className="px-4 pb-8 pt-24 lg:ml-64 lg:px-6 lg:pt-6">
          <header className="mb-5 flex flex-col gap-4 xl:flex-row xl:items-center xl:justify-between">
            <div>
              <p className="text-sm text-slate-600 dark:text-slate-400">
                {activeExperiment
                  ? `${activeExperiment.name} / ${activeExperiment.branchName}`
                  : "No active experiment selected"}
              </p>
              <h2 className="mt-1 text-2xl font-semibold">Experiment Dashboard</h2>
            </div>
            <div className="flex flex-wrap items-center gap-2">
              <StatusBadge connection={connection} />
              {error ? (
                <span className="inline-flex items-center gap-2 rounded-md border border-amber-300 bg-amber-50 px-3 py-2 text-sm text-amber-800 dark:border-amber-500/40 dark:bg-amber-500/10 dark:text-amber-200">
                  <AlertTriangle size={16} />
                  API unavailable
                </span>
              ) : null}
              <button
                className="rounded-md border border-slate-300 px-3 py-2 text-sm font-medium hover:bg-slate-100 dark:border-slate-700 dark:hover:bg-slate-900"
                type="button"
                onClick={refresh}
              >
                Refresh
              </button>
            </div>
          </header>

          <div className="mb-5 grid grid-cols-2 gap-3 xl:grid-cols-4">
            {stats.map((stat) => (
              <div
                key={stat.label}
                className="rounded-lg border border-slate-200 bg-white p-4 dark:border-slate-800 dark:bg-slate-900"
              >
                <p className="text-xs font-medium uppercase tracking-normal text-slate-500">
                  {stat.label}
                </p>
                <p className="mt-2 text-2xl font-semibold">{stat.value}</p>
              </div>
            ))}
          </div>

          {activeView === "commits" && <CommitGraphView graph={snapshot.commitGraph} />}
          {activeView === "branches" && <BranchGraphView graph={snapshot.branchGraph} />}
          {activeView === "metrics" && <MetricsTimelineView metrics={snapshot.metricsTimeline} />}
          {activeView === "experiment" && (
            <ExperimentInspectorView experiments={snapshot.experiments} />
          )}
          {activeView === "decisions" && (
            <ControllerDecisionsView decisions={snapshot.controllerDecisions} />
          )}
          {activeView === "rollback" && (
            <RollbackHistoryView history={snapshot.rollbackHistory} />
          )}
          {activeView === "logs" && <LiveLogsView logs={snapshot.liveLogs} />}
          {activeView === "override" && (
            <OverrideConsole decisions={snapshot.controllerDecisions} onSubmit={submitOverride} />
          )}
        </section>
      </main>
    </div>
  );
}

function StatusBadge({ connection }: { connection: string }) {
  const online = connection === "connected";
  return (
    <span
      className={`inline-flex items-center gap-2 rounded-md border px-3 py-2 text-sm ${
        online
          ? "border-emerald-300 bg-emerald-50 text-emerald-700 dark:border-emerald-500/40 dark:bg-emerald-500/10 dark:text-emerald-200"
          : "border-slate-300 bg-white text-slate-600 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-300"
      }`}
    >
      <Activity size={16} />
      {connection}
    </span>
  );
}
