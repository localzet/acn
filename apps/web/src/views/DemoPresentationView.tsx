import {
  GitBranch,
  GitCommit,
  Pause,
  Play,
  RotateCcw,
  ShieldCheck,
  SkipBack,
  SkipForward,
} from "lucide-react";
import type { ReactNode } from "react";
import {
  Area,
  AreaChart,
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import type { DashboardSnapshot, DemoPlaybackStep } from "../types/dashboard";

type DemoPlayback = {
  snapshot: DashboardSnapshot;
  step: DemoPlaybackStep;
  stepIndex: number;
  stepCount: number;
  playing: boolean;
  progress: number;
  play: () => void;
  pause: () => void;
  next: () => void;
  previous: () => void;
  restart: () => void;
};

export function DemoPresentationView({ playback }: { playback: DemoPlayback }) {
  const { snapshot, step } = playback;
  const latestMetric = snapshot.metricsTimeline.at(-1);

  return (
    <main className="demo-stage min-h-screen overflow-hidden bg-[#070b14] text-slate-100">
      <div className="pointer-events-none fixed inset-0 opacity-70">
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_20%_20%,rgba(34,197,94,0.24),transparent_28%),radial-gradient(circle_at_80%_10%,rgba(56,189,248,0.22),transparent_30%),linear-gradient(135deg,rgba(15,23,42,0.4),rgba(2,6,23,0.95))]" />
        <div className="demo-grid absolute inset-0" />
      </div>

      <section className="relative z-10 grid min-h-screen grid-rows-[auto_1fr_auto] gap-5 p-5 xl:p-7">
        <header className="flex flex-col gap-4 rounded-lg border border-white/10 bg-slate-950/70 p-4 shadow-2xl shadow-cyan-950/30 backdrop-blur xl:flex-row xl:items-center xl:justify-between">
          <div>
            <p className="text-xs font-semibold uppercase text-emerald-300">ACN Demo Mode</p>
            <h1 className="mt-1 text-2xl font-semibold xl:text-4xl">{step.title}</h1>
            <p className="mt-2 max-w-3xl text-sm text-slate-300 xl:text-base">{step.subtitle}</p>
          </div>
          <PlaybackControls playback={playback} />
        </header>

        <div className="grid min-h-0 gap-5 xl:grid-cols-[1.35fr_0.9fr]">
          <section className="grid min-h-0 gap-5">
            <DemoCommitGraph snapshot={snapshot} focus={step.focus} />
            <div className="grid gap-5 lg:grid-cols-2">
              <MetricCard
                label="Validation accuracy"
                value={latestMetric?.validationAccuracy ?? 0}
                tone="emerald"
              />
              <MetricCard
                label="Forgetting score"
                value={latestMetric?.forgettingScore ?? 0}
                tone="amber"
              />
            </div>
            <DemoMetrics snapshot={snapshot} />
          </section>

          <aside className="grid min-h-0 gap-5">
            <DemoTimeline snapshot={snapshot} activeFocus={step.focus} />
            <DemoRollback snapshot={snapshot} active={step.focus === "rollback"} />
            <DemoOverride active={step.focus === "override"} />
            <DemoBranches snapshot={snapshot} />
          </aside>
        </div>

        <footer className="rounded-lg border border-white/10 bg-slate-950/80 p-3 backdrop-blur">
          <div className="h-2 overflow-hidden rounded-full bg-slate-800">
            <div
              className="h-full rounded-full bg-gradient-to-r from-emerald-400 via-cyan-300 to-blue-400 transition-all duration-700"
              style={{ width: `${playback.progress * 100}%` }}
            />
          </div>
        </footer>
      </section>
    </main>
  );
}

function PlaybackControls({ playback }: { playback: DemoPlayback }) {
  return (
    <div className="flex flex-wrap items-center gap-2">
      <IconButton label="Previous" onClick={playback.previous}>
        <SkipBack size={18} />
      </IconButton>
      <IconButton label={playback.playing ? "Pause" : "Play"} onClick={playback.playing ? playback.pause : playback.play}>
        {playback.playing ? <Pause size={18} /> : <Play size={18} />}
      </IconButton>
      <IconButton label="Next" onClick={playback.next}>
        <SkipForward size={18} />
      </IconButton>
      <IconButton label="Restart" onClick={playback.restart}>
        <RotateCcw size={18} />
      </IconButton>
      <span className="ml-2 text-sm text-slate-400">
        {playback.stepIndex + 1}/{playback.stepCount}
      </span>
    </div>
  );
}

function IconButton({
  label,
  children,
  onClick,
}: {
  label: string;
  children: ReactNode;
  onClick: () => void;
}) {
  return (
    <button
      className="inline-flex size-10 items-center justify-center rounded-md border border-white/10 bg-white/5 text-slate-100 transition hover:bg-white/10"
      type="button"
      aria-label={label}
      title={label}
      onClick={onClick}
    >
      {children}
    </button>
  );
}

function DemoCommitGraph({
  snapshot,
  focus,
}: {
  snapshot: DashboardSnapshot;
  focus: DemoPlaybackStep["focus"];
}) {
  return (
    <div className="rounded-lg border border-white/10 bg-slate-950/75 p-5 backdrop-blur">
      <div className="mb-4 flex items-center justify-between">
        <div>
          <p className="text-xs font-semibold uppercase text-cyan-300">Training evolution</p>
          <h2 className="text-lg font-semibold">Animated commit graph</h2>
        </div>
        <GitCommit className="text-cyan-300" size={22} />
      </div>
      <div className="relative h-72 overflow-hidden rounded-md border border-slate-800 bg-slate-950">
        <svg className="absolute inset-0 h-full w-full" viewBox="0 0 900 280">
          {snapshot.commitGraph.edges.map((edge) => {
            const sourceIndex = snapshot.commitGraph.nodes.findIndex((node) => node.id === edge.parentId);
            const targetIndex = snapshot.commitGraph.nodes.findIndex((node) => node.id === edge.childId);
            if (sourceIndex < 0 || targetIndex < 0) return null;
            const source = pointForIndex(sourceIndex);
            const target = pointForIndex(targetIndex);
            return (
              <line
                key={`${edge.parentId}-${edge.childId}`}
                className="demo-edge"
                x1={source.x}
                y1={source.y}
                x2={target.x}
                y2={target.y}
              />
            );
          })}
          {snapshot.commitGraph.nodes.map((node, index) => {
            const point = pointForIndex(index);
            const hot = focus === "commit" || node.id.includes("recovered");
            return (
              <g key={node.id} className="demo-node">
                <circle
                  cx={point.x}
                  cy={point.y}
                  r={hot ? 22 : 17}
                  fill={hot ? "#22c55e" : "#38bdf8"}
                  opacity="0.95"
                />
                <text x={point.x - 58} y={point.y + 48} fill="#cbd5e1" fontSize="13">
                  {node.message}
                </text>
              </g>
            );
          })}
        </svg>
      </div>
    </div>
  );
}

function pointForIndex(index: number) {
  const points = [
    { x: 120, y: 140 },
    { x: 330, y: 140 },
    { x: 540, y: 85 },
    { x: 735, y: 185 },
  ];
  return points[index] ?? { x: 120 + index * 180, y: 140 };
}

function MetricCard({
  label,
  value,
  tone,
}: {
  label: string;
  value: number;
  tone: "emerald" | "amber";
}) {
  const color = tone === "emerald" ? "text-emerald-300" : "text-amber-300";
  return (
    <div className="rounded-lg border border-white/10 bg-slate-950/75 p-5 backdrop-blur">
      <p className="text-xs font-semibold uppercase text-slate-400">{label}</p>
      <p className={`mt-3 text-4xl font-semibold ${color}`}>{(value * 100).toFixed(0)}%</p>
    </div>
  );
}

function DemoMetrics({ snapshot }: { snapshot: DashboardSnapshot }) {
  return (
    <div className="rounded-lg border border-white/10 bg-slate-950/75 p-5 backdrop-blur">
      <h2 className="mb-3 text-lg font-semibold">Live metric updates</h2>
      <div className="h-56">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={snapshot.metricsTimeline}>
            <CartesianGrid stroke="#1e293b" strokeDasharray="3 3" />
            <XAxis dataKey="stageId" stroke="#94a3b8" />
            <YAxis stroke="#94a3b8" />
            <Tooltip contentStyle={{ background: "#020617", border: "1px solid #334155" }} />
            <Line dataKey="validationAccuracy" stroke="#34d399" strokeWidth={3} dot />
            <Line dataKey="forgettingScore" stroke="#f59e0b" strokeWidth={3} dot />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

function DemoTimeline({
  snapshot,
  activeFocus,
}: {
  snapshot: DashboardSnapshot;
  activeFocus: DemoPlaybackStep["focus"];
}) {
  return (
    <div className="rounded-lg border border-white/10 bg-slate-950/75 p-5 backdrop-blur">
      <div className="mb-4 flex items-center gap-2">
        <ShieldCheck className="text-emerald-300" size={20} />
        <h2 className="text-lg font-semibold">Controller decision timeline</h2>
      </div>
      <div className="space-y-3">
        {snapshot.controllerDecisions.map((decision) => (
          <div
            key={decision.id}
            className={`rounded-md border p-3 transition ${
              activeFocus === "override" && decision.status === "approved"
                ? "border-emerald-300 bg-emerald-400/10"
                : "border-slate-800 bg-slate-900/70"
            }`}
          >
            <p className="font-medium">{decision.action}</p>
            <p className="text-sm text-slate-400">
              confidence {(decision.confidence * 100).toFixed(0)}% · {decision.status}
            </p>
          </div>
        ))}
      </div>
    </div>
  );
}

function DemoRollback({ snapshot, active }: { snapshot: DashboardSnapshot; active: boolean }) {
  const rollback = snapshot.rollbackHistory[0];
  return (
    <div className={`rounded-lg border p-5 backdrop-blur ${active ? "border-amber-300 bg-amber-400/10" : "border-white/10 bg-slate-950/75"}`}>
      <h2 className="mb-3 text-lg font-semibold">Rollback visualization</h2>
      {rollback ? (
        <div className="grid grid-cols-[1fr_auto_1fr] items-center gap-3 text-sm">
          <code className="truncate rounded bg-red-500/15 px-2 py-2 text-red-200">{rollback.fromCommitId}</code>
          <RotateCcw className={active ? "demo-spin text-amber-200" : "text-slate-500"} size={24} />
          <code className="truncate rounded bg-emerald-500/15 px-2 py-2 text-emerald-200">{rollback.toCommitId}</code>
        </div>
      ) : (
        <p className="text-sm text-slate-400">Waiting for rollback trigger.</p>
      )}
    </div>
  );
}

function DemoOverride({ active }: { active: boolean }) {
  return (
    <div className={`rounded-lg border p-5 backdrop-blur ${active ? "border-cyan-300 bg-cyan-400/10" : "border-white/10 bg-slate-950/75"}`}>
      <h2 className="mb-3 text-lg font-semibold">Human override simulation</h2>
      <div className="flex items-center justify-between gap-4">
        <div>
          <p className="text-sm text-slate-300">Operator approval</p>
          <p className="text-xs text-slate-500">Citadel override ticket ACN-DEMO-42</p>
        </div>
        <span className={`rounded-md px-3 py-2 text-sm font-semibold ${active ? "bg-emerald-300 text-slate-950" : "bg-slate-800 text-slate-300"}`}>
          {active ? "approved" : "queued"}
        </span>
      </div>
    </div>
  );
}

function DemoBranches({ snapshot }: { snapshot: DashboardSnapshot }) {
  return (
    <div className="rounded-lg border border-white/10 bg-slate-950/75 p-5 backdrop-blur">
      <div className="mb-3 flex items-center gap-2">
        <GitBranch className="text-cyan-300" size={20} />
        <h2 className="text-lg font-semibold">Branch evolution</h2>
      </div>
      <div className="space-y-2">
        {snapshot.branchGraph.nodes.map((branch) => (
          <div key={branch.id} className="rounded-md border border-slate-800 bg-slate-900/70 p-3">
            <p className="font-medium">{branch.name}</p>
            <p className="truncate text-xs text-slate-500">head {branch.headCommitId}</p>
          </div>
        ))}
      </div>
    </div>
  );
}
