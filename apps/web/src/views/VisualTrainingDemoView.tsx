import {
  CheckCircle2,
  Cpu,
  Pause,
  Play,
  RotateCcw,
  ShieldAlert,
  SlidersHorizontal,
  Upload,
  XCircle,
} from "lucide-react";
import type { ChangeEvent } from "react";
import type { ReactNode } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import type { VisualDemoInference, VisualDemoSnapshot } from "../types/visualDemo";

type VisualTrainingDemoProps = {
  snapshot: VisualDemoSnapshot;
  error: string | null;
  inference: VisualDemoInference | null;
  onStart: (autoMode: boolean) => void;
  onPause: () => void;
  onResume: () => void;
  onRollback: () => void;
  onSetAutoMode: (enabled: boolean) => void;
  onApprove: (decisionId: string) => void;
  onReject: (decisionId: string) => void;
  onPredict: (imageDataUrl: string) => void;
};

export function VisualTrainingDemoView({
  snapshot,
  error,
  inference,
  onStart,
  onPause,
  onResume,
  onRollback,
  onSetAutoMode,
  onApprove,
  onReject,
  onPredict,
}: VisualTrainingDemoProps) {
  const latest = snapshot.metrics.at(-1);
  const pendingDecision = snapshot.decisions.find((decision) => decision.status === "pending");

  return (
    <main className="min-h-screen bg-slate-950 text-slate-100">
      <section className="mx-auto grid max-w-[1600px] gap-5 px-4 py-5">
        <header className="grid gap-4 rounded-lg border border-slate-800 bg-slate-900 p-5 lg:grid-cols-[1fr_auto] lg:items-center">
          <div>
            <p className="text-sm font-semibold uppercase text-cyan-300">ACN Visual Live Demo</p>
            <h1 className="mt-2 text-3xl font-semibold">
              Watch the neural network learn, fail, recover, adapt and improve
            </h1>
            <p className="mt-2 max-w-4xl text-slate-300">
              A small CNN learns to distinguish visually understandable airplane and ship images.
              ACN tracks checkpoints, detects degradation, rolls back and resumes training.
            </p>
          </div>
          <ControlPanel
            snapshot={snapshot}
            pendingDecisionId={pendingDecision?.id}
            onStart={onStart}
            onPause={onPause}
            onResume={onResume}
            onRollback={onRollback}
            onSetAutoMode={onSetAutoMode}
            onApprove={onApprove}
            onReject={onReject}
          />
        </header>

        {error ? (
          <div className="rounded-lg border border-amber-500/40 bg-amber-500/10 p-3 text-amber-100">
            {error}
          </div>
        ) : null}

        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
          <Stat label="Epoch" value={String(snapshot.epoch)} />
          <Stat label="Accuracy" value={latest ? `${Math.round(latest.accuracy * 100)}%` : "0%"} />
          <Stat label="Validation loss" value={latest ? latest.validationLoss.toFixed(3) : "-"} />
          <Stat label="Rollbacks" value={String(snapshot.rollbackCount)} />
        </div>

        <div className="grid gap-5 xl:grid-cols-[1.2fr_0.8fr]">
          <section className="grid gap-5">
            <Panel title="Live training curves">
              <div className="grid gap-5 lg:grid-cols-2">
                <Chart
                  data={snapshot.metrics}
                  lines={[
                    ["trainLoss", "#38bdf8"],
                    ["validationLoss", "#f59e0b"],
                  ]}
                />
                <Chart
                  data={snapshot.metrics}
                  lines={[
                    ["accuracy", "#22c55e"],
                    ["learningRate", "#a78bfa"],
                  ]}
                />
              </div>
            </Panel>

            <Panel title="Current validation predictions">
              <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
                {snapshot.predictions.map((prediction) => (
                  <div
                    key={prediction.id}
                    className={`rounded-lg border p-2 ${
                      prediction.correct
                        ? "border-emerald-500/40 bg-emerald-500/10"
                        : "border-rose-500/40 bg-rose-500/10"
                    }`}
                  >
                    <img
                      className="h-24 w-full rounded-md object-cover image-render-pixel"
                      src={prediction.image}
                      alt={prediction.actualClass}
                    />
                    <div className="mt-2 flex items-center justify-between gap-2 text-sm">
                      <span>{prediction.predictedClass}</span>
                      {prediction.correct ? (
                        <CheckCircle2 className="text-emerald-300" size={16} />
                      ) : (
                        <XCircle className="text-rose-300" size={16} />
                      )}
                    </div>
                    <p className="text-xs text-slate-400">
                      actual {prediction.actualClass} / {Math.round(prediction.confidence * 100)}%
                    </p>
                  </div>
                ))}
              </div>
            </Panel>
          </section>

          <aside className="grid gap-5">
            <Panel title="Controller state">
              <div className="grid gap-3 text-sm">
                <KeyValue label="Status" value={snapshot.status} />
                <KeyValue label="Stage" value={snapshot.stage} />
                <KeyValue label="Controller" value={snapshot.controllerState} />
                <KeyValue label="Branch" value={snapshot.currentBranch} />
                <KeyValue label="Checkpoint" value={snapshot.activeCheckpointId ?? "-"} />
                <KeyValue
                  label="GPU"
                  value={`${snapshot.gpuUsage.device ?? "cpu"} ${
                    snapshot.gpuUsage.memoryAllocatedMb
                      ? `${snapshot.gpuUsage.memoryAllocatedMb} MB`
                      : ""
                  }`}
                />
              </div>
            </Panel>

            <Panel title="Checkpoint timeline">
              <div className="space-y-2">
                {snapshot.checkpoints.map((checkpoint) => (
                  <div
                    key={checkpoint.id}
                    className="rounded-md border border-slate-800 bg-slate-950 p-3 text-sm"
                  >
                    <div className="flex items-center justify-between">
                      <span className="font-medium">{checkpoint.id}</span>
                      <span
                        className={
                          checkpoint.stable ? "text-emerald-300" : "text-slate-500"
                        }
                      >
                        {checkpoint.stable ? "stable" : "candidate"}
                      </span>
                    </div>
                    <p className="mt-1 text-slate-400">
                      epoch {checkpoint.epoch} / acc {Math.round(checkpoint.accuracy * 100)}%
                    </p>
                  </div>
                ))}
              </div>
            </Panel>

            <Panel title="Adaptive event feed">
              <div className="max-h-72 space-y-2 overflow-auto pr-1">
                {snapshot.events
                  .slice()
                  .reverse()
                  .map((event) => (
                    <div
                      key={event.id}
                      className="rounded-md border border-slate-800 bg-slate-950 p-3 text-sm"
                    >
                      <p className="font-medium">{event.message}</p>
                      <p className="text-xs text-slate-500">{event.createdAt}</p>
                    </div>
                  ))}
              </div>
            </Panel>

            <Panel title="Final inference test">
              <label className="flex cursor-pointer items-center justify-center gap-2 rounded-md border border-dashed border-slate-700 p-4 text-sm text-slate-300 hover:bg-slate-800">
                <Upload size={18} />
                Upload image for current model
                <input
                  className="hidden"
                  type="file"
                  accept="image/*"
                  onChange={(event) => void handleUpload(event, onPredict)}
                />
              </label>
              {inference ? (
                <div className="mt-3 rounded-md border border-cyan-500/40 bg-cyan-500/10 p-3">
                  <p className="text-sm text-slate-300">Prediction</p>
                  <p className="text-xl font-semibold">{inference.predictedClass}</p>
                  <p className="text-sm text-slate-400">
                    confidence {Math.round(inference.confidence * 100)}% / checkpoint{" "}
                    {inference.modelCheckpoint}
                  </p>
                </div>
              ) : null}
            </Panel>
          </aside>
        </div>
      </section>
    </main>
  );
}

function ControlPanel({
  snapshot,
  pendingDecisionId,
  onStart,
  onPause,
  onResume,
  onRollback,
  onSetAutoMode,
  onApprove,
  onReject,
}: {
  snapshot: VisualDemoSnapshot;
  pendingDecisionId?: string;
  onStart: (autoMode: boolean) => void;
  onPause: () => void;
  onResume: () => void;
  onRollback: () => void;
  onSetAutoMode: (enabled: boolean) => void;
  onApprove: (decisionId: string) => void;
  onReject: (decisionId: string) => void;
}) {
  return (
    <div className="flex flex-wrap items-center gap-2">
      <Button onClick={() => onStart(snapshot.autoMode)} icon={<Play size={16} />}>
        Start
      </Button>
      <Button onClick={onPause} icon={<Pause size={16} />}>
        Pause
      </Button>
      <Button onClick={onResume} icon={<Play size={16} />}>
        Resume
      </Button>
      <Button onClick={onRollback} icon={<RotateCcw size={16} />}>
        Rollback
      </Button>
      <Button onClick={() => onSetAutoMode(!snapshot.autoMode)} icon={<SlidersHorizontal size={16} />}>
        {snapshot.autoMode ? "AUTO" : "MANUAL"}
      </Button>
      {pendingDecisionId ? (
        <>
          <Button onClick={() => onApprove(pendingDecisionId)} icon={<ShieldAlert size={16} />}>
            Approve
          </Button>
          <Button onClick={() => onReject(pendingDecisionId)} icon={<XCircle size={16} />}>
            Reject
          </Button>
        </>
      ) : null}
    </div>
  );
}

function Button({
  children,
  icon,
  onClick,
}: {
  children: string;
  icon: ReactNode;
  onClick: () => void;
}) {
  return (
    <button
      className="inline-flex items-center gap-2 rounded-md border border-slate-700 bg-slate-950 px-3 py-2 text-sm font-medium text-slate-100 hover:bg-slate-800"
      type="button"
      onClick={onClick}
    >
      {icon}
      {children}
    </button>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-slate-800 bg-slate-900 p-4">
      <p className="text-xs font-semibold uppercase text-slate-400">{label}</p>
      <p className="mt-2 text-3xl font-semibold">{value}</p>
    </div>
  );
}

function Panel({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="rounded-lg border border-slate-800 bg-slate-900 p-4">
      <h2 className="mb-3 text-lg font-semibold">{title}</h2>
      {children}
    </section>
  );
}

function KeyValue({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between gap-4 rounded-md bg-slate-950 px-3 py-2">
      <span className="text-slate-400">{label}</span>
      <span className="text-right font-medium">{value}</span>
    </div>
  );
}

function Chart({
  data,
  lines,
}: {
  data: VisualDemoSnapshot["metrics"];
  lines: [string, string][];
}) {
  return (
    <div className="h-72">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data}>
          <CartesianGrid stroke="#1e293b" />
          <XAxis dataKey="epoch" stroke="#94a3b8" />
          <YAxis stroke="#94a3b8" width={44} />
          <Tooltip
            contentStyle={{
              background: "#020617",
              border: "1px solid #334155",
              borderRadius: 8,
            }}
          />
          {lines.map(([key, color]) => (
            <Line
              key={key}
              type="monotone"
              dataKey={key}
              stroke={color}
              strokeWidth={3}
              dot={false}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

async function handleUpload(
  event: ChangeEvent<HTMLInputElement>,
  onPredict: (imageDataUrl: string) => void,
) {
  const file = event.target.files?.[0];
  if (!file) return;
  const imageDataUrl = await readFile(file);
  onPredict(imageDataUrl);
}

function readFile(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(file);
  });
}
