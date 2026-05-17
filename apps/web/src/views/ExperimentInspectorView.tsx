import { EmptyState } from "../components/EmptyState";
import { Panel } from "../components/Panel";
import type { ExperimentSummary } from "../types/dashboard";

export function ExperimentInspectorView({ experiments }: { experiments: ExperimentSummary[] }) {
  return (
    <Panel title="Experiment Inspector" description="Persisted experiment lifecycle and active state.">
      {experiments.length === 0 ? (
        <EmptyState label="No experiments returned by the API." />
      ) : (
        <div className="grid gap-3 xl:grid-cols-2">
          {experiments.map((experiment) => (
            <article
              key={experiment.id}
              className="rounded-md border border-slate-200 p-4 dark:border-slate-800"
            >
              <div className="flex items-start justify-between gap-3">
                <div>
                  <h4 className="font-semibold">{experiment.name}</h4>
                  <p className="mt-1 text-sm text-slate-500">{experiment.id}</p>
                </div>
                <span className="rounded-md bg-slate-100 px-2 py-1 text-xs font-medium uppercase tracking-normal dark:bg-slate-800">
                  {experiment.status}
                </span>
              </div>
              <dl className="mt-4 grid grid-cols-1 gap-3 text-sm sm:grid-cols-2">
                <KeyValue label="Branch" value={experiment.branchName} />
                <KeyValue label="Stage" value={experiment.currentStageId ?? "none"} />
                <KeyValue label="Commit" value={experiment.currentCommitId ?? "none"} />
                <KeyValue label="Best" value={experiment.bestCommitId ?? "none"} />
              </dl>
            </article>
          ))}
        </div>
      )}
    </Panel>
  );
}

function KeyValue({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt className="text-xs font-medium uppercase tracking-normal text-slate-500">{label}</dt>
      <dd className="mt-1 truncate font-mono text-xs">{value}</dd>
    </div>
  );
}
