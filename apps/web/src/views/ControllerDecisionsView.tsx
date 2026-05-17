import { EmptyState } from "../components/EmptyState";
import { Panel } from "../components/Panel";
import type { ControllerDecision } from "../types/dashboard";

export function ControllerDecisionsView({
  decisions,
}: {
  decisions: ControllerDecision[];
}) {
  return (
    <Panel
      title="Controller Decisions"
      description="Explainable adaptive policy decisions emitted by the controller."
    >
      {decisions.length === 0 ? (
        <EmptyState label="No controller decisions returned by the API." />
      ) : (
        <div className="space-y-3">
          {decisions.map((decision) => (
            <article
              key={decision.id}
              className="rounded-md border border-slate-200 p-4 dark:border-slate-800"
            >
              <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
                <div>
                  <h4 className="font-semibold">{decision.action}</h4>
                  <p className="text-sm text-slate-500">
                    {decision.branchName} / {decision.commitId ?? "no commit"}
                  </p>
                </div>
                <span className="rounded-md bg-slate-100 px-2 py-1 text-xs font-medium uppercase tracking-normal dark:bg-slate-800">
                  {decision.status} · {(decision.confidence * 100).toFixed(0)}%
                </span>
              </div>
              <ul className="mt-3 list-disc space-y-1 pl-5 text-sm text-slate-600 dark:text-slate-300">
                {decision.reasons.map((reason) => (
                  <li key={reason}>{reason}</li>
                ))}
              </ul>
            </article>
          ))}
        </div>
      )}
    </Panel>
  );
}
