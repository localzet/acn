import { EmptyState } from "../components/EmptyState";
import { Panel } from "../components/Panel";
import type { RollbackEvent } from "../types/dashboard";

export function RollbackHistoryView({ history }: { history: RollbackEvent[] }) {
  return (
    <Panel title="Rollback History" description="Audited rollback operations and target commits.">
      {history.length === 0 ? (
        <EmptyState label="No rollback history returned by the API." />
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full min-w-[760px] text-left text-sm">
            <thead className="border-b border-slate-200 text-xs uppercase tracking-normal text-slate-500 dark:border-slate-800">
              <tr>
                <th className="py-2 pr-4">Branch</th>
                <th className="py-2 pr-4">From</th>
                <th className="py-2 pr-4">To</th>
                <th className="py-2 pr-4">Actor</th>
                <th className="py-2 pr-4">Reason</th>
              </tr>
            </thead>
            <tbody>
              {history.map((event) => (
                <tr key={event.id} className="border-b border-slate-100 dark:border-slate-800">
                  <td className="py-3 pr-4">{event.branchName}</td>
                  <td className="py-3 pr-4 font-mono text-xs">{event.fromCommitId}</td>
                  <td className="py-3 pr-4 font-mono text-xs">{event.toCommitId}</td>
                  <td className="py-3 pr-4">{event.actor}</td>
                  <td className="py-3 pr-4">{event.reason}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Panel>
  );
}
