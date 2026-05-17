import { useMemo, useState } from "react";

import { EmptyState } from "../components/EmptyState";
import { Panel } from "../components/Panel";
import type { ControllerDecision, OverridePayload } from "../types/dashboard";

export function OverrideConsole({
  decisions,
  onSubmit,
}: {
  decisions: ControllerDecision[];
  onSubmit: (payload: OverridePayload) => Promise<void>;
}) {
  const pendingDecisions = useMemo(
    () => decisions.filter((decision) => decision.status === "pending" || decision.status === "denied"),
    [decisions],
  );
  const [decisionId, setDecisionId] = useState("");
  const [approvedBy, setApprovedBy] = useState("");
  const [reason, setReason] = useState("");
  const [ticketId, setTicketId] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [message, setMessage] = useState<string | null>(null);

  async function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setSubmitting(true);
    setMessage(null);
    try {
      await onSubmit({
        decisionId,
        approvedBy,
        reason,
        ticketId: ticketId || undefined,
      });
      setMessage("Override submitted.");
      setReason("");
      setTicketId("");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Override failed.");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <Panel title="Override Console" description="Human approval path for Citadel-supported overrides.">
      {pendingDecisions.length === 0 ? (
        <EmptyState label="No pending or denied decisions are available for override." />
      ) : (
        <form className="grid gap-4 lg:grid-cols-2" onSubmit={handleSubmit}>
          <label className="grid gap-2 text-sm">
            Decision
            <select
              className="rounded-md border border-slate-300 bg-white px-3 py-2 dark:border-slate-700 dark:bg-slate-950"
              value={decisionId}
              required
              onChange={(event) => setDecisionId(event.target.value)}
            >
              <option value="">Select decision</option>
              {pendingDecisions.map((decision) => (
                <option key={decision.id} value={decision.id}>
                  {decision.action} / {decision.branchName}
                </option>
              ))}
            </select>
          </label>
          <label className="grid gap-2 text-sm">
            Approved by
            <input
              className="rounded-md border border-slate-300 bg-white px-3 py-2 dark:border-slate-700 dark:bg-slate-950"
              value={approvedBy}
              required
              onChange={(event) => setApprovedBy(event.target.value)}
            />
          </label>
          <label className="grid gap-2 text-sm lg:col-span-2">
            Reason
            <textarea
              className="min-h-28 rounded-md border border-slate-300 bg-white px-3 py-2 dark:border-slate-700 dark:bg-slate-950"
              value={reason}
              required
              onChange={(event) => setReason(event.target.value)}
            />
          </label>
          <label className="grid gap-2 text-sm">
            Ticket
            <input
              className="rounded-md border border-slate-300 bg-white px-3 py-2 dark:border-slate-700 dark:bg-slate-950"
              value={ticketId}
              onChange={(event) => setTicketId(event.target.value)}
            />
          </label>
          <div className="flex items-end gap-3">
            <button
              className="rounded-md bg-slate-900 px-4 py-2 text-sm font-medium text-white disabled:opacity-60 dark:bg-slate-100 dark:text-slate-950"
              type="submit"
              disabled={submitting}
            >
              Submit Override
            </button>
            {message ? <p className="text-sm text-slate-500">{message}</p> : null}
          </div>
        </form>
      )}
    </Panel>
  );
}
