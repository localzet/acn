import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { EmptyState } from "../components/EmptyState";
import { Panel } from "../components/Panel";
import type { MetricPoint } from "../types/dashboard";

export function MetricsTimelineView({ metrics }: { metrics: MetricPoint[] }) {
  return (
    <Panel
      title="Metrics Timeline"
      description="Training, validation, forgetting, retention, and adaptation metrics over stages."
    >
      {metrics.length === 0 ? (
        <EmptyState label="No metrics timeline data returned by the API." />
      ) : (
        <div className="h-[560px]">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={metrics} margin={{ top: 12, right: 24, bottom: 12, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="stageId" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Line dataKey="trainLoss" stroke="#2563eb" dot={false} name="Train loss" />
              <Line dataKey="validationLoss" stroke="#dc2626" dot={false} name="Validation loss" />
              <Line dataKey="oldClassRetention" stroke="#059669" dot={false} name="Retention" />
              <Line dataKey="newClassAdaptation" stroke="#7c3aed" dot={false} name="Adaptation" />
              <Line dataKey="forgettingScore" stroke="#d97706" dot={false} name="Forgetting" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </Panel>
  );
}
