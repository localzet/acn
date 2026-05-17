import { appConfig } from "../config";
import type { DashboardSnapshot, OverridePayload } from "../types/dashboard";

export const emptySnapshot: DashboardSnapshot = {
  commitGraph: { nodes: [], edges: [] },
  branchGraph: { nodes: [], edges: [] },
  metricsTimeline: [],
  experiments: [],
  controllerDecisions: [],
  rollbackHistory: [],
  liveLogs: [],
};

export async function fetchDashboardSnapshot(signal?: AbortSignal): Promise<DashboardSnapshot> {
  const response = await fetch(`${appConfig.apiBaseUrl}/api/v1/dashboard/snapshot`, {
    headers: { Accept: "application/json" },
    signal,
  });
  if (!response.ok) {
    throw new Error(`Dashboard snapshot request failed with ${response.status}`);
  }
  return (await response.json()) as DashboardSnapshot;
}

export async function submitOverride(payload: OverridePayload): Promise<void> {
  const response = await fetch(`${appConfig.apiBaseUrl}/api/v1/overrides`, {
    method: "POST",
    headers: {
      Accept: "application/json",
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    throw new Error(`Override request failed with ${response.status}`);
  }
}
