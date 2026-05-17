import { useCallback, useEffect, useState } from "react";

import {
  emptySnapshot,
  fetchDashboardSnapshot,
  submitOverride as submitOverrideRequest,
} from "../api/dashboardApi";
import { connectLiveUpdates } from "../api/liveUpdates";
import type { DashboardEvent, DashboardSnapshot, OverridePayload } from "../types/dashboard";

type ConnectionState = "connected" | "connecting" | "disconnected";

export function useDashboardData() {
  const [snapshot, setSnapshot] = useState<DashboardSnapshot>(emptySnapshot);
  const [connection, setConnection] = useState<ConnectionState>("disconnected");
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    const controller = new AbortController();
    try {
      const nextSnapshot = await fetchDashboardSnapshot(controller.signal);
      setSnapshot(nextSnapshot);
      setError(null);
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Dashboard request failed");
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  useEffect(() => {
    const connection = connectLiveUpdates(
      (event) => setSnapshot((current) => applyDashboardEvent(current, event)),
      setConnection,
    );
    return () => connection.close();
  }, []);

  const submitOverride = useCallback(
    async (payload: OverridePayload) => {
      await submitOverrideRequest(payload);
      await refresh();
    },
    [refresh],
  );

  return { snapshot, connection, error, refresh, submitOverride };
}

function applyDashboardEvent(snapshot: DashboardSnapshot, event: DashboardEvent): DashboardSnapshot {
  if (event.type === "snapshot") {
    return event.payload;
  }
  if (event.type === "log") {
    return { ...snapshot, liveLogs: [event.payload, ...snapshot.liveLogs].slice(0, 200) };
  }
  if (event.type === "decision") {
    return {
      ...snapshot,
      controllerDecisions: [event.payload, ...snapshot.controllerDecisions],
    };
  }
  if (event.type === "metrics") {
    return {
      ...snapshot,
      metricsTimeline: [...snapshot.metricsTimeline, event.payload],
    };
  }
  if (event.type === "rollback") {
    return {
      ...snapshot,
      rollbackHistory: [event.payload, ...snapshot.rollbackHistory],
    };
  }
  return snapshot;
}
