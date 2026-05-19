import { useCallback, useEffect, useState } from "react";

import {
  approveVisualDemoDecision,
  emptyVisualDemoSnapshot,
  compareVisualDemoImage,
  exportVisualDemoReport,
  fetchVisualDemoState,
  pauseVisualDemo,
  predictVisualDemoImage,
  rejectVisualDemoDecision,
  resumeVisualDemo,
  rollbackVisualDemo,
  setVisualDemoAutoMode,
  startVisualDemo,
  visualDemoEventsUrl,
} from "../api/visualDemoApi";
import type { VisualDemoInference, VisualDemoSnapshot } from "../types/visualDemo";
import type { VisualDemoComparison } from "../types/visualDemo";

export function useVisualDemo() {
  const [snapshot, setSnapshot] = useState<VisualDemoSnapshot>(emptyVisualDemoSnapshot);
  const [error, setError] = useState<string | null>(null);
  const [inference, setInference] = useState<VisualDemoInference | null>(null);
  const [comparison, setComparison] = useState<VisualDemoComparison | null>(null);
  const [exportPaths, setExportPaths] = useState<Record<string, string> | null>(null);

  const refresh = useCallback(async () => {
    try {
      setSnapshot(await fetchVisualDemoState());
      setError(null);
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Visual demo request failed");
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  useEffect(() => {
    const source = new EventSource(visualDemoEventsUrl());
    source.onmessage = (message) => {
      const event = JSON.parse(message.data) as { type: string; payload: VisualDemoSnapshot };
      if (event.type === "snapshot") {
        setSnapshot(event.payload);
      }
    };
    source.onerror = () => setError("Live demo stream disconnected");
    return () => source.close();
  }, []);

  const start = useCallback(async (autoMode: boolean) => setSnapshot(await startVisualDemo(autoMode)), []);
  const pause = useCallback(async () => setSnapshot(await pauseVisualDemo()), []);
  const resume = useCallback(async () => setSnapshot(await resumeVisualDemo()), []);
  const rollback = useCallback(async () => setSnapshot(await rollbackVisualDemo()), []);
  const setAutoMode = useCallback(
    async (enabled: boolean) => setSnapshot(await setVisualDemoAutoMode(enabled)),
    [],
  );
  const approve = useCallback(
    async (decisionId: string) => setSnapshot(await approveVisualDemoDecision(decisionId)),
    [],
  );
  const reject = useCallback(
    async (decisionId: string) => setSnapshot(await rejectVisualDemoDecision(decisionId)),
    [],
  );
  const predict = useCallback(async (imageDataUrl: string, checkpointId?: string) => {
    setInference(await predictVisualDemoImage(imageDataUrl, checkpointId));
  }, []);
  const compare = useCallback(
    async (imageDataUrl: string, baselineCheckpointId?: string, candidateCheckpointId?: string) => {
      setComparison(
        await compareVisualDemoImage(imageDataUrl, baselineCheckpointId, candidateCheckpointId),
      );
    },
    [],
  );
  const exportReport = useCallback(async () => {
    setExportPaths(await exportVisualDemoReport());
  }, []);

  return {
    snapshot,
    error,
    inference,
    comparison,
    exportPaths,
    start,
    pause,
    resume,
    rollback,
    setAutoMode,
    approve,
    reject,
    predict,
    compare,
    exportReport,
  };
}
