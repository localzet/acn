import { useEffect, useMemo, useState } from "react";

import { demoSnapshot, demoSteps } from "../demo/demoPreset";
import type { DashboardSnapshot } from "../types/dashboard";

const STEP_INTERVAL_MS = 3200;

export function useDemoPlayback() {
  const [stepIndex, setStepIndex] = useState(0);
  const [playing, setPlaying] = useState(true);

  useEffect(() => {
    if (!playing) return undefined;
    const timer = window.setInterval(() => {
      setStepIndex((current) => (current + 1) % demoSteps.length);
    }, STEP_INTERVAL_MS);
    return () => window.clearInterval(timer);
  }, [playing]);

  const snapshot = useMemo(() => buildSnapshotForStep(stepIndex), [stepIndex]);

  return {
    snapshot,
    step: demoSteps[stepIndex],
    stepIndex,
    stepCount: demoSteps.length,
    playing,
    progress: (stepIndex + 1) / demoSteps.length,
    play: () => setPlaying(true),
    pause: () => setPlaying(false),
    next: () => setStepIndex((current) => Math.min(current + 1, demoSteps.length - 1)),
    previous: () => setStepIndex((current) => Math.max(current - 1, 0)),
    restart: () => setStepIndex(0),
  };
}

function buildSnapshotForStep(stepIndex: number): DashboardSnapshot {
  const visibleCommits = Math.min(demoSnapshot.commitGraph.nodes.length, Math.max(1, stepIndex));
  const visibleMetrics = Math.min(demoSnapshot.metricsTimeline.length, Math.max(1, stepIndex));
  const branchCount = stepIndex >= 6 ? 3 : stepIndex >= 3 ? 2 : 1;
  const decisionCount = stepIndex >= 6 ? 3 : stepIndex >= 5 ? 2 : stepIndex >= 3 ? 1 : 0;

  return {
    ...demoSnapshot,
    commitGraph: {
      nodes: demoSnapshot.commitGraph.nodes.slice(0, visibleCommits),
      edges: demoSnapshot.commitGraph.edges.filter((edge) =>
        demoSnapshot.commitGraph.nodes
          .slice(0, visibleCommits)
          .some((node) => node.id === edge.childId),
      ),
    },
    branchGraph: {
      nodes: demoSnapshot.branchGraph.nodes.slice(0, branchCount),
      edges: demoSnapshot.branchGraph.edges.slice(0, Math.max(0, branchCount - 1)),
    },
    metricsTimeline: demoSnapshot.metricsTimeline.slice(0, visibleMetrics),
    controllerDecisions: demoSnapshot.controllerDecisions.slice(0, decisionCount),
    rollbackHistory: stepIndex >= 5 ? demoSnapshot.rollbackHistory : [],
    liveLogs: demoSnapshot.liveLogs.slice(0, Math.min(demoSnapshot.liveLogs.length, stepIndex + 1)),
  };
}
