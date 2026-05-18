import { appConfig } from "../config";
import type { VisualDemoInference, VisualDemoSnapshot } from "../types/visualDemo";

const demoBaseUrl = `${appConfig.apiBaseUrl}/api/v1/demo`;

export const emptyVisualDemoSnapshot: VisualDemoSnapshot = {
  status: "idle",
  autoMode: true,
  epoch: 0,
  stage: "ready",
  controllerState: "idle",
  currentBranch: "main",
  activeCheckpointId: null,
  rollbackCount: 0,
  gpuUsage: { device: "cpu", memoryAllocatedMb: null, memoryReservedMb: null },
  metrics: [],
  checkpoints: [],
  predictions: [],
  events: [],
  decisions: [],
  runtimeStatus: {
    postgres: { connected: false, message: "not loaded" },
    mlflow: { connected: false, message: "not loaded" },
    minio: { connected: false, message: "not loaded" },
    artifactStorage: { connected: false, message: "not loaded" },
  },
  mlflowRunId: null,
  artifacts: [],
};

export async function fetchVisualDemoState(): Promise<VisualDemoSnapshot> {
  return requestSnapshot("/state");
}

export async function startVisualDemo(autoMode: boolean): Promise<VisualDemoSnapshot> {
  return requestSnapshot("/start", { auto_mode: autoMode });
}

export async function pauseVisualDemo(): Promise<VisualDemoSnapshot> {
  return requestSnapshot("/pause");
}

export async function resumeVisualDemo(): Promise<VisualDemoSnapshot> {
  return requestSnapshot("/resume");
}

export async function rollbackVisualDemo(): Promise<VisualDemoSnapshot> {
  return requestSnapshot("/rollback");
}

export async function setVisualDemoAutoMode(enabled: boolean): Promise<VisualDemoSnapshot> {
  return requestSnapshot("/auto-mode", { enabled });
}

export async function approveVisualDemoDecision(decisionId: string): Promise<VisualDemoSnapshot> {
  return requestSnapshot("/approve", { decision_id: decisionId });
}

export async function rejectVisualDemoDecision(decisionId: string): Promise<VisualDemoSnapshot> {
  return requestSnapshot("/reject", { decision_id: decisionId });
}

export async function predictVisualDemoImage(imageDataUrl: string): Promise<VisualDemoInference> {
  const response = await fetch(`${demoBaseUrl}/predict`, {
    method: "POST",
    headers: {
      Accept: "application/json",
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ image_data_url: imageDataUrl }),
  });
  if (!response.ok) {
    throw new Error(`Prediction failed with ${response.status}`);
  }
  return (await response.json()) as VisualDemoInference;
}

export function visualDemoEventsUrl(): string {
  return `${demoBaseUrl}/events`;
}

async function requestSnapshot(path: string, body?: object): Promise<VisualDemoSnapshot> {
  const response = await fetch(`${demoBaseUrl}${path}`, {
    method: body ? "POST" : "GET",
    headers: {
      Accept: "application/json",
      ...(body ? { "Content-Type": "application/json" } : {}),
    },
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!response.ok) {
    throw new Error(`Visual demo request failed with ${response.status}`);
  }
  return (await response.json()) as VisualDemoSnapshot;
}
