export type VisualDemoStatus =
  | "idle"
  | "running"
  | "paused"
  | "awaiting_approval"
  | "completed"
  | "failed";

export type VisualDemoMetric = {
  timestamp: string;
  epoch: number;
  trainLoss: number;
  validationLoss: number;
  accuracy: number;
  learningRate: number;
  stage: string;
};

export type VisualDemoCheckpoint = {
  id: string;
  epoch: number;
  validationLoss: number;
  accuracy: number;
  stable: boolean;
  createdAt: string;
  artifactUri: string | null;
  sizeBytes: number | null;
  mlflowRunId: string | null;
  storage: string | null;
};

export type VisualDemoPrediction = {
  id: string;
  image: string;
  actualClass: string;
  predictedClass: string;
  confidence: number;
  correct: boolean;
};

export type VisualDemoEvent = {
  id: string;
  level: string;
  message: string;
  createdAt: string;
};

export type VisualDemoDecision = {
  id: string;
  action: string;
  status: string;
  reason: string;
  createdAt: string;
};

export type VisualDemoSnapshot = {
  status: VisualDemoStatus;
  autoMode: boolean;
  epoch: number;
  stage: string;
  controllerState: string;
  currentBranch: string;
  activeCheckpointId: string | null;
  rollbackCount: number;
  gpuUsage: {
    device: string | null;
    memoryAllocatedMb: number | null;
    memoryReservedMb: number | null;
  };
  metrics: VisualDemoMetric[];
  checkpoints: VisualDemoCheckpoint[];
  predictions: VisualDemoPrediction[];
  events: VisualDemoEvent[];
  decisions: VisualDemoDecision[];
  runtimeStatus: Record<string, { connected: boolean; message: string }>;
  mlflowRunId: string | null;
  artifacts: VisualDemoArtifact[];
};

export type VisualDemoInference = {
  predictedClass: string;
  confidence: number;
  modelCheckpoint: string;
};

export type VisualDemoArtifact = {
  commitId: string;
  checkpointId: string;
  artifactUri: string;
  artifactSizeBytes: number | null;
  storage: string;
  mlflowRunId: string | null;
  validationLoss: number | null;
  accuracy: number | null;
};
