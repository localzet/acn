export type CommitNode = {
  id: string;
  branchId: string;
  checkpointId: string;
  message: string;
  createdAt: string;
  metrics: Record<string, number | string | boolean | null>;
};

export type GraphEdge = {
  parentId: string;
  childId: string;
};

export type CommitGraph = {
  nodes: CommitNode[];
  edges: GraphEdge[];
};

export type BranchNode = {
  id: string;
  name: string;
  headCommitId: string | null;
  baseCommitId: string | null;
  status?: string;
};

export type BranchGraph = {
  nodes: BranchNode[];
  edges: GraphEdge[];
};

export type MetricPoint = {
  timestamp: string;
  stageId: string;
  trainLoss: number | null;
  validationLoss: number | null;
  trainAccuracy: number | null;
  validationAccuracy: number | null;
  forgettingScore: number | null;
  oldClassRetention: number | null;
  newClassAdaptation: number | null;
};

export type ExperimentSummary = {
  id: string;
  name: string;
  status: string;
  branchName: string;
  currentStageId: string | null;
  currentCommitId: string | null;
  bestCommitId: string | null;
  updatedAt: string;
};

export type ControllerDecision = {
  id: string;
  action: string;
  confidence: number;
  branchName: string;
  commitId: string | null;
  reasons: string[];
  createdAt: string;
  status: "pending" | "approved" | "denied" | "executed";
};

export type RollbackEvent = {
  id: string;
  branchName: string;
  fromCommitId: string;
  toCommitId: string;
  actor: string;
  createdAt: string;
  reason: string;
};

export type LiveLogEntry = {
  id: string;
  level: "debug" | "info" | "warning" | "error";
  source: string;
  message: string;
  createdAt: string;
};

export type DashboardSnapshot = {
  commitGraph: CommitGraph;
  branchGraph: BranchGraph;
  metricsTimeline: MetricPoint[];
  experiments: ExperimentSummary[];
  controllerDecisions: ControllerDecision[];
  rollbackHistory: RollbackEvent[];
  liveLogs: LiveLogEntry[];
};

export type DashboardEvent =
  | { type: "snapshot"; payload: DashboardSnapshot }
  | { type: "log"; payload: LiveLogEntry }
  | { type: "decision"; payload: ControllerDecision }
  | { type: "metrics"; payload: MetricPoint }
  | { type: "rollback"; payload: RollbackEvent };

export type OverridePayload = {
  decisionId: string;
  approvedBy: string;
  reason: string;
  ticketId?: string;
};

export type DemoPlaybackStep = {
  id: string;
  title: string;
  subtitle: string;
  focus:
    | "experiment"
    | "commit"
    | "metrics"
    | "branch"
    | "forgetting"
    | "rollback"
    | "override"
    | "summary";
};
