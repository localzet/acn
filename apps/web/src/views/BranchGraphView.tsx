import { Background, Controls, ReactFlow, type Edge, type Node } from "@xyflow/react";
import { useMemo } from "react";

import { EmptyState } from "../components/EmptyState";
import { Panel } from "../components/Panel";
import type { BranchGraph } from "../types/dashboard";

export function BranchGraphView({ graph }: { graph: BranchGraph }) {
  const { nodes, edges } = useMemo(() => buildGraph(graph), [graph]);

  return (
    <Panel
      title="Branch Graph"
      description="Branch heads, base commits, and experiment branches for training evolution."
    >
      {nodes.length === 0 ? (
        <EmptyState label="No branch graph data returned by the API." />
      ) : (
        <div className="h-[620px] overflow-hidden rounded-md border border-slate-200 dark:border-slate-800">
          <ReactFlow nodes={nodes} edges={edges} fitView>
            <Background />
            <Controls />
          </ReactFlow>
        </div>
      )}
    </Panel>
  );
}

function buildGraph(graph: BranchGraph): { nodes: Node[]; edges: Edge[] } {
  return {
    nodes: graph.nodes.map((node, index) => ({
      id: node.id,
      position: { x: (index % 3) * 300, y: Math.floor(index / 3) * 180 },
      data: {
        label: (
          <div className="max-w-52 text-left">
            <p className="truncate text-sm font-semibold">{node.name}</p>
            <p className="truncate text-xs text-slate-500">head {node.headCommitId ?? "none"}</p>
          </div>
        ),
      },
    })),
    edges: graph.edges.map((edge) => ({
      id: `${edge.parentId}-${edge.childId}`,
      source: edge.parentId,
      target: edge.childId,
    })),
  };
}
