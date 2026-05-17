import { Background, Controls, MiniMap, ReactFlow, type Edge, type Node } from "@xyflow/react";
import { useMemo } from "react";

import { EmptyState } from "../components/EmptyState";
import { Panel } from "../components/Panel";
import type { CommitGraph } from "../types/dashboard";

export function CommitGraphView({ graph }: { graph: CommitGraph }) {
  const { nodes, edges } = useMemo(() => buildGraph(graph), [graph]);

  return (
    <Panel
      title="Commit Graph"
      description="Immutable checkpoint commits and parent-child relationships from the version store."
    >
      {nodes.length === 0 ? (
        <EmptyState label="No commit graph data returned by the API." />
      ) : (
        <div className="h-[620px] overflow-hidden rounded-md border border-slate-200 dark:border-slate-800">
          <ReactFlow nodes={nodes} edges={edges} fitView>
            <Background />
            <MiniMap pannable zoomable />
            <Controls />
          </ReactFlow>
        </div>
      )}
    </Panel>
  );
}

function buildGraph(graph: CommitGraph): { nodes: Node[]; edges: Edge[] } {
  return {
    nodes: graph.nodes.map((node, index) => ({
      id: node.id,
      position: { x: (index % 4) * 260, y: Math.floor(index / 4) * 160 },
      data: {
        label: (
          <div className="max-w-48 text-left">
            <p className="truncate text-sm font-semibold">{node.message || node.id}</p>
            <p className="truncate text-xs text-slate-500">branch {node.branchId}</p>
          </div>
        ),
      },
      type: "default",
    })),
    edges: graph.edges.map((edge) => ({
      id: `${edge.parentId}-${edge.childId}`,
      source: edge.parentId,
      target: edge.childId,
      animated: false,
    })),
  };
}
