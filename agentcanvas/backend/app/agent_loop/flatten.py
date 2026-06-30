"""Recursive graph flattener — expands composite nodes before execution.

The GraphExecutor expects a flat graph (no nesting). This module
recursively expands any node that has a ``subgraph`` into the parent
graph, rewiring GraphIn/GraphOut boundary nodes to the parent edges.

Maintains a FlattenMap for error tracing back to original composite context.

All operations use typed :class:`GraphDefinition` / :class:`NodeDef` /
:class:`EdgeDef` from ``graph_def.py``.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field

from ..graph_def import AccessGrantDef, ContainerDef, EdgeDef, GraphDefinition, NodeDef

log = logging.getLogger("agentcanvas.flatten")


@dataclass
class FlattenMap:
    """Maps flattened node IDs to their original composite path."""

    flat_to_original: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def trace(self, flat_id: str) -> str:
        """Human-readable path for a flattened node ID."""
        path = self.flat_to_original.get(flat_id)
        if not path:
            return flat_id
        return " > ".join(path)


def flatten_graph(graph: GraphDefinition) -> tuple[GraphDefinition, FlattenMap]:
    """Recursively expand composite nodes into a flat graph.

    Returns the flattened ``GraphDefinition`` and a ``FlattenMap`` for error
    tracing.
    """
    fmap = FlattenMap()
    result = _flatten_recursive(graph, (), fmap)
    if fmap.flat_to_original:
        log.info(
            "Flattened %d composite nodes (%d total mapped nodes)",
            sum(1 for v in fmap.flat_to_original.values() if len(v) == 2),
            len(fmap.flat_to_original),
        )
    return result, fmap


def _flatten_recursive(
    graph: GraphDefinition,
    parent_path: tuple[str, ...],
    fmap: FlattenMap,
) -> GraphDefinition:
    """Recursively process one graph level."""
    nodes: list[NodeDef] = list(graph.nodes)
    edges: list[EdgeDef] = list(graph.edges)
    containers: list[ContainerDef] = list(graph.containers)
    access_grants: list[AccessGrantDef] = list(graph.access_grants)

    flat_nodes: list[NodeDef] = []
    new_edges: list[EdgeDef] = []
    new_containers: list[ContainerDef] = []
    new_access_grants: list[AccessGrantDef] = []
    nodes_to_remove: set = set()

    for node in nodes:
        # Only expand if the node has an embedded subgraph with nodes
        if node.subgraph is None or not node.subgraph.nodes:
            flat_nodes.append(node)
            continue

        composite_id = node.id
        node_path = (*parent_path, composite_id)
        nodes_to_remove.add(composite_id)

        # Deep copy to avoid mutating the original
        sub: GraphDefinition = copy.deepcopy(node.subgraph)

        # Prefix inner node IDs for uniqueness
        prefix = f"{composite_id}__"
        id_map: dict[str, str] = {}

        for inner_node in sub.nodes:
            old_id = inner_node.id
            new_id = f"{prefix}{old_id}"
            id_map[old_id] = new_id
            inner_node.id = new_id
            # Record in flatten map
            fmap.flat_to_original[new_id] = (*node_path, old_id)

        # Remap inner edges
        for inner_edge in sub.edges:
            inner_edge.id = f"{prefix}{inner_edge.id}"
            inner_edge.source = id_map.get(inner_edge.source, inner_edge.source)
            inner_edge.target = id_map.get(inner_edge.target, inner_edge.target)

        # Rewrite known id-bearing config fields against id_map. Today only
        # `pairedWith` carries inner-id refs — Initialize / iterOut both
        # reference their paired iterIn by string id, and iterIn references
        # iterOut the same way. Without this rewrite, a composite that
        # contains a three-pivot loop produces flat nodes whose
        # `config.pairedWith` still names the bare (un-prefixed) id and
        # `analyze_scopes` / `_synthesize_iterin_ports` cannot resolve the
        # inner triple. Future id-bearing config keys must be added to this
        # list (intentionally narrow to avoid stomping unrelated string
        # values that happen to match an inner id).
        for inner_node in sub.nodes:
            cfg = inner_node.config or {}
            pw = cfg.get("pairedWith")
            if isinstance(pw, str) and pw in id_map:
                cfg["pairedWith"] = id_map[pw]

        # Prefix and collect inner containers + access grants
        for inner_container in sub.containers:
            inner_container.id = f"{prefix}{inner_container.id}"
            new_containers.append(inner_container)
        for inner_ag in sub.access_grants:
            new_access_grants.append(
                AccessGrantDef(
                    id=f"{prefix}{inner_ag.id}",
                    node_id=id_map.get(inner_ag.node_id, f"{prefix}{inner_ag.node_id}"),
                    container_id=f"{prefix}{inner_ag.container_id}",
                )
            )

        # Identify graphIn/graphOut nodes for rewiring.
        #
        # Two flatten modes for boundary nodes:
        #
        #   strip  (default)  — composite is a pure DAG (no inner scope):
        #                       graphIn / graphOut are erased and their
        #                       parent edge + inner edge are spliced into a
        #                       single direct edge. Backward compat for the
        #                       17 existing composite graphs.
        #
        #   keep   (scoped)   — composite contains an inner scope (subgraph
        #                       has any iterIn): graphIn / graphOut nodes
        #                       SURVIVE flatten, kept as scope-boundary
        #                       latches so that the executor's per-scope
        #                       graphOut latch mechanism
        #                       (`_propagate_graphout_latches`, see
        #                       graph_executor.py:1204) can buffer
        #                       per-inner-iter writes and flush them once
        #                       on inner-scope termination, giving
        #                       function-return-value semantics across the
        #                       composite boundary. Without this, inner
        #                       producers wire DIRECTLY to outer consumers
        #                       (every per-iter value passes through), and
        #                       outer iterOut's required gates fill on the
        #                       first inner iter, causing outer to cycle
        #                       without waiting for inner termination
        #                       (TODO #59 tracks the cleaner long-term fix:
        #                       composite as a runtime first-class entity).
        sub_has_inner_scope = any(n.type == "iterIn" for n in sub.nodes)

        graph_in_map: dict[str, str] = {}  # portName -> prefixed graphIn node id
        graph_out_map: dict[str, str] = {}  # portName -> prefixed graphOut node id

        remaining_inner_nodes: list[NodeDef] = []
        for inner_node in sub.nodes:
            ntype = inner_node.type
            config = inner_node.config
            if ntype == "graphIn":
                port_name = config.get("portName", "input")
                graph_in_map[port_name] = inner_node.id
                if sub_has_inner_scope:
                    remaining_inner_nodes.append(inner_node)
            elif ntype == "graphOut":
                port_name = config.get("portName", "output")
                graph_out_map[port_name] = inner_node.id
                if sub_has_inner_scope:
                    remaining_inner_nodes.append(inner_node)
            else:
                remaining_inner_nodes.append(inner_node)

        # If no graphIn/graphOut, keep all nodes (backward compat — iterIn handles it)
        if not graph_in_map and not graph_out_map:
            remaining_inner_nodes = list(sub.nodes)

        # Rewire parent edges:
        #   strip mode  → splice composite.X edge with the inner edge it
        #                 anchored, dropping the boundary node from the chain.
        #   keep mode   → retarget the composite end of the parent edge onto
        #                 the boundary node itself (inputs land on graphIn,
        #                 outputs source from graphOut).
        remaining_parent_edges: list[EdgeDef] = []
        for edge in edges:
            if edge.target == composite_id and graph_in_map:
                handle = edge.targetHandle or "input"
                graph_in_id = graph_in_map.get(handle)
                if graph_in_id:
                    if sub_has_inner_scope:
                        # keep mode: outer producer → graphIn.value (boundary)
                        new_edges.append(
                            EdgeDef(
                                id=f"flat_{edge.id}_to_{graph_in_id}",
                                source=edge.source,
                                target=graph_in_id,
                                sourceHandle=edge.sourceHandle,
                                targetHandle="value",
                            )
                        )
                    else:
                        # strip mode: splice through to the inner consumer
                        for inner_edge in sub.edges:
                            if inner_edge.source == graph_in_id:
                                new_edges.append(
                                    EdgeDef(
                                        id=f"flat_{edge.id}_{inner_edge.id}",
                                        source=edge.source,
                                        target=inner_edge.target,
                                        sourceHandle=edge.sourceHandle,
                                        targetHandle=inner_edge.targetHandle,
                                    )
                                )
                else:
                    remaining_parent_edges.append(edge)
            elif edge.source == composite_id and graph_out_map:
                handle = edge.sourceHandle or "output"
                graph_out_id = graph_out_map.get(handle)
                if graph_out_id:
                    if sub_has_inner_scope:
                        # keep mode: graphOut.value → outer consumer (boundary)
                        new_edges.append(
                            EdgeDef(
                                id=f"flat_{graph_out_id}_to_{edge.id}",
                                source=graph_out_id,
                                target=edge.target,
                                sourceHandle="value",
                                targetHandle=edge.targetHandle,
                            )
                        )
                    else:
                        # strip mode: splice through from the inner producer
                        for inner_edge in sub.edges:
                            if inner_edge.target == graph_out_id:
                                new_edges.append(
                                    EdgeDef(
                                        id=f"flat_{inner_edge.id}_{edge.id}",
                                        source=inner_edge.source,
                                        target=edge.target,
                                        sourceHandle=inner_edge.sourceHandle,
                                        targetHandle=edge.targetHandle,
                                    )
                                )
                else:
                    remaining_parent_edges.append(edge)
            else:
                remaining_parent_edges.append(edge)

        edges = remaining_parent_edges

        # Inner edges touching graphIn/graphOut:
        #   strip mode  → drop them (they were spliced into the parent edge).
        #   keep mode   → preserve them (they connect the kept boundary nodes
        #                 to the inner producer/consumer).
        if (graph_in_map or graph_out_map) and not sub_has_inner_scope:
            port_ids = set(graph_in_map.values()) | set(graph_out_map.values())
            inner_edges_filtered = [
                e for e in sub.edges if e.source not in port_ids and e.target not in port_ids
            ]
        else:
            inner_edges_filtered = list(sub.edges)

        flat_nodes.extend(remaining_inner_nodes)
        new_edges.extend(inner_edges_filtered)

    # Remove composite nodes from parent edges
    if nodes_to_remove:
        edges = [
            e for e in edges if e.source not in nodes_to_remove and e.target not in nodes_to_remove
        ]

    all_edges = edges + new_edges
    all_containers = containers + new_containers
    all_access_grants = access_grants + new_access_grants

    result = GraphDefinition(
        name=graph.name,
        description=graph.description,
        nodes=flat_nodes,
        edges=all_edges,
        containers=all_containers,
        access_grants=all_access_grants,
        step_budget=graph.step_budget,
        presetId=graph.presetId,
    )

    # Recurse: check if any of the flattened nodes also have subgraphs
    has_nested = any(n.subgraph is not None and len(n.subgraph.nodes) > 0 for n in flat_nodes)
    if has_nested:
        result = _flatten_recursive(result, parent_path, fmap)

    return result
