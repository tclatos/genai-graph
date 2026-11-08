"""Graph algorithms for visualization pruning and layout simplification."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from genai_graph.kg.ingest.extract import NodeRecord, RelationshipRecord


def prune_graph_by_max_siblings(
    nodes: list[NodeRecord],
    relationships: list[RelationshipRecord],
    max_siblings: int = 10,
    target_node_types: set[str] | None = None,
    focus_node_id: str | None = None,
) -> tuple[list[NodeRecord], list[RelationshipRecord], int]:
    """Prune graph edges so that each parent node has at most `max_siblings` children.

    Prevents visual clutter (e.g. hundreds of MarkdownSection nodes attached to a single
    document or section) while preserving hierarchical structure and document reading order.

    Args:
        nodes: List of NodeRecord instances
        relationships: List of RelationshipRecord instances
        max_siblings: Maximum number of sibling children per parent and relationship type.
                      If <= 0 or None, no pruning is performed.
        target_node_types: If set, only prune children belonging to these types (e.g. {'MarkdownSection'}).
                           If None, prune across all child types.
        focus_node_id: Optional root/focus node ID to start traversal from.

    Returns:
        Tuple of (pruned_nodes, pruned_relationships, total_pruned_edges_count)
    """
    if not max_siblings or max_siblings <= 0 or not relationships:
        return nodes, relationships, 0

    node_by_id = {n.node_id: n for n in nodes}

    def get_sort_key(node_id: str) -> tuple[int, str]:
        n = node_by_id.get(node_id)
        if not n:
            return (999999, "")
        p = n.properties
        # Document sequence / line_start order
        seq = p.get("sequence")
        if seq is not None:
            try:
                return (int(seq), p.get("title") or p.get("name") or "")
            except (ValueError, TypeError):
                pass
        ls = p.get("line_start")
        if ls is not None:
            try:
                return (int(ls), p.get("title") or p.get("name") or "")
            except (ValueError, TypeError):
                pass
        return (0, p.get("title") or p.get("name") or "")

    # Map outgoing edges by parent and relationship type
    adj: dict[str, dict[str, list[RelationshipRecord]]] = defaultdict(lambda: defaultdict(list))
    in_degree: dict[str, int] = defaultdict(int)
    all_node_ids = {n.node_id for n in nodes}

    for r in relationships:
        adj[r.from_id][r.name].append(r)
        in_degree[r.to_id] += 1
        if r.from_id not in in_degree:
            in_degree[r.from_id] = 0

    visited_node_ids: set[str] = set()
    kept_relationships: list[RelationshipRecord] = []
    pruned_count = 0

    matched_root_id = None
    if focus_node_id:
        if focus_node_id in all_node_ids:
            matched_root_id = focus_node_id
        else:
            for n in nodes:
                p = n.properties
                if (
                    p.get("name") == focus_node_id
                    or p.get("opportunity_id") == focus_node_id
                    or p.get("folder_id") == focus_node_id
                    or p.get("id") == focus_node_id
                    or p.get("title") == focus_node_id
                ):
                    matched_root_id = n.node_id
                    break

    if matched_root_id:
        initial_roots = [matched_root_id]
    else:
        initial_roots = [nid for nid in all_node_ids if in_degree[nid] == 0]
        if not initial_roots and all_node_ids:
            initial_roots = [next(iter(all_node_ids))]

    def traverse(start_roots: list[str]) -> None:
        nonlocal pruned_count
        q = deque(start_roots)
        for root_id in start_roots:
            visited_node_ids.add(root_id)

        while q:
            curr = q.popleft()
            for _rel_name, edge_list in adj[curr].items():
                sample_child_type = edge_list[0].to_type if edge_list else None
                should_limit = True
                if target_node_types is not None and sample_child_type not in target_node_types:
                    should_limit = False

                if should_limit and len(edge_list) > max_siblings:
                    # Sort edges by document sequence/line_start to keep top sequential items
                    sorted_edges = sorted(edge_list, key=lambda e: get_sort_key(e.to_id))
                    selected_edges = sorted_edges[:max_siblings]
                    pruned_count += len(edge_list) - max_siblings
                else:
                    selected_edges = edge_list

                for edge in selected_edges:
                    kept_relationships.append(edge)
                    if edge.to_id not in visited_node_ids:
                        visited_node_ids.add(edge.to_id)
                        q.append(edge.to_id)

    traverse(initial_roots)

    # Process remaining unvisited components (e.g. other root nodes)
    for nid in all_node_ids:
        if in_degree[nid] == 0 and nid not in visited_node_ids:
            traverse([nid])

    kept_nodes = [n for n in nodes if n.node_id in visited_node_ids]
    return kept_nodes, kept_relationships, pruned_count
