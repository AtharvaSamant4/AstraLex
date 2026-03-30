"""
evidence_graph.py — Build and query a lightweight evidence graph.

Organises retrieved chunks into a graph structure where:

  Nodes  = individual evidence pieces (chunks, concepts, sections)
  Edges  = relationships (defines, references, related_to, explains,
           punished_by, same_act, same_section, overlapping_text)

The graph enables the reasoning stage to see how pieces of evidence
connect — for instance that Section 299 *defines* culpable homicide
while Section 300 *extends* it to murder, and Section 302 *prescribes*
the punishment.

Implementation uses plain dicts/lists (no NetworkX dependency).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from rag.chunker import Chunk

logger = logging.getLogger(__name__)


# ── Data types ─────────────────────────────────────────────────────────────

@dataclass
class EvidenceNode:
    """A single node in the evidence graph."""
    node_id: str
    node_type: str               # "chunk" | "concept" | "section"
    chunk: Chunk | None = None   # backing chunk (if type == "chunk")
    label: str = ""              # human-readable label
    score: float = 0.0           # retrieval / rerank score
    task_id: int | None = None   # which research task produced this

    def __hash__(self) -> int:
        return hash(self.node_id)


@dataclass
class EvidenceEdge:
    """A directed edge between two evidence nodes."""
    source: str     # node_id
    target: str     # node_id
    relation: str   # defines | references | related_to | explains | punished_by
    weight: float = 1.0


@dataclass
class EvidenceGraph:
    """Container for nodes + edges with convenience accessors."""
    nodes: dict[str, EvidenceNode] = field(default_factory=dict)
    edges: list[EvidenceEdge] = field(default_factory=list)

    # ── Mutation ───────────────────────────────────────────────────────────

    def add_node(self, node: EvidenceNode) -> None:
        self.nodes[node.node_id] = node

    def add_edge(self, edge: EvidenceEdge) -> None:
        self.edges.append(edge)

    def has_node(self, node_id: str) -> bool:
        return node_id in self.nodes

    # ── Queries ────────────────────────────────────────────────────────────

    def neighbours(self, node_id: str) -> list[tuple[EvidenceNode, str]]:
        """Return (neighbour_node, relation) for outgoing edges."""
        out: list[tuple[EvidenceNode, str]] = []
        for e in self.edges:
            if e.source == node_id and e.target in self.nodes:
                out.append((self.nodes[e.target], e.relation))
            elif e.target == node_id and e.source in self.nodes:
                out.append((self.nodes[e.source], e.relation))
        return out

    def get_chunks(self) -> list[Chunk]:
        """Return all chunks stored in the graph, deduplicated."""
        seen: set[str] = set()
        chunks: list[Chunk] = []
        for n in self.nodes.values():
            if n.chunk and n.node_id not in seen:
                chunks.append(n.chunk)
                seen.add(n.node_id)
        return chunks

    def top_nodes(self, k: int = 10) -> list[EvidenceNode]:
        """Return top-k nodes by score (descending)."""
        scored = [n for n in self.nodes.values() if n.score > 0]
        scored.sort(key=lambda n: n.score, reverse=True)
        return scored[:k]

    def summary_stats(self) -> dict[str, int]:
        return {
            "nodes": len(self.nodes),
            "edges": len(self.edges),
            "chunks": sum(1 for n in self.nodes.values() if n.chunk is not None),
        }

    # ── Serialisation (for prompt injection) ───────────────────────────────

    def to_context_string(self, top_k: int = 10) -> str:
        """
        Render the top-k evidence nodes into a numbered context block,
        annotated with graph relationships.
        """
        top = self.top_nodes(top_k)
        parts: list[str] = []
        for i, node in enumerate(top, 1):
            if not node.chunk:
                continue
            c = node.chunk
            source = f"{c['act']} — {c['section']}: {c['title']}"
            score_str = f" (relevance {node.score:.3f})"

            # Gather relationship annotations
            rels = self.neighbours(node.node_id)
            rel_strs: list[str] = []
            for neighbour, relation in rels:
                if neighbour.chunk:
                    nb_label = f"{neighbour.chunk['act']} {neighbour.chunk['section']}"
                else:
                    nb_label = neighbour.label
                rel_strs.append(f"{relation} → {nb_label}")

            rel_block = ""
            if rel_strs:
                rel_block = "\n  RELATED: " + " | ".join(rel_strs[:5])

            parts.append(
                f"[{i}] SOURCE: {source}{score_str}{rel_block}\n"
                f"{c['text']}"
            )
        return "\n\n".join(parts)


# ── Builder functions ──────────────────────────────────────────────────────

_SECTION_RE = re.compile(
    r"section\s+(\d+[A-Za-z]?)",
    re.IGNORECASE,
)


def _extract_section_refs(text: str) -> list[str]:
    """Pull out section number references from text."""
    return [m.group(1) for m in _SECTION_RE.finditer(text)]


def _text_overlap(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).quick_ratio()


def build_evidence_graph(
    chunks: list[Chunk],
    scores: list[float] | None = None,
    task_ids: list[int | None] | None = None,
) -> EvidenceGraph:
    """
    Build an evidence graph from a list of chunks.

    Edges are inferred automatically:
      • same_act        — chunks from the same Act
      • same_section    — chunks from the same section
      • references      — chunk text mentions another chunk's section number
      • overlapping     — high text overlap (near-duplicate evidence)
    """
    graph = EvidenceGraph()
    scores = scores or [0.0] * len(chunks)
    task_ids = task_ids or [None] * len(chunks)

    # ── Add nodes ──────────────────────────────────────────────────────────
    for i, chunk in enumerate(chunks):
        nid = chunk.get("chunk_id", f"chunk_{i}")
        node = EvidenceNode(
            node_id=nid,
            node_type="chunk",
            chunk=chunk,
            label=f"{chunk['act']} — {chunk['section']}",
            score=scores[i],
            task_id=task_ids[i],
        )
        graph.add_node(node)

    node_list = list(graph.nodes.values())

    # ── Infer edges ────────────────────────────────────────────────────────
    for i, ni in enumerate(node_list):
        ci = ni.chunk
        if ci is None:
            continue

        for j, nj in enumerate(node_list):
            if j <= i:
                continue
            cj = nj.chunk
            if cj is None:
                continue

            # Same act
            if ci["act"] == cj["act"]:
                if ci["section"] == cj["section"]:
                    graph.add_edge(EvidenceEdge(ni.node_id, nj.node_id, "same_section"))
                else:
                    graph.add_edge(EvidenceEdge(ni.node_id, nj.node_id, "same_act", weight=0.5))

            # Cross-references via section numbers
            refs_i = _extract_section_refs(ci["text"])
            refs_j = _extract_section_refs(cj["text"])

            sec_j = re.sub(r"^Section\s*", "", cj["section"], flags=re.IGNORECASE)
            sec_i = re.sub(r"^Section\s*", "", ci["section"], flags=re.IGNORECASE)

            if sec_j in refs_i:
                graph.add_edge(EvidenceEdge(ni.node_id, nj.node_id, "references"))
            if sec_i in refs_j:
                graph.add_edge(EvidenceEdge(nj.node_id, ni.node_id, "references"))

            # Overlapping text
            if _text_overlap(ci["text"], cj["text"]) >= 0.6:
                graph.add_edge(EvidenceEdge(ni.node_id, nj.node_id, "overlapping", weight=0.3))

    stats = graph.summary_stats()
    logger.info(
        "Evidence graph built: %d nodes, %d edges, %d chunks",
        stats["nodes"], stats["edges"], stats["chunks"],
    )
    return graph
