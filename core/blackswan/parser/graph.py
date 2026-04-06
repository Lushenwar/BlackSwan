"""
Dependency Graph for BlackSwan (Phase 2, Step 2B).

DependencyGraphBuilder ingests a Python source file, combines the structural
data produced by ASTAnalyzer with a targeted second AST walk, and constructs
a directed acyclic graph (DAG) where:

  - Nodes  = variables and function parameters
  - Edges  = data-flow dependencies (A → B means B's computation uses A)

Each node carries detector_tags derived from its RHS expression:
  MatrixPSDDetector        — @ operator or covariance-construction call
  ConditionNumberDetector  — linalg.inv / solve / pinv / lstsq call
  DivisionStabilityDetector — Div or FloorDiv operator
  BoundsDetector           — variable appears in a return statement

AST limitations handled explicitly (per CLAUDE.md):
  Pandas chaining  — treated as a single intermediate node; the chain root
                     object is the sole data-flow dependency.  No synthetic
                     sub-nodes are created for intermediate method names.
  Unresolved inputs — a variable referenced in RHS that is neither a known
                      local variable nor an import / builtin is recorded as an
                      "unresolved_input" node so the DAG stays connected.
  NumPy broadcasting — shape inference is deferred to the runtime tracer;
                       the graph only records structural data flow.
"""

from __future__ import annotations

import ast
import builtins
import json
from dataclasses import dataclass, field
from pathlib import Path

from .ast_analyzer import (
    ASTAnalyzer,
    _MATRIX_CONSTRUCT_CALLS,  # noqa: PLC2701 — shared internal constant
    _INV_CALLS,                # noqa: PLC2701
    _RHSInfo,
    _rhs_info,
    _assignment_targets,
    _name_from_node,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BUILTIN_NAMES: frozenset[str] = frozenset(dir(builtins))

_DIVISION_DETECTOR         = "DivisionStabilityDetector"
_MATRIX_PSD_DETECTOR       = "MatrixPSDDetector"
_CONDITION_NUMBER_DETECTOR = "ConditionNumberDetector"
_BOUNDS_DETECTOR           = "BoundsDetector"

NODE_ROOT_INPUT      = "root_input"
NODE_INTERMEDIATE    = "intermediate"
NODE_UNRESOLVED      = "unresolved_input"
NODE_RETURN_VALUE    = "return_value"   # reserved for future use


# ---------------------------------------------------------------------------
# Public data classes
# ---------------------------------------------------------------------------

@dataclass
class GraphNode:
    """One node in the dependency DAG."""
    id: str
    label: str
    lineno: int | None
    node_type: str           # one of the NODE_* constants above
    detector_tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "lineno": self.lineno,
            "node_type": self.node_type,
            "detector_tags": list(self.detector_tags),
        }


@dataclass
class GraphEdge:
    """Directed edge: data flows from *source* to *target*."""
    source: str
    target: str

    def to_dict(self) -> dict:
        return {"source": self.source, "target": self.target}


@dataclass
class DependencyGraph:
    """
    Complete dependency graph for one Python source file.

    Produced by DependencyGraphBuilder.build().  Serialise with to_dict() /
    to_json() for the Phase 3 webview panel.
    """
    nodes: list[GraphNode]
    edges: list[GraphEdge]

    def to_dict(self) -> dict:
        return {
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
        }

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(), indent=indent)


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

class DependencyGraphBuilder:
    """
    Build a DependencyGraph from a Python source file.

    Safe to construct once and call build() multiple times (result is
    identical).  Never executes the source file.
    """

    def __init__(self, source_path: str | Path) -> None:
        self._path = Path(source_path).resolve()
        # ASTAnalyzer raises FileNotFoundError / SyntaxError as needed.
        self._analyzer = ASTAnalyzer(self._path)
        self._analysis = self._analyzer.analyze()
        # We need the raw AST for a second walk (RHS variable names).
        import ast as _ast
        self._tree = _ast.parse(
            self._path.read_text(encoding="utf-8"),
            filename=str(self._path),
        )

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def build(self) -> DependencyGraph:
        """Parse source and return the populated DependencyGraph."""
        nodes: dict[str, GraphNode] = {}
        edges_seen: set[tuple[str, str]] = set()
        edges: list[GraphEdge] = []

        # Names that should never become unresolved_input nodes.
        exclude: frozenset[str] = _BUILTIN_NAMES | self._import_scope()

        # --- Pass 1: create all nodes in source order -------------------
        param_names = self._create_param_nodes(nodes)
        self._create_assignment_nodes(nodes, param_names)

        # --- Pass 2: create edges (requires all nodes to exist first) ---
        self._create_edges(nodes, edges, edges_seen, param_names, exclude)

        # --- Pass 3: tag returned variables with BoundsDetector ---------
        self._tag_return_nodes(nodes)

        return DependencyGraph(
            nodes=sorted(nodes.values(), key=_node_sort_key),
            edges=edges,
        )

    # ------------------------------------------------------------------
    # Pass 1 helpers
    # ------------------------------------------------------------------

    def _create_param_nodes(self, nodes: dict[str, GraphNode]) -> set[str]:
        """Add root_input nodes for every function parameter."""
        param_names: set[str] = set()
        for fn in self._analysis.functions:
            for param in fn.params:
                param_names.add(param.name)
                if param.name not in nodes:
                    nodes[param.name] = GraphNode(
                        id=param.name,
                        label=param.name,
                        lineno=fn.lineno,
                        node_type=NODE_ROOT_INPUT,
                        detector_tags=[],
                    )
        return param_names

    def _create_assignment_nodes(
        self, nodes: dict[str, GraphNode], param_names: set[str]
    ) -> None:
        """
        Add intermediate nodes for assignment targets.

        Assignments where the target is a function parameter are skipped
        (they represent default-value guards like ``if x is None: x = ...``
        and the parameter is already a root_input node).
        """
        raw = _sorted_assign_nodes(self._tree)
        for ast_node in raw:
            if isinstance(ast_node, ast.Assign):
                targets = _assignment_targets(ast_node.targets)
                rhs = ast_node.value
            else:  # ast.AnnAssign
                t = _name_from_node(ast_node.target)
                targets = [t] if t else []
                rhs = ast_node.value  # already checked non-None upstream

            tags = _detector_tags(_rhs_info(rhs))

            for target in targets:
                if target in param_names:
                    continue  # skip re-assignments of params
                if target in nodes:
                    # Re-assignment (e.g. inside a loop): merge tags only.
                    for tag in tags:
                        if tag not in nodes[target].detector_tags:
                            nodes[target].detector_tags.append(tag)
                else:
                    nodes[target] = GraphNode(
                        id=target,
                        label=target,
                        lineno=ast_node.lineno,
                        node_type=NODE_INTERMEDIATE,
                        detector_tags=list(tags),
                    )

    # ------------------------------------------------------------------
    # Pass 2 helpers
    # ------------------------------------------------------------------

    def _create_edges(
        self,
        nodes: dict[str, GraphNode],
        edges: list[GraphEdge],
        edges_seen: set[tuple[str, str]],
        param_names: set[str],
        exclude: frozenset[str],
    ) -> None:
        """Walk the raw AST once more to collect RHS variable dependencies."""
        for ast_node in _sorted_assign_nodes(self._tree):
            if isinstance(ast_node, ast.Assign):
                targets = _assignment_targets(ast_node.targets)
                rhs = ast_node.value
            else:
                t = _name_from_node(ast_node.target)
                targets = [t] if t else []
                rhs = ast_node.value

            rhs_vars = _collect_rhs_names(rhs, exclude)

            for target in targets:
                if target in param_names:
                    continue
                if target not in nodes:
                    continue  # should not happen after pass 1

                for var in rhs_vars:
                    if var == target:
                        continue  # no self-loops

                    pair = (var, target)
                    if pair in edges_seen:
                        continue

                    if var in nodes:
                        edges.append(GraphEdge(source=var, target=target))
                        edges_seen.add(pair)
                    else:
                        # Create an unresolved_input node for this variable.
                        uid = f"unresolved:{var}"
                        if uid not in nodes:
                            nodes[uid] = GraphNode(
                                id=uid,
                                label=var,
                                lineno=None,
                                node_type=NODE_UNRESOLVED,
                                detector_tags=[],
                            )
                        upair = (uid, target)
                        if upair not in edges_seen:
                            edges.append(GraphEdge(source=uid, target=target))
                            edges_seen.add(upair)

    # ------------------------------------------------------------------
    # Pass 3 helper
    # ------------------------------------------------------------------

    def _tag_return_nodes(self, nodes: dict[str, GraphNode]) -> None:
        """Add BoundsDetector to nodes that are directly returned."""
        for ret in self._analysis.returns:
            for name in ret.names:
                if name in nodes and _BOUNDS_DETECTOR not in nodes[name].detector_tags:
                    nodes[name].detector_tags.append(_BOUNDS_DETECTOR)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def _import_scope(self) -> frozenset[str]:
        """Return all names introduced by import statements (aliases, etc.)."""
        names: set[str] = set()
        for imp in self._analysis.imports:
            if imp.alias:
                names.add(imp.alias)
            elif imp.module:
                # "import numpy" without alias: the top-level name is "numpy"
                names.add(imp.module.split(".")[0])
            for name in imp.names:
                names.add(name)
        return frozenset(names)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sorted_assign_nodes(tree: ast.AST) -> list[ast.stmt]:
    """Collect Assign and AnnAssign nodes from the tree, sorted by lineno."""
    result = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            result.append(node)
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            result.append(node)
    result.sort(key=lambda n: n.lineno)
    return result


def _collect_rhs_names(expr: ast.expr, exclude: frozenset[str]) -> list[str]:
    """
    Collect all ast.Name references in *expr*, excluding *exclude*.

    Deduplicates while preserving left-to-right order of first occurrence.
    Method names on Attribute nodes (e.g. the "groupby" in ``df.groupby()``)
    are NOT collected — only genuine variable Name nodes are extracted.  This
    means Pandas method chains naturally produce only the root object as a
    dependency, satisfying the "treat chain as single node" requirement.
    """
    seen: dict[str, None] = {}
    for node in ast.walk(expr):
        if isinstance(node, ast.Name) and node.id not in exclude:
            seen.setdefault(node.id, None)
    return list(seen)


def _detector_tags(info: _RHSInfo) -> list[str]:
    """Map an _RHSInfo to a list of detector tag strings."""
    tags: list[str] = []
    if info.has_division:
        tags.append(_DIVISION_DETECTOR)
    if info.has_matmult or bool(set(info.calls) & _MATRIX_CONSTRUCT_CALLS):
        tags.append(_MATRIX_PSD_DETECTOR)
    if bool(set(info.calls) & _INV_CALLS):
        tags.append(_CONDITION_NUMBER_DETECTOR)
    return tags


def _node_sort_key(node: GraphNode) -> tuple:
    """Sort nodes: root_inputs first (by lineno), then intermediates, then unresolved."""
    order = {NODE_ROOT_INPUT: 0, NODE_INTERMEDIATE: 1, NODE_UNRESOLVED: 2}
    return (order.get(node.node_type, 3), node.lineno or 0, node.id)
