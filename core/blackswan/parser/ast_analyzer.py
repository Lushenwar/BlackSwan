"""
AST Analyzer for BlackSwan (Phase 2, Step 2A).

ASTAnalyzer parses a Python source file and extracts the structural elements
that BlackSwan's dependency graph and auto-detector tagging require:

  - FunctionDef nodes     → stress-testable units; includes full parameter
                            signatures with default-presence flags and type
                            annotations (resolves the Phase 1 limitation where
                            required parameters without defaults were unknown).

  - Assign / AnnAssign    → DAG nodes; annotated with what is computed in the
                            RHS (division, matrix-multiply @, known calls).

  - Call nodes            → detect np.linalg.inv, np.cov, etc. for
                            auto-detector tagging (Step 2C).

  - BinOp Div / FloorDiv  → tag assignments for DivisionStabilityDetector.

  - Return statements     → identify output variables for BoundsDetector.

  - Import statements     → determine which numpy features are available so
                            the correct detector suites are activated.

Assumption (per CLAUDE.md engineering guidelines):
  This module extracts structural facts about the source. It does NOT execute
  the source, does NOT infer types at runtime, and does NOT resolve dynamic
  values. Pandas chaining, walrus operators, and comprehension intermediates
  are captured as-is; the dependency graph (Step 2B) handles their semantics.
"""

import ast
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Call-name sets used for semantic tagging
# ---------------------------------------------------------------------------

# Calls whose result is a covariance / correlation / outer-product matrix.
_MATRIX_CONSTRUCT_CALLS: frozenset[str] = frozenset({
    "cov", "corrcoef", "outer", "dot",
    "np.cov", "np.corrcoef", "np.outer", "np.dot",
})

# Calls that perform matrix inversion or near-inversion (ConditionNumber target).
_INV_CALLS: frozenset[str] = frozenset({
    "inv", "solve", "lstsq", "pinv",
    "linalg.inv", "linalg.solve", "linalg.lstsq", "linalg.pinv",
    "np.linalg.inv", "np.linalg.solve", "np.linalg.lstsq", "np.linalg.pinv",
})


# ---------------------------------------------------------------------------
# Data classes -- public API
# ---------------------------------------------------------------------------

@dataclass
class ParamInfo:
    """
    One parameter from a function definition.

    has_default is True when the parameter has a default value in the source
    (e.g. x=1.0 or data=None).  The CLI uses this to build base_inputs
    without executing the function.  Phase 1 limitation: parameters without
    defaults cause TypeError at call time; Phase 2 uses annotation + has_default
    to generate safe placeholder values.
    """
    name: str
    has_default: bool
    annotation: str | None = None   # stringified annotation, e.g. "np.ndarray"


@dataclass
class FunctionInfo:
    """A function definition extracted from the source."""
    name: str
    lineno: int
    end_lineno: int
    params: list[ParamInfo] = field(default_factory=list)


@dataclass
class AssignmentInfo:
    """
    One assignment statement (ast.Assign or ast.AnnAssign).

    targets    -- variable names on the LHS (e.g. ["cov_matrix"]).
    calls      -- dotted call names found anywhere in the RHS expression
                 (e.g. ["np.diag", "np.full"]).
    has_division / has_matmult -- True when the corresponding operator appears
                 in the RHS; used by the auto-detector tagger (Step 2C).
    """
    targets: list[str]
    lineno: int
    has_division: bool = False    # Div or FloorDiv in RHS
    has_matmult: bool = False     # @ (MatMult) in RHS
    calls: list[str] = field(default_factory=list)


@dataclass
class ImportInfo:
    """One import statement."""
    lineno: int
    module: str | None            # e.g. "numpy" for import numpy as np
    alias: str | None             # e.g. "np"
    names: list[str]              # for from numpy import linalg, random
    is_from_import: bool


@dataclass
class ReturnInfo:
    """One return statement."""
    lineno: int
    names: list[str]              # bare Name nodes returned, e.g. ["cov_matrix"]


@dataclass
class SourceAnalysis:
    """
    Complete structural extraction of one Python source file.

    Raw collections (functions, assignments, imports, returns) store every
    extracted node.  Convenience properties (division_lines,
    matrix_construct_lines, inv_call_lines, numpy_imported, numpy_alias)
    aggregate them for the dependency graph and detector tagger.
    """
    file_path: Path
    functions: list[FunctionInfo]
    assignments: list[AssignmentInfo]
    imports: list[ImportInfo]
    returns: list[ReturnInfo]

    # ------------------------------------------------------------------
    # Convenience aggregates (derived, not stored)
    # ------------------------------------------------------------------

    @property
    def division_lines(self) -> list[int]:
        """Lines of assignments whose RHS contains a division operator."""
        return sorted({a.lineno for a in self.assignments if a.has_division})

    @property
    def matrix_construct_lines(self) -> list[int]:
        """
        Lines of assignments whose RHS contains a matrix-construction pattern:
        the @ operator or a call to a known covariance/correlation function.
        """
        return sorted({
            a.lineno for a in self.assignments
            if a.has_matmult or bool(set(a.calls) & _MATRIX_CONSTRUCT_CALLS)
        })

    @property
    def inv_call_lines(self) -> list[int]:
        """Lines of assignments whose RHS calls np.linalg.inv or a sibling."""
        return sorted({
            a.lineno for a in self.assignments
            if bool(set(a.calls) & _INV_CALLS)
        })

    @property
    def numpy_imported(self) -> bool:
        """True when the file imports numpy (directly or via alias)."""
        return any(
            imp.module == "numpy" or (imp.module or "").startswith("numpy.")
            for imp in self.imports
        )

    @property
    def numpy_alias(self) -> str | None:
        """
        The alias under which numpy was imported (usually "np"), or the
        module name itself if imported without an alias.  None if numpy is
        not imported.
        """
        for imp in self.imports:
            if imp.module == "numpy":
                return imp.alias if imp.alias else "numpy"
        return None


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class ASTAnalyzer:
    """
    Parses a Python source file and extracts BlackSwan-relevant structure.

    Safe to construct once and call analyze() multiple times (though the
    result is always identical -- the source is read at construction time).
    Never executes the source file.
    """

    def __init__(self, source_path: str | Path) -> None:
        self._path = Path(source_path).resolve()
        self._source = self._path.read_text(encoding="utf-8")   # raises if missing
        # ast.parse raises SyntaxError if the source is malformed.
        self._tree = ast.parse(self._source, filename=str(self._path))

    def analyze(self) -> SourceAnalysis:
        """Return a SourceAnalysis populated from the parsed AST."""
        return SourceAnalysis(
            file_path=self._path,
            functions=self._extract_functions(),
            assignments=self._extract_assignments(),
            imports=self._extract_imports(),
            returns=self._extract_returns(),
        )

    # ------------------------------------------------------------------
    # Extraction helpers
    # ------------------------------------------------------------------

    def _extract_functions(self) -> list[FunctionInfo]:
        functions = []
        for node in ast.walk(self._tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            params = _extract_params(node)
            functions.append(FunctionInfo(
                name=node.name,
                lineno=node.lineno,
                end_lineno=node.end_lineno,
                params=params,
            ))
        # Return in source order.
        functions.sort(key=lambda f: f.lineno)
        return functions

    def _extract_assignments(self) -> list[AssignmentInfo]:
        assignments = []
        for node in ast.walk(self._tree):
            if isinstance(node, ast.Assign):
                targets = _assignment_targets(node.targets)
                info = _rhs_info(node.value)
                assignments.append(AssignmentInfo(
                    targets=targets,
                    lineno=node.lineno,
                    has_division=info.has_division,
                    has_matmult=info.has_matmult,
                    calls=info.calls,
                ))
            elif isinstance(node, ast.AnnAssign) and node.value is not None:
                target = _name_from_node(node.target)
                targets = [target] if target else []
                info = _rhs_info(node.value)
                assignments.append(AssignmentInfo(
                    targets=targets,
                    lineno=node.lineno,
                    has_division=info.has_division,
                    has_matmult=info.has_matmult,
                    calls=info.calls,
                ))
        assignments.sort(key=lambda a: a.lineno)
        return assignments

    def _extract_imports(self) -> list[ImportInfo]:
        imports = []
        for node in ast.walk(self._tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(ImportInfo(
                        lineno=node.lineno,
                        module=alias.name,
                        alias=alias.asname,
                        names=[],
                        is_from_import=False,
                    ))
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                imported_names = [alias.name for alias in node.names]
                # One ImportInfo per from-import statement (not per name).
                imports.append(ImportInfo(
                    lineno=node.lineno,
                    module=module,
                    alias=None,
                    names=imported_names,
                    is_from_import=True,
                ))
        imports.sort(key=lambda i: i.lineno)
        return imports

    def _extract_returns(self) -> list[ReturnInfo]:
        returns = []
        for node in ast.walk(self._tree):
            if not isinstance(node, ast.Return):
                continue
            names: list[str] = []
            if node.value is not None:
                names = _collect_return_names(node.value)
            returns.append(ReturnInfo(lineno=node.lineno, names=names))
        returns.sort(key=lambda r: r.lineno)
        return returns


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_params(func_node: ast.FunctionDef) -> list[ParamInfo]:
    """Extract ParamInfo for every argument in a function definition."""
    args = func_node.args
    all_args = args.posonlyargs + args.args + args.kwonlyargs
    if args.vararg:
        all_args.append(args.vararg)
    if args.kwarg:
        all_args.append(args.kwarg)

    # Defaults align to the END of args.args (and posonlyargs+args.args combined).
    # posonlyargs defaults are in args.defaults; kwonly defaults in kw_defaults.
    positional = args.posonlyargs + args.args
    n_positional = len(positional)
    n_defaults = len(args.defaults)
    # The last n_defaults positional params have defaults.
    has_default_flags = [False] * (n_positional - n_defaults) + [True] * n_defaults

    params = []
    for i, arg in enumerate(positional):
        annotation = _annotation_str(arg.annotation)
        params.append(ParamInfo(
            name=arg.arg,
            has_default=has_default_flags[i],
            annotation=annotation,
        ))

    # kwonlyargs: each has a corresponding entry in kw_defaults (None = no default).
    for arg, default in zip(args.kwonlyargs, args.kw_defaults):
        annotation = _annotation_str(arg.annotation)
        params.append(ParamInfo(
            name=arg.arg,
            has_default=(default is not None),
            annotation=annotation,
        ))

    # *args and **kwargs always count as "having a default" (they're optional).
    if args.vararg:
        params.append(ParamInfo(name=args.vararg.arg, has_default=True,
                                annotation=_annotation_str(args.vararg.annotation)))
    if args.kwarg:
        params.append(ParamInfo(name=args.kwarg.arg, has_default=True,
                                annotation=_annotation_str(args.kwarg.annotation)))

    return params


def _annotation_str(annotation: ast.expr | None) -> str | None:
    """Return a human-readable string for a type annotation node, or None."""
    if annotation is None:
        return None
    try:
        return ast.unparse(annotation)
    except Exception:
        return None


@dataclass
class _RHSInfo:
    has_division: bool = False
    has_matmult: bool = False
    calls: list[str] = field(default_factory=list)


def _rhs_info(node: ast.expr) -> _RHSInfo:
    """Walk an expression subtree and extract division flags and call names."""
    info = _RHSInfo()
    for child in ast.walk(node):
        if isinstance(child, ast.BinOp):
            if isinstance(child.op, (ast.Div, ast.FloorDiv)):
                info.has_division = True
            if isinstance(child.op, ast.MatMult):
                info.has_matmult = True
        if isinstance(child, ast.Call):
            name = _dotted_call_name(child)
            if name:
                info.calls.append(name)
    return info


def _dotted_call_name(call: ast.Call) -> str:
    """
    Extract a dotted string name from a Call node's func attribute.

    Examples:
      np.linalg.inv(m)  gives  "np.linalg.inv"
      np.cov(data)      gives  "np.cov"
      inv(m)            gives  "inv"
    Returns '' when the call target is not a resolvable name chain.
    """
    return _expr_to_dotted(call.func)


def _expr_to_dotted(node: ast.expr) -> str:
    """Recursively build a dotted name from Name / Attribute chains."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _expr_to_dotted(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _assignment_targets(targets: list[ast.expr]) -> list[str]:
    """Collect simple Name targets from an ast.Assign targets list."""
    names = []
    for t in targets:
        if isinstance(t, ast.Name):
            names.append(t.id)
        elif isinstance(t, (ast.Tuple, ast.List)):
            for elt in t.elts:
                name = _name_from_node(elt)
                if name:
                    names.append(name)
    return names


def _name_from_node(node: ast.expr) -> str | None:
    """Return the id if node is ast.Name, otherwise None."""
    return node.id if isinstance(node, ast.Name) else None


def _collect_return_names(node: ast.expr) -> list[str]:
    """
    Extract bare variable names from a return expression.

    - return cov_matrix            gives ["cov_matrix"]
    - return float(np.dot(...))    gives []   (no bare name at top level)
    - return {"weights": w}        gives []   (dict literal, not a name)
    - return a, b                  gives ["a", "b"]  (tuple)
    """
    if isinstance(node, ast.Name):
        return [node.id]
    if isinstance(node, (ast.Tuple, ast.List)):
        return [n for elt in node.elts for n in _collect_return_names(elt)]
    return []
