"""
Tests for DependencyGraphBuilder (Phase 2, Step 2B).
"""
from __future__ import annotations
import json
from pathlib import Path
import pytest
from blackswan.parser.graph import DependencyGraph, DependencyGraphBuilder, GraphEdge, GraphNode

FIXTURE_DIR = Path(__file__).parent / "fixtures"
BROKEN_COV  = FIXTURE_DIR / "broken_covariance.py"
CLEAN_MODEL = FIXTURE_DIR / "clean_model.py"
SIMPLE_PORT = FIXTURE_DIR / "simple_portfolio.py"
OVERFLOW    = FIXTURE_DIR / "overflow_leverage.py"
INT_SHORT   = FIXTURE_DIR / "intentional_short.py"
DIV_EDGE    = FIXTURE_DIR / "division_edge.py"


def _build(path: Path) -> DependencyGraph:
    return DependencyGraphBuilder(path).build()


# ---------------------------------------------------------------------------
# API contract
# ---------------------------------------------------------------------------

class TestAPIContract:
    def test_returns_dependency_graph_instance(self):
        assert isinstance(_build(BROKEN_COV), DependencyGraph)

    def test_accepts_string_path(self):
        assert isinstance(DependencyGraphBuilder(str(BROKEN_COV)).build(), DependencyGraph)

    def test_raises_on_missing_file(self):
        with pytest.raises((FileNotFoundError, OSError)):
            DependencyGraphBuilder("/nonexistent/path.py").build()

    def test_raises_on_syntax_error(self, tmp_path):
        bad = tmp_path / "bad.py"
        bad.write_text("def f(:\n    pass\n")
        with pytest.raises(SyntaxError):
            DependencyGraphBuilder(bad).build()

    def test_graph_has_nodes_list(self):
        g = _build(BROKEN_COV)
        assert isinstance(g.nodes, list)

    def test_graph_has_edges_list(self):
        g = _build(BROKEN_COV)
        assert isinstance(g.edges, list)


# ---------------------------------------------------------------------------
# Node types
# ---------------------------------------------------------------------------

class TestNodeTypes:
    def setup_method(self):
        self.graph = _build(BROKEN_COV)
        self.node_map = {n.id: n for n in self.graph.nodes}

    def test_weights_is_root_input(self):
        assert self.node_map["weights"].node_type == "root_input"

    def test_vol_is_root_input(self):
        assert self.node_map["vol"].node_type == "root_input"

    def test_correlation_is_root_input(self):
        assert self.node_map["correlation"].node_type == "root_input"

    def test_corr_val_is_intermediate(self):
        assert self.node_map["corr_val"].node_type == "intermediate"

    def test_corr_matrix_is_intermediate(self):
        assert self.node_map["corr_matrix"].node_type == "intermediate"

    def test_cov_matrix_is_intermediate(self):
        assert self.node_map["cov_matrix"].node_type == "intermediate"

    def test_n_is_intermediate(self):
        assert self.node_map["n"].node_type == "intermediate"

    def test_all_nodes_have_valid_type(self):
        valid = {"root_input", "intermediate", "unresolved_input", "return_value"}
        for node in self.graph.nodes:
            assert node.node_type in valid, f"{node.id!r} has invalid type {node.node_type!r}"

    def test_node_ids_are_unique(self):
        ids = [n.id for n in self.graph.nodes]
        assert len(ids) == len(set(ids))

    def test_root_input_nodes_have_positive_lineno(self):
        for node in self.graph.nodes:
            if node.node_type == "root_input":
                assert node.lineno is not None and node.lineno >= 1

    def test_intermediate_nodes_have_positive_lineno(self):
        for node in self.graph.nodes:
            if node.node_type == "intermediate":
                assert node.lineno is not None and node.lineno >= 1

    def test_node_label_is_non_empty_string(self):
        for node in self.graph.nodes:
            assert isinstance(node.label, str) and len(node.label) > 0


# ---------------------------------------------------------------------------
# Detector tagging
# ---------------------------------------------------------------------------

class TestDetectorTags:
    def test_cov_matrix_tagged_matrix_psd(self):
        node_map = {n.id: n for n in _build(BROKEN_COV).nodes}
        assert "MatrixPSDDetector" in node_map["cov_matrix"].detector_tags

    def test_cov_matrix_tagged_bounds_detector(self):
        # cov_matrix is the return value of broken_covariance
        node_map = {n.id: n for n in _build(BROKEN_COV).nodes}
        assert "BoundsDetector" in node_map["cov_matrix"].detector_tags

    def test_clean_model_nodes_have_no_detector_tags(self):
        for node in _build(CLEAN_MODEL).nodes:
            assert node.detector_tags == [], f"{node.id!r} unexpectedly has tags"

    def test_inv_cov_tagged_condition_number(self):
        node_map = {n.id: n for n in _build(SIMPLE_PORT).nodes}
        assert "ConditionNumberDetector" in node_map["inv_cov"].detector_tags

    def test_weights_assignment_tagged_division_stability(self):
        # weights = raw_weights / raw_weights.sum() in simple_portfolio
        node_map = {n.id: n for n in _build(SIMPLE_PORT).nodes}
        assert "DivisionStabilityDetector" in node_map["weights"].detector_tags

    def test_raw_weights_tagged_matrix_psd(self):
        # raw_weights = inv_cov @ ones in simple_portfolio
        node_map = {n.id: n for n in _build(SIMPLE_PORT).nodes}
        assert "MatrixPSDDetector" in node_map["raw_weights"].detector_tags

    def test_normalised_tagged_division_stability(self):
        # normalised = returns / vol in division_edge
        node_map = {n.id: n for n in _build(DIV_EDGE).nodes}
        assert "DivisionStabilityDetector" in node_map["normalised"].detector_tags

    def test_overflow_nodes_have_no_fragility_detector_tags(self):
        # overflow_leverage uses multiply (not divide/matmult/inv), so no fragility tags.
        # BoundsDetector on the return value is expected and acceptable.
        fragility = {"DivisionStabilityDetector", "MatrixPSDDetector", "ConditionNumberDetector"}
        for node in _build(OVERFLOW).nodes:
            overlap = set(node.detector_tags) & fragility
            assert not overlap, f"{node.id!r} unexpectedly has fragility tags {overlap}"

    def test_detector_tags_are_lists(self):
        for node in _build(BROKEN_COV).nodes:
            assert isinstance(node.detector_tags, list)

    def test_root_input_nodes_have_no_detector_tags(self):
        for node in _build(BROKEN_COV).nodes:
            if node.node_type == "root_input":
                assert node.detector_tags == []


# ---------------------------------------------------------------------------
# Edges
# ---------------------------------------------------------------------------

class TestEdges:
    def setup_method(self):
        self.graph = _build(BROKEN_COV)
        self.pairs = {(e.source, e.target) for e in self.graph.edges}

    def test_correlation_flows_to_corr_val(self):
        assert ("correlation", "corr_val") in self.pairs

    def test_vol_flows_to_n(self):
        # n = len(vol)
        assert ("vol", "n") in self.pairs

    def test_n_flows_to_corr_matrix(self):
        # corr_matrix = np.full((n, n), corr_val)
        assert ("n", "corr_matrix") in self.pairs

    def test_corr_val_flows_to_corr_matrix(self):
        assert ("corr_val", "corr_matrix") in self.pairs

    def test_corr_matrix_flows_to_cov_matrix(self):
        assert ("corr_matrix", "cov_matrix") in self.pairs

    def test_vol_flows_to_cov_matrix(self):
        # vol used inside np.diag(vol)
        assert ("vol", "cov_matrix") in self.pairs

    def test_no_self_loops(self):
        for e in self.graph.edges:
            assert e.source != e.target

    def test_edges_reference_existing_node_ids(self):
        node_ids = {n.id for n in self.graph.nodes}
        for e in self.graph.edges:
            assert e.source in node_ids, f"unknown source {e.source!r}"
            assert e.target in node_ids, f"unknown target {e.target!r}"

    def test_no_duplicate_edges(self):
        pairs = [(e.source, e.target) for e in self.graph.edges]
        assert len(pairs) == len(set(pairs))

    def test_simple_portfolio_inv_cov_depends_on_cov_param(self):
        pairs = {(e.source, e.target) for e in _build(SIMPLE_PORT).edges}
        assert ("cov", "inv_cov") in pairs

    def test_simple_portfolio_raw_weights_depends_on_inv_cov(self):
        pairs = {(e.source, e.target) for e in _build(SIMPLE_PORT).edges}
        assert ("inv_cov", "raw_weights") in pairs


# ---------------------------------------------------------------------------
# Unresolved inputs
# ---------------------------------------------------------------------------

class TestUnresolvedInputs:
    def test_unresolved_variable_creates_unresolved_node(self, tmp_path):
        src = tmp_path / "ext.py"
        src.write_text(
            "import numpy as np\n"
            "def f(w=None):\n"
            "    if w is None:\n"
            "        w = np.array([0.5, 0.5])\n"
            "    value = w @ EXTERNAL_COV\n"
            "    return value\n"
        )
        graph = DependencyGraphBuilder(src).build()
        unresolved = [n for n in graph.nodes if n.node_type == "unresolved_input"]
        assert any("EXTERNAL_COV" in n.label for n in unresolved)

    def test_unresolved_node_type_is_correct(self, tmp_path):
        src = tmp_path / "ext2.py"
        src.write_text(
            "def g(x=1.0):\n"
            "    result = x * GLOBAL_SCALE\n"
            "    return result\n"
        )
        graph = DependencyGraphBuilder(src).build()
        unresolved = [n for n in graph.nodes if n.node_type == "unresolved_input"]
        assert len(unresolved) >= 1

    def test_unresolved_node_lineno_is_none(self, tmp_path):
        src = tmp_path / "ext3.py"
        src.write_text(
            "def h(x=1.0):\n"
            "    y = x + MYSTERY\n"
            "    return y\n"
        )
        graph = DependencyGraphBuilder(src).build()
        for node in graph.nodes:
            if node.node_type == "unresolved_input":
                assert node.lineno is None

    def test_builtins_not_treated_as_unresolved(self):
        graph = _build(BROKEN_COV)
        labels = {n.label for n in graph.nodes if n.node_type == "unresolved_input"}
        for builtin in ("len", "float", "int", "range", "True", "False", "None"):
            assert builtin not in labels

    def test_import_aliases_not_treated_as_unresolved(self):
        graph = _build(BROKEN_COV)
        labels = {n.label for n in graph.nodes if n.node_type == "unresolved_input"}
        assert "np" not in labels

    def test_unresolved_node_has_edge_to_dependent(self, tmp_path):
        src = tmp_path / "ext4.py"
        src.write_text(
            "def h(x=1.0):\n"
            "    y = x + MYSTERY\n"
            "    return y\n"
        )
        graph = DependencyGraphBuilder(src).build()
        unresolved_ids = {n.id for n in graph.nodes if n.node_type == "unresolved_input"}
        edge_sources = {e.source for e in graph.edges}
        assert unresolved_ids & edge_sources  # at least one unresolved node has an outgoing edge


# ---------------------------------------------------------------------------
# Pandas method-chain handling
# ---------------------------------------------------------------------------

class TestPandasChains:
    def test_method_chain_creates_single_intermediate_node(self, tmp_path):
        src = tmp_path / "chain.py"
        src.write_text(
            "import pandas as pd\n"
            "def f(data=None):\n"
            "    if data is None:\n"
            "        data = pd.DataFrame({'a': [1, 2, 3]})\n"
            "    result = data.groupby('a').sum().reset_index()\n"
            "    return result\n"
        )
        graph = DependencyGraphBuilder(src).build()
        result_nodes = [n for n in graph.nodes if n.id == "result"]
        assert len(result_nodes) == 1

    def test_method_chain_node_is_intermediate(self, tmp_path):
        src = tmp_path / "chain2.py"
        src.write_text(
            "import pandas as pd\n"
            "def f(df=None):\n"
            "    if df is None:\n"
            "        df = pd.DataFrame({'x': [1, 2, 3]})\n"
            "    out = df.sort_values('x').head(2).reset_index(drop=True)\n"
            "    return out\n"
        )
        graph = DependencyGraphBuilder(src).build()
        node_map = {n.id: n for n in graph.nodes}
        assert node_map["out"].node_type == "intermediate"

    def test_method_chain_does_not_produce_extra_nodes_for_method_names(self, tmp_path):
        src = tmp_path / "chain3.py"
        src.write_text(
            "import pandas as pd\n"
            "def f(df=None):\n"
            "    if df is None:\n"
            "        df = pd.DataFrame({'x': [1, 2, 3]})\n"
            "    out = df.dropna().fillna(0).reset_index()\n"
            "    return out\n"
        )
        graph = DependencyGraphBuilder(src).build()
        # 'dropna', 'fillna', 'reset_index' are method names, not variables
        node_ids = {n.id for n in graph.nodes}
        assert "dropna" not in node_ids
        assert "fillna" not in node_ids
        assert "reset_index" not in node_ids


# ---------------------------------------------------------------------------
# JSON serialisation
# ---------------------------------------------------------------------------

class TestJSONSerialization:
    def setup_method(self):
        self.graph = _build(BROKEN_COV)
        self.d = self.graph.to_dict()

    def test_to_dict_has_nodes_key(self):
        assert "nodes" in self.d

    def test_to_dict_has_edges_key(self):
        assert "edges" in self.d

    def test_nodes_list_is_non_empty(self):
        assert len(self.d["nodes"]) > 0

    def test_each_node_dict_has_id(self):
        for n in self.d["nodes"]:
            assert "id" in n

    def test_each_node_dict_has_label(self):
        for n in self.d["nodes"]:
            assert "label" in n

    def test_each_node_dict_has_node_type(self):
        for n in self.d["nodes"]:
            assert "node_type" in n

    def test_each_node_dict_has_lineno(self):
        for n in self.d["nodes"]:
            assert "lineno" in n

    def test_each_node_dict_has_detector_tags(self):
        for n in self.d["nodes"]:
            assert "detector_tags" in n

    def test_each_edge_dict_has_source(self):
        for e in self.d["edges"]:
            assert "source" in e

    def test_each_edge_dict_has_target(self):
        for e in self.d["edges"]:
            assert "target" in e

    def test_to_json_is_valid_json(self):
        parsed = json.loads(self.graph.to_json())
        assert "nodes" in parsed

    def test_to_json_node_count_matches(self):
        parsed = json.loads(self.graph.to_json())
        assert len(parsed["nodes"]) == len(self.graph.nodes)

    def test_to_json_edge_count_matches(self):
        parsed = json.loads(self.graph.to_json())
        assert len(parsed["edges"]) == len(self.graph.edges)


# ---------------------------------------------------------------------------
# All fixtures build cleanly
# ---------------------------------------------------------------------------

class TestAllFixtures:
    @pytest.mark.parametrize("fixture_path", [
        BROKEN_COV, CLEAN_MODEL, SIMPLE_PORT, OVERFLOW, INT_SHORT, DIV_EDGE,
    ])
    def test_fixture_builds_without_error(self, fixture_path):
        assert isinstance(_build(fixture_path), DependencyGraph)

    def test_simple_portfolio_has_condition_number_tagged_node(self):
        tagged = [n for n in _build(SIMPLE_PORT).nodes
                  if "ConditionNumberDetector" in n.detector_tags]
        assert len(tagged) >= 1

    def test_simple_portfolio_has_matrix_psd_tagged_node(self):
        tagged = [n for n in _build(SIMPLE_PORT).nodes
                  if "MatrixPSDDetector" in n.detector_tags]
        assert len(tagged) >= 1

    def test_simple_portfolio_has_division_tagged_node(self):
        tagged = [n for n in _build(SIMPLE_PORT).nodes
                  if "DivisionStabilityDetector" in n.detector_tags]
        assert len(tagged) >= 1

    def test_overflow_has_no_fragility_detector_tagged_nodes(self):
        fragility = {"DivisionStabilityDetector", "MatrixPSDDetector", "ConditionNumberDetector"}
        for node in _build(OVERFLOW).nodes:
            assert not (set(node.detector_tags) & fragility), f"{node.id!r} unexpectedly tagged"
