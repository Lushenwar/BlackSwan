"""
mcp_server.py — BlackSwan MCP (Model Context Protocol) server.

Exposes four tools that let Claude (or any MCP-compatible AI agent) interact
with the BlackSwan stress-testing engine directly from the IDE or chat:

  run_blackswan       Run a stress test on a Python file and return findings.
  list_scenarios      List all available preset scenarios.
  get_finding_detail  Return the full detail of a specific shatter point.
  explain_finding     Generate a plain-language explanation of a failure.

Transport: stdio (compatible with Claude Code and claude_desktop_config.json).

Setup — add to ~/Library/Application Support/Claude/claude_desktop_config.json
(macOS) or %APPDATA%\\Claude\\claude_desktop_config.json (Windows):

    {
      "mcpServers": {
        "blackswan": {
          "command": "blackswan-mcp"
        }
      }
    }

Or, if running from the repo without installing:

    {
      "mcpServers": {
        "blackswan": {
          "command": "python",
          "args": ["-m", "blackswan.mcp_server"]
        }
      }
    }
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import sys
from pathlib import Path
from typing import Any

try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp import types as mcp_types
except ImportError:
    _MCP_AVAILABLE = False
else:
    _MCP_AVAILABLE = True


# ---------------------------------------------------------------------------
# Tool: list_scenarios
# ---------------------------------------------------------------------------

def _tool_list_scenarios() -> list[dict[str, Any]]:
    """Return metadata for every available preset scenario."""
    from .scenarios.registry import list_scenarios, load_scenario

    results = []
    for name in list_scenarios():
        try:
            sc = load_scenario(name)
            results.append({
                "name": sc.name,
                "display_name": sc.display_name,
                "description": sc.description,
                "default_iterations": sc.default_iterations,
                "default_seed": sc.default_seed,
                "perturbation_targets": [p.target for p in sc.perturbations],
            })
        except Exception as exc:
            results.append({"name": name, "error": str(exc)})
    return results


# ---------------------------------------------------------------------------
# Tool: run_blackswan
# ---------------------------------------------------------------------------

def _tool_run_blackswan(
    file_path: str,
    scenario: str,
    function_name: str | None = None,
    iterations: int | None = None,
    seed: int | None = None,
    mode: str = "full",
) -> dict[str, Any]:
    """
    Run a BlackSwan stress test and return the full response dict.

    This is the same result the CLI produces — structured as the JSON contract.
    """
    from .scenarios.registry import load_scenario
    from .engine.runner import StressRunner
    from .parser.auto_tagger import AutoTagger
    from .engine.repro import build_reproducibility_card
    from .cli import _load_function, _build_response, _IterationOverride

    fp = Path(file_path).resolve()
    if not fp.exists():
        return {"status": "error", "message": f"File not found: {file_path}"}

    try:
        sc = load_scenario(scenario)
    except FileNotFoundError as exc:
        return {"status": "error", "message": str(exc)}

    try:
        fn, base_inputs = _load_function(fp, function_name)
    except Exception as exc:
        return {"status": "error", "message": str(exc)}

    effective_iterations = iterations if iterations is not None else sc.default_iterations
    effective_seed = seed if seed is not None else sc.default_seed

    detectors = AutoTagger(fp).detector_suite()
    runner = StressRunner(
        fn=fn,
        base_inputs=base_inputs,
        scenario=_IterationOverride(sc, effective_iterations),
        detectors=detectors,
        seed=effective_seed,
        mode=mode,
    )
    result = runner.run()

    return _build_response(
        result=result,
        scenario=sc,
        seed=effective_seed,
        file_path=fp,
        mode=mode,
        iterations_requested=effective_iterations,
    )


# ---------------------------------------------------------------------------
# Tool: get_finding_detail
# ---------------------------------------------------------------------------

def _tool_get_finding_detail(
    response: dict[str, Any],
    finding_id: str,
) -> dict[str, Any]:
    """
    Extract a single shatter point from a previously obtained response dict.

    finding_id is the 'id' field of a shatter_point (e.g. 'sp_001').
    """
    shatter_points = response.get("shatter_points", [])
    for sp in shatter_points:
        if sp.get("id") == finding_id:
            return sp
    return {
        "error": f"No finding with id {finding_id!r}.",
        "available_ids": [sp.get("id") for sp in shatter_points],
    }


# ---------------------------------------------------------------------------
# Tool: explain_finding
# ---------------------------------------------------------------------------

def _tool_explain_finding(
    failure_type: str,
    message: str,
    frequency: str,
    fix_hint: str,
    causal_chain: list[dict],
) -> str:
    """
    Return a plain-language explanation prompt for a shatter point.

    This tool returns a formatted explanation context that Claude can use to
    reason about the failure.  It does not call an external API.
    """
    from .sarif import _RULE_META, _FALLBACK_RULE

    meta = _RULE_META.get(failure_type, _FALLBACK_RULE)
    chain_text = "\n".join(
        f"  • Line {link.get('line', '?')}: `{link.get('variable', '?')}` ({link.get('role', '?')})"
        for link in causal_chain
    )

    return (
        f"## BlackSwan Failure Explanation\n\n"
        f"**Type:** {meta['name']} (`{failure_type}`)\n\n"
        f"**What this failure means:**\n{meta['full']}\n\n"
        f"**Engine message:** {message}\n\n"
        f"**Observed frequency:** {frequency}\n\n"
        f"**Causal chain (root → failure site):**\n{chain_text or '  (not available)'}\n\n"
        f"**Suggested fix:** {fix_hint or 'No fix hint available.'}\n\n"
        f"---\n"
        f"*This information comes from the BlackSwan stress-testing engine. "
        f"All code fixes should be applied via `python -m blackswan fix`.*"
    )


# ---------------------------------------------------------------------------
# MCP server wiring
# ---------------------------------------------------------------------------

def _build_server() -> "Server":
    server = Server("blackswan")

    @server.list_tools()
    async def handle_list_tools() -> list[mcp_types.Tool]:
        return [
            mcp_types.Tool(
                name="run_blackswan",
                description=(
                    "Run a BlackSwan stress test on a Python file. "
                    "Returns a structured report of all shatter points found: "
                    "the exact lines where mathematical logic breaks under stress, "
                    "their frequency, causal chains, and fix hints."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Absolute path to the Python file to stress-test.",
                        },
                        "scenario": {
                            "type": "string",
                            "description": (
                                "Preset scenario name. Use list_scenarios to see options. "
                                "Common values: liquidity_crash, vol_spike, "
                                "correlation_breakdown, rate_shock, missing_data."
                            ),
                        },
                        "function_name": {
                            "type": "string",
                            "description": (
                                "Name of the function to test. "
                                "If omitted, the first public function is used."
                            ),
                        },
                        "iterations": {
                            "type": "integer",
                            "description": "Number of Monte Carlo iterations (default: scenario default).",
                        },
                        "seed": {
                            "type": "integer",
                            "description": "RNG seed for reproducibility.",
                        },
                        "mode": {
                            "type": "string",
                            "enum": ["full", "fast"],
                            "description": (
                                "'full' (default): sweep + attribution tracing. "
                                "'fast': sweep only, no causal chain."
                            ),
                        },
                    },
                    "required": ["file_path", "scenario"],
                },
            ),
            mcp_types.Tool(
                name="list_scenarios",
                description=(
                    "List all BlackSwan preset stress scenarios with their descriptions "
                    "and perturbation targets. Use this to discover what scenarios are "
                    "available before calling run_blackswan."
                ),
                inputSchema={"type": "object", "properties": {}, "required": []},
            ),
            mcp_types.Tool(
                name="get_finding_detail",
                description=(
                    "Extract the full detail of a specific shatter point from a "
                    "run_blackswan response. Use the 'id' field (e.g. 'sp_001') "
                    "from the shatter_points array."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "response": {
                            "type": "object",
                            "description": "The full response dict returned by run_blackswan.",
                        },
                        "finding_id": {
                            "type": "string",
                            "description": "The shatter point id to retrieve (e.g. 'sp_001').",
                        },
                    },
                    "required": ["response", "finding_id"],
                },
            ),
            mcp_types.Tool(
                name="explain_finding",
                description=(
                    "Generate a plain-language explanation of a BlackSwan failure. "
                    "Pass the fields from a shatter_point object. "
                    "Returns a formatted Markdown explanation of the mathematical "
                    "invariant that was violated and what it means in production."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "failure_type": {
                            "type": "string",
                            "description": "The failure_type field from a shatter_point.",
                        },
                        "message": {
                            "type": "string",
                            "description": "The message field from a shatter_point.",
                        },
                        "frequency": {
                            "type": "string",
                            "description": "The frequency field from a shatter_point.",
                        },
                        "fix_hint": {
                            "type": "string",
                            "description": "The fix_hint field from a shatter_point.",
                        },
                        "causal_chain": {
                            "type": "array",
                            "description": "The causal_chain array from a shatter_point.",
                            "items": {"type": "object"},
                        },
                    },
                    "required": ["failure_type", "message", "frequency"],
                },
            ),
        ]

    @server.call_tool()
    async def handle_call_tool(
        name: str, arguments: dict[str, Any]
    ) -> list[mcp_types.TextContent]:
        import asyncio

        loop = asyncio.get_event_loop()

        if name == "list_scenarios":
            result = await loop.run_in_executor(None, _tool_list_scenarios)
            return [mcp_types.TextContent(
                type="text",
                text=json.dumps(result, indent=2),
            )]

        if name == "run_blackswan":
            result = await loop.run_in_executor(
                None,
                lambda: _tool_run_blackswan(
                    file_path=arguments["file_path"],
                    scenario=arguments["scenario"],
                    function_name=arguments.get("function_name"),
                    iterations=arguments.get("iterations"),
                    seed=arguments.get("seed"),
                    mode=arguments.get("mode", "full"),
                ),
            )
            return [mcp_types.TextContent(
                type="text",
                text=json.dumps(result, indent=2),
            )]

        if name == "get_finding_detail":
            result = _tool_get_finding_detail(
                response=arguments["response"],
                finding_id=arguments["finding_id"],
            )
            return [mcp_types.TextContent(
                type="text",
                text=json.dumps(result, indent=2),
            )]

        if name == "explain_finding":
            text = _tool_explain_finding(
                failure_type=arguments.get("failure_type", ""),
                message=arguments.get("message", ""),
                frequency=arguments.get("frequency", ""),
                fix_hint=arguments.get("fix_hint", ""),
                causal_chain=arguments.get("causal_chain", []),
            )
            return [mcp_types.TextContent(type="text", text=text)]

        raise ValueError(f"Unknown tool: {name!r}")

    return server


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    if not _MCP_AVAILABLE:
        print(
            "ERROR: 'mcp' package is not installed.\n"
            "Install it with: pip install blackswan[mcp]\n"
            "or: pip install mcp",
            file=sys.stderr,
        )
        sys.exit(1)

    import asyncio

    server = _build_server()

    async def _run():
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options(),
            )

    asyncio.run(_run())


if __name__ == "__main__":
    main()
