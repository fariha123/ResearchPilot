"""Valency Bond Streamable HTTP MCP client, compatible with MCP Python SDK 2.x.

No bearer tokens are ever logged or included in exception messages by this module.
"""
import asyncio
import json
import os
from contextlib import asynccontextmanager

import httpx2
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

URL = os.getenv("VALENCY_MCP_URL", "https://mcp.valency.io/")


def _clean_token(token):
    token = (token or "").strip()
    if token.lower().startswith("bearer "):
        token = token[7:].strip()
    if not token:
        raise ValueError("Valency bearer token is missing. Copy the complete tok_... value from Valency Settings.")
    return token


@asynccontextmanager
async def connect(token):
    headers = {"Authorization": "Bearer " + _clean_token(token)}
    timeout = httpx2.Timeout(30.0, read=120.0)
    async with httpx2.AsyncClient(headers=headers, timeout=timeout, follow_redirects=True) as http_client:
        async with streamable_http_client(URL, http_client=http_client) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield session


def decode(result):
    # MCP SDK uses Python snake_case attributes; older wire JSON uses camelCase.
    structured = getattr(result, "structured_content", None)
    if structured is None:
        structured = getattr(result, "structuredContent", None)
    if structured is not None:
        return structured
    parts = [getattr(block, "text", "") for block in (getattr(result, "content", None) or [])]
    text = "\n".join(p for p in parts if p)
    if not text:
        return {"raw": str(result)}
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return {"raw": text}


def _schema(tool):
    """Access MCP Tool schema across SDK versions and Python/wire aliases."""
    schema = getattr(tool, "input_schema", None)
    if schema is None:
        schema = getattr(tool, "inputSchema", None)
    if schema is None and hasattr(tool, "model_dump"):
        raw = tool.model_dump(by_alias=True)
        schema = raw.get("inputSchema") or raw.get("input_schema")
    return schema if isinstance(schema, dict) else {}


def _arguments(tool, question, start, end, limit):
    schema = _schema(tool)
    props = schema.get("properties") or {}
    required = schema.get("required") or []
    arguments = {}
    options = {
        "query": question, "search_query": question, "text": question,
        "keywords": question, "abstract": question, "title": question,
        "start_date": start, "from_date": start, "date_from": start,
        "published_after": start, "publication_date_from": start,
        "end_date": end, "to_date": end, "date_to": end,
        "published_before": end, "publication_date_to": end,
        "limit": limit, "max_results": limit, "per_page": limit, "count": limit,
        "num_results": limit, "page_size": limit,
    }
    for key, value in options.items():
        if key in props:
            spec = props[key] or {}
            # Tool schemas sometimes expect a string even for result limits.
            arguments[key] = str(value) if spec.get("type") == "string" and isinstance(value, int) else value
    # Never call a search with an empty query.
    if not any(k in arguments for k in ("query", "search_query", "text", "keywords", "abstract", "title")):
        return None, "Cannot determine its query parameter from advertised schema"
    missing = [name for name in required if name not in arguments]
    if missing:
        return None, "Unsupported required parameter(s): " + ", ".join(missing)
    return arguments, None


async def diagnose_bond(token):
    async with connect(token) as session:
        tools = (await session.list_tools()).tools
        names = [t.name for t in tools]
        details = {t.name: sorted((_schema(t).get("properties") or {}).keys()) for t in tools}
        sources = None
        if "list_sources" in names:
            outcome = await session.call_tool("list_sources", {})
            if getattr(outcome, "is_error", False) or getattr(outcome, "isError", False):
                sources = {"error": decode(outcome)}
            else:
                sources = decode(outcome)
        return {"tools": names, "parameters": details, "sources": sources}


def diagnose(token):
    return asyncio.run(diagnose_bond(token))


async def query_bond(token, question, start, end, limit=25):
    async with connect(token) as session:
        advertised = (await session.list_tools()).tools
        tool_map = {t.name: t for t in advertised}
        preferred = ["search_by_abstract", "search_by_title"]
        candidates = [name for name in preferred if name in tool_map]
        if not candidates:
            candidates = [name for name in tool_map if "search" in name.lower()]
        if not candidates:
            raise RuntimeError("Connected to Bond, but no search tools advertised. Tools: " + ", ".join(tool_map))
        results = []
        errors = []
        for name in candidates[:3]:
            args, reason = _arguments(tool_map[name], question, start, end, limit)
            if reason:
                errors.append(f"{name}: {reason}; advertised parameters: {list((_schema(tool_map[name]).get('properties') or {}).keys())}")
                continue
            try:
                outcome = await session.call_tool(name, args)
                if getattr(outcome, "is_error", False) or getattr(outcome, "isError", False):
                    errors.append(f"{name}: Bond reported a tool error: {str(decode(outcome))[:500]}")
                    continue
                results.append((name, decode(outcome)))
            except Exception as exc:
                errors.append(f"{name}: {type(exc).__name__}: {str(exc)[:350]}")
        if not results:
            raise RuntimeError("Bond connection succeeded, but no search returned usable results. " + " | ".join(errors))
        return results


def search(token, question, start, end, limit=25):
    return asyncio.run(query_bond(token, question, start, end, limit))
