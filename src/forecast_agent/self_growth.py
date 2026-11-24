"""Dynamic self-growing agent support.

This module provides the infrastructure for the agent to dynamically grow
itself based on incoming queries. When a query requires a capability that
does not yet exist, the agent can:

1. Analyze the query to determine what tools are needed
2. Create new tool implementations on the fly
3. Register them in the MCP server
4. Make them available to the pipeline stages
"""
from __future__ import annotations

import re
import inspect
import importlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

@dataclass
class ToolSignature:
    name: str
    description: str
    parameters: dict[str, Any]
    module: str
    identifier: str

class ToolRegistry:
    def __init__(self):
        self.tools: dict[str, ToolSignature] = {}

    def register(self, tool: ToolSignature) -> None:
        self.tools[tool.name] = tool

    def find_tool(self, query: str) -> Optional[ToolSignature]:
        query_lower = query.lower()
        for tool in self.tools.values():
            if tool.name.lower() in query_lower:
                return tool
        keywords = ["stock", "price", "forecast", "probability",
                    "technical indicator", "rsi", "sma", "ema", "volatility",
                    "news", "event", "event table", "reference",]
        for tool in self.tools.values():
            tool_lower = tool.description.lower()
            tool_lower += " " + tool.name.lower()
            for kw in keywords:
                if kw in query_lower and kw in tool_lower:
                    return tool
        return None

    def list_tools(self) -> list[ToolSignature]:
        return list(self.tools.values())

    def register_tool(self, name: str, description: str, parameters: dict[str, Any], module: str, identifier: str) -> None:
        tool = ToolSignature(
            name=name,
            description=description,
            parameters=parameters,
            module=module,
            identifier=identifier,
        )
        self.tools[tool.name] = tool
        print(f"Registered tool in registry: {tool.name}")

    def load_tool_module(self, module_name: str):
        """Load a tool module by name."""
        try:
            return importlib.import_module("src.forecast_agent.mcp_servers." + module_name)
        except ImportError:
            try:
                return importlib.import_module("forecast_agent.mcp_servers." + module_name)
            except ImportError:
                return None

class DynamicToolGenerator:
    def __init__(self, registry: ToolRegistry):
        self.registry = registry
        self._tool_modules: dict[str, Any] = {}
        self._tool_directory: Path = Path("src/forecast_agent/mcp_servers")

    def generate_tool(self, query: str, tool_type: Optional[str] = None) -> dict[str, Any]:
        query_lower = query.lower()
        if tool_type is None:
            tool_type = self._classify_query(query_lower)
        return self._tool_templates[tool_type](query, query_lower)

    def _classify_query(self, query: str) -> str:
        if any(kw in query for kw in ["stock", "price", "forecast", "percent", "%"]):
            return "stock_analysis"
        if any(kw in query for kw in ["rsi", "sma", "ema", "indicator", "technical"]):
            return "indicator"
        if any(kw in query for kw in ["reference", "base rate", "prior", "class", "base_rate"]):
            return "reference"
        if any(kw in query for kw in ["event", "table", "dates", "timeline", "when"]):
            return "event"
        if any(kw in query for kw in ["politic", "election", "vote", "government", "president", "policy", "tax", "regulation", "law", "congress", "senate", "supreme", "treaty", "diplomacy", "embassy"]):
            return "politics"
        if any(kw in query for kw in ["sport", "game", "match", "team", "player", "score", "winner", "championship", "league", "football", "basketball", "cricket", "soccer", "racing", "olympic", "tournament", "playoff"]):
            return "sports"
        if any(kw in query for kw in ["education", "school", "student", "learning", "teacher", "university", "college", "classroom", "curriculum", "academic"]):
            return "education"
        if any(kw in query for kw in ["environment", "climate", "weather", "temperature", "pollution", "emission", "carbon", "greenhouse", "renewable", "solar", "wind", "ecosystem", "deforestation", "conservation", "rain", "storm", "flood", "drought"]):
            return "environment"
        if any(kw in query for kw in ["economy", "inflation", "unemployment", "gdp", "recession", "growth", "interest rate", "fiscal", "monetary", "currency", "exchange rate", "trade"]):
            return "economics"
        if any(kw in query for kw in ["tech", "technology", "ai", "artificial intelligence", "software", "computer", "cyber", "security", "robot", "automation", "app", "algorithm", "cloud", "blockchain", "data"]):
            return "technology"
        if any(kw in query for kw in ["vaccine", "vaccination", "disease", "treatment", "hospital", "patient", "medical", "pharma", "drug", "virus", "pandemic", "healthcare", "medicine", "symptom", "clinical", "epidemic", "health", "healthcare", "health"]):
            return "health"
        if any(kw in query for kw in ["physic", "quantum", "research", "scientific", "scientific breakthrough", "experiment", "astronomy", "biology", "chemistry", "data science", "science", "fusion", "energy"]):
            return "science"
        return "general"

    def _extract_entities(self, query_lower: str) -> list[str]:
        entities = []
        words = re.findall(r"[a-z]+", query_lower)
        for w in words:
            if len(w) > 3:
                entities.append(w)
        return entities

    def _generate_stock_analysis_tool(self, query: str, query_lower: str) -> dict[str, Any]:
        return {
            "name": "forecast_stock_" + query_lower[:30],
            "description": f"Forecast stock probability over 14 days based on query: {query}...",
            "parameters": {
                "data": {"type": "array", "description": "Historical stock data series"},
                "horizon_days": {"type": "integer", "description": "Forecast horizon (14)"},
                "method": {"type": "string", "description": "Forecast method: naive, ets, arima"},
                "confidence": {"type": "number", "description": "Confidence level (0.0-1.0)", "default": 0.0},
                "domain": {"type": "string", "description": "The domain (stock)"},
                "positive_direction": {"type": "string", "description": "What constitutes a positive outcome"},
            },
            "module": "quant_server",
            "identifier": "stock_forecast",
        }

    def _generate_indicator_tool(self, query: str, query_lower: str) -> dict[str, Any]:
        return {
            "name": "forecast_indicator_" + query_lower[:30],
            "description": f"Calculate technical indicator based on query: {query}...",
            "parameters": {
                "data": {"type": "array", "description": "Historical price data"},
                "indicator": {"type": "string", "description": "Indicator: rsi, sma, ema"},
                "period": {"type": "integer", "description": "Indicator period"},
            },
            "module": "quant_server",
            "identifier": "technical_indicator",
        }

    def _generate_forecast_tool(self, query: str, query_lower: str) -> dict[str, Any]:
        return {
            "name": "forecast_" + query_lower[:30],
            "description": f"Forecast probability of positive outcome over 14 days based on query: {query}...",
            "parameters": {
                "data": {"type": "array", "description": "Historical data series"},
                "horizon_days": {"type": "integer", "description": "Forecast horizon (14)"},
                "method": {"type": "string", "description": "Forecast method: naive, ets, arima"},
                "confidence": {"type": "number", "description": "Confidence level (0.0-1.0)", "default": 0.0},
                "domain": {"type": "string", "description": "The domain"},
                "positive_direction": {"type": "string", "description": "What constitutes a positive outcome"},
            },
            "module": "quant_server",
            "identifier": "forecast",
        }

    def _generate_reference_tool(self, query: str, query_lower: str) -> dict[str, Any]:
        return {
            "name": "get_reference_" + query_lower[:30],
            "description": f"Reference data for query: {query}...",
            "parameters": {
                "query": {"type": "string", "description": "Reference query"},
            },
            "module": "data_server",
            "identifier": "reference",
        }

    def _generate_event_tool(self, query: str, query_lower: str) -> dict[str, Any]:
        return {
            "name": "get_event_table_" + query_lower[:30],
            "description": f"Event table for query: {query}...",
            "parameters": {
                "entity": {"type": "string", "description": "Entity to filter"},
                "start_date": {"type": "string", "description": "Start date"},
                "end_date": {"type": "string", "description": "End date"},
            },
            "module": "data_server",
            "identifier": "event_table",
        }

    def _generate_generic_domain_tool(self, query: str, query_lower: str) -> dict[str, Any]:
        entities = self._extract_entities(query_lower)
        horizon = 14
        horizon_match = re.search(r"(\d+)\s*(day|days|week|weeks|month|months|year|years)", query_lower)
        if horizon_match:
            num = int(horizon_match.group(1))
            unit = horizon_match.group(2)
            if "week" in unit:
                horizon = num * 7
            elif "month" in unit:
                horizon = num * 30
            elif "year" in unit:
                horizon = num * 365
            else:
                horizon = num

        domain = self._determine_domain(query_lower)
        domain_stem = domain.replace("_", "")
        clean_query = re.sub(r"[^a-z0-9]+", "_", query_lower).strip("_")
        clean_query = clean_query[:30]

        name = f"forecast_{domain}_{clean_query}"
        description = f"Forecast {domain} probability of positive outcome over {horizon} days based on query: {query[:80]}..."

        parameters = {
            "data": {"type": "array", "description": f"Historical {domain} data series"},
            "horizon_days": {"type": "integer", "description": f"Forecast horizon ({horizon})"},
            "method": {"type": "string", "description": "Forecast method: naive, ets, arima"},
            "confidence": {"type": "number", "description": "Confidence level (0.0-1.0)", "default": 0.0},
            "domain": {"type": "string", "description": f"The domain ({domain})"},
            "positive_direction": {"type": "string", "description": "What constitutes a positive outcome"},
        }

        return {
            "name": name,
            "description": description,
            "parameters": parameters,
            "module": "quant_server",
            "identifier": f"{domain}_forecast",
        }

    def _determine_domain(self, query: str) -> str:
        domain_keywords = {
            "politics": ["politic", "election", "vote", "government", "president", "policy", "tax", "regulation", "law", "congress", "senate", "supreme", "treaty", "diplomacy", "embassy"],
            "sports": ["sport", "game", "match", "team", "player", "score", "winner", "championship", "league", "football", "basketball", "cricket", "soccer", "racing", "olympic", "tournament", "playoff"],
            "education": ["education", "school", "student", "learning", "teacher", "university", "college", "classroom", "curriculum", "academic"],
            "environment": ["environment", "climate", "weather", "temperature", "pollution", "emission", "carbon", "greenhouse", "renewable", "solar", "wind", "ecosystem", "deforestation", "conservation", "rain", "storm", "flood", "drought"],
            "economics": ["economy", "inflation", "unemployment", "gdp", "recession", "growth", "interest rate", "fiscal", "monetary", "currency", "exchange rate", "trade"],
            "technology": ["tech", "technology", "ai", "artificial intelligence", "software", "computer", "cyber", "security", "robot", "automation", "app", "algorithm", "cloud", "blockchain", "data"],
            "health": ["vaccine", "vaccination", "disease", "treatment", "hospital", "patient", "medical", "pharma", "drug", "virus", "pandemic", "healthcare", "medicine", "symptom", "clinical", "epidemic", "health"],
            "science": ["physic", "quantum", "research", "scientific", "scientific breakthrough", "experiment", "astronomy", "biology", "chemistry", "data science", "science", "fusion", "energy"],
        }

        for domain, keywords in domain_keywords.items():
            if any(kw in query for kw in keywords):
                return domain
        return "general"

    @property
    def _tool_templates(self) -> dict[str, Callable]:
        return {
            "stock_analysis": self._generate_stock_analysis_tool,
            "indicator": self._generate_indicator_tool,
            "forecast": self._generate_forecast_tool,
            "reference": self._generate_reference_tool,
            "event": self._generate_event_tool,
            "politics": self._generate_generic_domain_tool,
            "sports": self._generate_generic_domain_tool,
            "economics": self._generate_generic_domain_tool,
            "health": self._generate_generic_domain_tool,
            "technology": self._generate_generic_domain_tool,
            "environment": self._generate_generic_domain_tool,
            "science": self._generate_generic_domain_tool,
            "education": self._generate_generic_domain_tool,
            "general": self._generate_generic_domain_tool,
        }


@dataclass
class ToolRegistrar:
    registry: ToolRegistry
    mcp_server: Optional[Any] = None

    def register_tool_definition(self, tool_def: dict[str, Any]) -> None:
        tool = ToolSignature(
            name=tool_def["name"],
            description=tool_def["description"],
            parameters=tool_def["parameters"],
            module=tool_def["module"],
            identifier=tool_def["identifier"],
        )
        self.registry.register(tool)
        print(f"Registered tool: {tool.name}")

    def register_tool_with_server(self, tool_def: dict[str, Any], server) -> ToolSignature:
        tool = ToolSignature(
            name=tool_def["name"],
            description=tool_def["description"],
            parameters=tool_def["parameters"],
            module=tool_def["module"],
            identifier=tool_def["identifier"],
        )
        self.registry.register(tool)
        print(f"Registered tool with server: {tool.name}")
        return tool

class SelfGrowingAgent:
    def __init__(self, registry: ToolRegistry, generator: DynamicToolGenerator):
        self.registry = registry
        self.generator = generator
        self.registrar = ToolRegistrar(registry=registry)
        self._mcp_servers: dict[str, Any] = {}

    def _get_or_create_server(self, module_name: str) -> Any:
        if module_name in self._mcp_servers:
            return self._mcp_servers[module_name]
        try:
            mod = self.registry.load_tool_module(module_name)
            if mod is None:
                raise ImportError(f"Could not load module {module_name}")
            create_fn = getattr(mod, f"create_{module_name}_server", None)
            if create_fn is None:
                create_fn = getattr(mod, f"create_{module_name}", None)
            if create_fn is None:
                raise AttributeError(f"Module {module_name} has no create_ function")
            server = create_fn()
            self._mcp_servers[module_name] = server
            return server
        except ImportError:
            from mcp.server.fastmcp import FastMCP
            server = FastMCP(name=f"dynamic-{module_name}")
            self._mcp_servers[module_name] = server
            return server

    def _analyze_query(self, query: str) -> list[dict[str, Any]]:
        needed = []
        query_lower = query.lower()
        tool_type = self.generator._classify_query(query_lower)
        tool_def = self.generator.generate_tool(query, tool_type)
        if tool_def:
            needed.append(tool_def)
        return needed
