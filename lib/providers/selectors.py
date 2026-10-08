"""Selectors choose a child; only its broker can admit paid execution."""

from lib.providers.catalog import PAID_TOOLS
from lib.providers.context import in_worker


def explicit_tool(inputs, candidates):
    route = inputs.get("studio_route_id")
    if in_worker() or not route:
        return False, None
    name = str(route).split(":", 1)[0]
    allowed = set(inputs.get("allowed_providers") or [])
    selected = next(
        (
            tool
            for tool in candidates
            if tool.name == name and (not allowed or tool.provider in allowed)
        ),
        None,
    )
    return True, selected


def delegate(tool, inputs):
    if tool.name in PAID_TOOLS and not in_worker():
        properties = tool.input_schema.get("properties", {})
        inputs = {
            key: value
            for key, value in inputs.items()
            if key in properties or key.startswith("studio_") or key == "scene_id"
        }
    return tool.execute(inputs)
