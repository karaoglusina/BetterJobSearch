"""Cluster tools: cluster_by_aspect, browse_cluster, aspect_distribution."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from .registry import ToolRegistry


def cluster_by_aspect(aspect: str) -> str:
    """Cluster jobs by a given aspect (skills, language, domain, etc.)."""
    # This is a placeholder — actual implementation requires loaded embeddings
    # and aspect data which are wired in at the API/coordinator level.
    return json.dumps({
        "status": "clustering_requested",
        "aspect": aspect,
        "message": f"Requested re-clustering by '{aspect}'. The UI will update the scatter plot.",
    })


def browse_cluster(cluster_id: int, n: int = 5) -> str:
    """Browse sample jobs from a specific cluster."""
    return json.dumps({
        "status": "browse_requested",
        "cluster_id": cluster_id,
        "n": n,
        "message": f"Showing {n} sample jobs from cluster {cluster_id}.",
    })


def cluster_by_concept(concept: str) -> str:
    """Cluster jobs by a free-text concept."""
    return json.dumps({
        "status": "concept_clustering_requested",
        "concept": concept,
        "message": f"Requested concept-based clustering for '{concept}'. The UI will update.",
    })


def aspect_distribution(aspect: str) -> str:
    """Show distribution of values for an aspect across all jobs."""
    return json.dumps({
        "status": "distribution_requested",
        "aspect": aspect,
        "message": f"Showing value distribution for '{aspect}'.",
    })


def register_cluster_tools(registry: ToolRegistry) -> None:
    """Register cluster tools."""
    registry.register("cluster_by_aspect", cluster_by_aspect, {
        "name": "cluster_by_aspect",
        "description": "Re-cluster the job scatter plot by a specific aspect (e.g., skills, language, domain, remote_policy). Triggers UI update.",
        "parameters": {
            "type": "object",
            "properties": {
                "aspect": {"type": "string", "description": "Aspect to cluster by: skills, tools, language, remote_policy, experience, education, benefits, domain, culture"},
            },
            "required": ["aspect"],
        },
    })

    registry.register("browse_cluster", browse_cluster, {
        "name": "browse_cluster",
        "description": "Show sample jobs from a specific cluster ID.",
        "parameters": {
            "type": "object",
            "properties": {
                "cluster_id": {"type": "integer", "description": "Cluster ID from the scatter plot"},
                "n": {"type": "integer", "description": "Number of sample jobs to show (default 5)", "default": 5},
            },
            "required": ["cluster_id"],
        },
    })

    registry.register("cluster_by_concept", cluster_by_concept, {
        "name": "cluster_by_concept",
        "description": "Re-cluster jobs by a free-text concept (e.g., 'customer-facing', 'startup culture'). Uses semantic similarity.",
        "parameters": {
            "type": "object",
            "properties": {
                "concept": {"type": "string", "description": "Free-text concept to cluster by"},
            },
            "required": ["concept"],
        },
    })

    registry.register("aspect_distribution", aspect_distribution, {
        "name": "aspect_distribution",
        "description": "Show the distribution of values for an aspect across all jobs (e.g., how many jobs require Dutch, how many are remote).",
        "parameters": {
            "type": "object",
            "properties": {
                "aspect": {"type": "string", "description": "Aspect name"},
            },
            "required": ["aspect"],
        },
    })
