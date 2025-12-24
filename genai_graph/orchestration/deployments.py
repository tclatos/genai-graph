"""Example Prefect deployments for KG orchestration.

This module is intentionally lightweight and serves primarily as documentation
for how to register deployments for the KG creation flow.
"""

from __future__ import annotations

from prefect.deployments import Deployment

from genai_graph.orchestration.flows import create_kg_flow


def build_local_create_kg_deployment(name: str = "local-kg-create", work_queue: str = "default") -> Deployment:
    """Build a Prefect deployment for the KG creation flow using a local agent.

    The caller is responsible for calling ``dep.apply()`` to register the
    deployment in a Prefect server or cloud environment.
    """

    return Deployment.build_from_flow(
        flow=create_kg_flow,
        name=name,
        work_queue_name=work_queue,
    )


if __name__ == "__main__":  # pragma: no cover - helper script
    deployment = build_local_create_kg_deployment()
    deployment.apply()
