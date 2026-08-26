"""验证两表最小服务的创建、推荐和同步学习闭环。"""

from __future__ import annotations

import argparse
import json
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from uuid import uuid4


def call_api(
    base_url: str,
    method: str,
    path: str,
    *,
    payload: dict[str, Any] | None = None,
) -> tuple[int, dict[str, Any] | None]:
    headers = {"Content-Type": "application/json"} if payload is not None else {}
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = Request(f"{base_url.rstrip('/')}{path}", data=body, headers=headers, method=method)
    try:
        with urlopen(request, timeout=10) as response:
            content = response.read()
            return response.status, json.loads(content) if content else None
    except HTTPError as error:
        content = error.read()
        detail = json.loads(content) if content else None
        raise RuntimeError(f"{method} {path} 返回 {error.code}: {detail}") from error


def route(
    route_id: str,
    *,
    time: float,
    cost: float,
    walking: float,
    transfers: int,
) -> dict[str, Any]:
    return {
        "route_id": route_id,
        "total_time_minutes": time,
        "total_cost": cost,
        "model_cost_unit": "CNY",
        "walking_distance_meters": walking,
        "transfer_count": transfers,
    }


def verify(base_url: str) -> dict[str, Any]:
    run_id = uuid4().hex
    fast = route("fast-expensive", time=20, cost=30, walking=300, transfers=1)
    balanced = route("balanced", time=30, cost=20, walking=500, transfers=1)
    cheap = route("cheap-slow", time=45, cost=5, walking=900, transfers=2)

    status, created = call_api(
        base_url,
        "POST",
        "/v1/users",
        payload={
            "external_user_id": f"verification-{run_id}",
            "initial_profile": {"mode": "default"},
        },
    )
    assert status == 201 and created is not None
    user_id = created["id"]
    before = created["profile"]["profile"]

    status, recommendation = call_api(
        base_url,
        "POST",
        "/v1/recommendations",
        payload={
            "user_id": user_id,
            "ranking_mode": "weighted",
            "routes": [fast, balanced, cheap],
            "constraints": {},
            "top_k": 3,
        },
    )
    assert status == 200 and recommendation is not None

    status, choice = call_api(
        base_url,
        "POST",
        f"/v1/users/{user_id}/choices",
        payload={"chosen_route": cheap, "rejected_route": fast},
    )
    assert status == 200 and choice is not None
    after = choice["profile"]["profile"]

    if after["evidence_count"] != 1:
        raise RuntimeError("画像没有累计本次有效选择")
    if before["coefficients"] == after["coefficients"]:
        raise RuntimeError("四维系数没有发生学习变化")

    _, persisted = call_api(base_url, "GET", f"/v1/users/{user_id}/profile")
    assert persisted is not None
    if persisted["profile"] != after:
        raise RuntimeError("更新后的画像没有保存到 user_profiles")

    return {
        "result": "PASS",
        "user_id": user_id,
        "ranked_route_ids": [item["route"]["route_id"] for item in recommendation["ranked_routes"]],
        "before": {
            "evidence_count": before["evidence_count"],
            "coefficients": before["coefficients"],
            "weights": before["weights"],
        },
        "after": {
            "evidence_count": after["evidence_count"],
            "coefficients": after["coefficients"],
            "weights": after["weights"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    print(json.dumps(verify(args.base_url), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
