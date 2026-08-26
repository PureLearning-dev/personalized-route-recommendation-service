from __future__ import annotations

from uuid import uuid4

import pytest
from sqlalchemy import func, inspect, select

from src.infrastructure.db.models import UserModel, UserProfileModel
from src.infrastructure.db.session import get_engine, get_session_factory

from .conftest import ApiHarness


def route(
    route_id: str,
    *,
    time: float,
    cost: float,
    walking: float,
    transfers: int,
) -> dict:
    return {
        "route_id": route_id,
        "total_time_minutes": time,
        "total_cost": cost,
        "model_cost_unit": "CNY",
        "walking_distance_meters": walking,
        "transfer_count": transfers,
    }


ROUTES = [
    route("fast-expensive", time=20, cost=30, walking=300, transfers=1),
    route("balanced", time=30, cost=20, walking=500, transfers=1),
    route("cheap-slow", time=45, cost=5, walking=900, transfers=2),
]


def create_user(
    api: ApiHarness,
    *,
    external_id: str = "traveller-001",
    mode: str = "default",
    preset: str | None = None,
):
    return api.client.post(
        "/v1/users",
        json={
            "external_user_id": external_id,
            "initial_profile": {"mode": mode, "preset": preset},
        },
    )


def recommendation_payload(user_id: str, *, ranking_mode: str = "weighted") -> dict:
    payload = {
        "user_id": user_id,
        "ranking_mode": ranking_mode,
        "routes": ROUTES,
        "constraints": {},
        "top_k": 3,
    }
    if ranking_mode == "jnd":
        payload["top_k"] = None
        payload["jnd"] = {
            "shortlist_size": 3,
            "top_k": 3,
            "thresholds": {
                "time_ratio": 0.1,
                "cost_ratio": 0.1,
                "walking_distance_ratio": 0.1,
                "transfers_ratio": 0.1,
            },
        }
    return payload


def test_database_has_only_two_business_tables(api: ApiHarness) -> None:
    tables = set(inspect(get_engine()).get_table_names())
    assert tables == {"alembic_version", "users", "user_profiles"}


def test_list_all_users_with_current_profiles(api: ApiHarness) -> None:
    assert api.client.get("/v1/users").json() == []
    first = create_user(api, external_id="list-user-001").json()
    second = create_user(
        api,
        external_id="list-user-002",
        mode="preset",
        preset="cost_priority",
    ).json()

    response = api.client.get("/v1/users")

    assert response.status_code == 200
    assert response.json() == [first, second]


def test_health_and_current_profile_crud(api: ApiHarness) -> None:
    assert api.client.get("/health/live").json() == {"status": "ok"}
    assert api.client.get("/health/ready").json() == {"status": "ready"}
    assert api.client.get(f"/v1/users/{uuid4()}/profile").status_code == 404

    created = create_user(api)
    assert created.status_code == 201
    body = created.json()
    user_id = body["id"]
    profile = body["profile"]["profile"]
    assert profile["evidence_count"] == 0
    assert profile["coefficients"] == {
        "time": 0.0,
        "cost": 0.0,
        "walking_distance": 0.0,
        "transfers": 0.0,
    }
    assert profile["weights"] == {
        "time": 0.25,
        "cost": 0.25,
        "walking_distance": 0.25,
        "transfers": 0.25,
    }

    queried = api.client.get(f"/v1/users/{user_id}/profile")
    assert queried.status_code == 200
    assert queried.json() == body["profile"]

    deleted = api.client.delete(f"/v1/users/{user_id}")
    assert deleted.status_code == 204
    assert api.client.get(f"/v1/users/{user_id}/profile").status_code == 404
    with get_session_factory()() as database:
        assert database.scalar(select(func.count()).select_from(UserModel)) == 0
        assert database.scalar(select(func.count()).select_from(UserProfileModel)) == 0


def test_named_preset_and_external_user_id_uniqueness(api: ApiHarness) -> None:
    default = create_user(api, external_id="same-external-id").json()
    preset = create_user(
        api,
        external_id="time-first",
        mode="preset",
        preset="time_priority",
    ).json()
    duplicate = create_user(api, external_id="same-external-id")

    assert duplicate.status_code == 409
    assert preset["profile"]["profile"]["weights"] != default["profile"]["profile"]["weights"]


@pytest.mark.parametrize("mode", ["weighted", "jnd"])
def test_recommendation_is_calculated_without_extra_tables(
    api: ApiHarness,
    mode: str,
) -> None:
    user_id = create_user(api, external_id=f"user-{mode}").json()["id"]
    response = api.client.post(
        "/v1/recommendations",
        json=recommendation_payload(user_id, ranking_mode=mode),
    )

    assert response.status_code == 200
    assert response.json()["candidate_count"] == 3
    assert response.json()["feasible_count"] == 3
    assert len(response.json()["ranked_routes"]) == 3
    if mode == "jnd":
        assert response.json()["explanation"]["attribute_priority"]

    with get_session_factory()() as database:
        assert database.scalar(select(func.count()).select_from(UserModel)) == 1
        assert database.scalar(select(func.count()).select_from(UserProfileModel)) == 1


def test_choice_updates_the_single_current_profile_row(api: ApiHarness) -> None:
    created = create_user(api)
    user_id = created.json()["id"]
    initial = created.json()["profile"]["profile"]

    choice = api.client.post(
        f"/v1/users/{user_id}/choices",
        json={
            "chosen_route": ROUTES[2],
            "rejected_route": ROUTES[0],
        },
    )
    assert choice.status_code == 200
    assert choice.json()["learning_applied"] is True
    learned = choice.json()["profile"]["profile"]
    assert learned["evidence_count"] == 1
    assert learned["weights"] != initial["weights"]

    current = api.client.get(f"/v1/users/{user_id}/profile").json()["profile"]
    assert current == learned
    with get_session_factory()() as database:
        assert database.scalar(select(func.count()).select_from(UserProfileModel)) == 1

    second = api.client.post(
        f"/v1/users/{user_id}/choices",
        json={
            "chosen_route": ROUTES[0],
            "rejected_route": ROUTES[2],
        },
    )
    assert second.json()["profile"]["profile"]["evidence_count"] == 2
    with get_session_factory()() as database:
        assert database.scalar(select(func.count()).select_from(UserProfileModel)) == 1


def test_identical_route_attributes_do_not_change_profile(api: ApiHarness) -> None:
    user_id = create_user(api).json()["id"]
    same_a = route("same-a", time=20, cost=10, walking=100, transfers=0)
    same_b = route("same-b", time=20, cost=10, walking=100, transfers=0)

    choice = api.client.post(
        f"/v1/users/{user_id}/choices",
        json={"chosen_route": same_a, "rejected_route": same_b},
    )

    assert choice.status_code == 200
    assert choice.json()["learning_applied"] is False
    assert choice.json()["profile"]["profile"]["evidence_count"] == 0


def test_invalid_recommendation_and_choice_contracts_return_422(api: ApiHarness) -> None:
    user_id = create_user(api).json()["id"]
    missing_jnd = recommendation_payload(user_id, ranking_mode="jnd")
    missing_jnd.pop("jnd")
    response = api.client.post("/v1/recommendations", json=missing_jnd)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "REQUEST_VALIDATION_ERROR"

    wrong_unit = recommendation_payload(user_id)
    wrong_unit["routes"] = [dict(item) for item in ROUTES]
    wrong_unit["routes"][0]["model_cost_unit"] = "USD"
    assert api.client.post("/v1/recommendations", json=wrong_unit).status_code == 422

    same_route_choice = {
        "chosen_route": ROUTES[0],
        "rejected_route": ROUTES[0],
    }
    assert (
        api.client.post(f"/v1/users/{user_id}/choices", json=same_route_choice).status_code == 422
    )
