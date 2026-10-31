"""Wire conversions for operator-owned collection sessions."""

import json
from dataclasses import asdict

from humanoid.types.collection import (
    CollectionAction,
    CollectionRequest,
    CollectionRunState,
    CollectionRunStatus,
    CollectionSettings,
)
from humanoid.types.lcm import collection_request_t, collection_run_status_t


def request_to_lcm(request: CollectionRequest) -> collection_request_t:
    wire = collection_request_t()
    wire.timestamp = round(request.timestamp * 1e9)
    wire.run_id = request.run_id
    wire.action = request.action.value
    wire.settings_json = json.dumps(asdict(request.settings)) if request.settings else ""
    return wire


def request_from_lcm(wire: collection_request_t) -> CollectionRequest:
    return CollectionRequest(
        wire.timestamp / 1e9,
        wire.run_id,
        CollectionAction(wire.action),
        CollectionSettings(**json.loads(wire.settings_json)) if wire.settings_json else None,
    )


def status_to_lcm(status: CollectionRunStatus) -> collection_run_status_t:
    wire = collection_run_status_t()
    for name, value in asdict(status).items():
        setattr(wire, name, value)
    wire.timestamp = round(status.timestamp * 1e9)
    wire.state = status.state.value
    return wire


def status_from_lcm(wire: collection_run_status_t) -> CollectionRunStatus:
    return CollectionRunStatus(
        timestamp=wire.timestamp / 1e9,
        state=CollectionRunState(wire.state),
        run_id=wire.run_id,
        task=wire.task,
        episodes=wire.episodes,
        max_attempts=wire.max_attempts,
        attempts=wire.attempts,
        successes=wire.successes,
        seed=wire.seed,
        phase=wire.phase,
        target=wire.target,
        sim_seconds=wire.sim_seconds,
        actions=wire.actions,
        samples=wire.samples,
        recording=wire.recording,
        message=wire.message,
    )
