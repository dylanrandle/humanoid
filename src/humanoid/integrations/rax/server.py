"""Trusted RPC dispatch, mounted by the existing operator UI server."""

from humanoid.integrations.rax.session import RealtimeSession


def dispatch(session: RealtimeSession, request: dict):
    if not isinstance(request, dict) or set(request) != {"method", "arguments"}:
        raise ValueError("Expected method and arguments.")
    method, arguments = request["method"], request["arguments"]
    if not isinstance(arguments, dict):
        raise ValueError("arguments must be an object.")
    if method == "start":
        if set(arguments) != {"task", "formal", "seed"} or arguments["task"] != session.task_name:
            raise ValueError("Invalid task or start arguments.")
        value, images = session.start(formal=arguments["formal"], seed=arguments["seed"])
    elif method == "call":
        if set(arguments) != {"name", "arguments"}:
            raise ValueError("Invalid call arguments.")
        value, images = session.call(arguments["name"], arguments["arguments"])
    elif method in {"evaluate", "close"} and not arguments:
        return session.evaluate() if method == "evaluate" else session.close()
    else:
        raise ValueError("Unknown method or unexpected arguments.")
    return {"value": value, "images": images}
