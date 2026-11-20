"""Import-light MCP schemas shared by RAX and the trusted execution service."""

import os

CONTACT_PREVIEW_TASK = "toy_car_contact_preview"


def tool_definitions() -> list[dict]:
    """The pinned RAX discovery hook receives no config. Each runner hosts one task."""
    if os.environ.get("HUMANOID_RAX_TASK") == CONTACT_PREVIEW_TASK:
        return contact_preview_tools()
    return simulation_tools()


def _vector(length: int) -> dict:
    return {"type": "array", "minItems": length, "maxItems": length, "items": {"type": "number"}}


def simulation_tools() -> list[dict]:
    duration = {"type": "number", "exclusiveMinimum": 0, "maximum": 2, "default": 1}
    specs = (
        ("status", "Measured robot state and tool limits. No scene ground truth.", {}, []),
        (
            "observe",
            "Native RGB and depth images, separate intrinsics, and optical-to-base transforms.",
            {},
            [],
        ),
        (
            "move_tool",
            "Bounded Cartesian target for the configured tool frame, expressed in the robot "
            "base frame. Quaternion order x,y,z,w. Translation and rotation interpolate over "
            "duration, capped by robot tool velocity limits. Reports measured errors; a command "
            "is not a successful grasp.",
            {"position": _vector(3), "quaternion": _vector(4), "duration": duration},
            ["position", "quaternion"],
        ),
        (
            "move_base",
            "Bounded displacement [x,y,yaw] in the base frame at call start. Uses measured "
            "pose feedback. Duration limits execution time, not travel speed. Base velocity "
            "stops at the end of the call.",
            {"delta": _vector(3), "duration": duration},
            ["delta"],
        ),
        (
            "gripper",
            "Command gripper joint positions in radians within reported limits. For a grasp, "
            "target slightly beyond first contact in the closing direction to maintain squeeze, "
            "rather than matching object width exactly. The rate-limited target persists during "
            "lift and hold. This is position control, not force control; "
            "verify possession visually.",
            {"positions": _vector(1), "duration": duration},
            ["positions"],
        ),
        (
            "hold",
            "Hold measured joints and stop the base for the bounded duration.",
            {"duration": duration},
            [],
        ),
    )
    return [
        {
            "name": f"humanoid_{name}",
            "description": description,
            "inputSchema": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        }
        for name, description, properties, required in specs
    ]


def validate_request(name: str, arguments: dict) -> None:
    schemas = {tool["name"]: tool["inputSchema"] for tool in simulation_tools()}
    if not isinstance(name, str) or name not in schemas:
        raise ValueError(f"Unknown tool {name!r}.")
    schema = schemas[name]
    if (
        not isinstance(arguments, dict)
        or arguments.keys() - schema["properties"].keys()
        or set(schema["required"]) - arguments.keys()
    ):
        raise ValueError("Missing or unsupported tool arguments.")


def contact_preview_tools() -> list[dict]:
    tools = [t for t in simulation_tools() if t["name"] in {"humanoid_observe", "humanoid_status"}]
    tools.append(
        {
            "name": "humanoid_mark_contacts",
            "description": (
                "Save two proposed finger contacts on the latest native RGB image for "
                "operator review. Coordinates are integer [u, v] pixels with top-left origin."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "observation_id": {"type": "string"},
                    "contacts": {
                        "type": "array",
                        "minItems": 2,
                        "maxItems": 2,
                        "items": {
                            "type": "array",
                            "minItems": 2,
                            "maxItems": 2,
                            "items": {"type": "integer"},
                        },
                    },
                },
                "required": ["observation_id", "contacts"],
                "additionalProperties": False,
            },
        }
    )
    return tools
