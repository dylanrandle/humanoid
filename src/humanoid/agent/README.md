# Agent control API

An agent can read robot state and RGB-D images over HTTP and submit bounded motion commands. The operator enables motion by selecting **Autonomy** in the console. `AgentPolicy` runs the commands through the same LCM orchestrator and controllers as the other policies.

## Try it in MuJoCo

1. Run `uv run start`. In the operator console, select **Triskel** and **Simulation**, then start the main stack and wait for the mode to show **Idle**.
2. Under **Control inputs → Agent API**, click **Start**. Local agents can now read state and camera frames while the robot is in Idle.
3. Select **Autonomy** to enable motion. From a terminal on the same machine, request a 2 cm translation along the base's local x-axis.

   ```sh
   uv run python - <<'PYTHON'
   from humanoid.agent.client import AgentControlClient
   from humanoid.types.agent import ControlOperation

   client = AgentControlClient()
   try:
       client.wait_until_ready()
       print("Initial joint positions", client.state().joint_positions)
       result = client.execute(
           ControlOperation.MOVE_BASE, {"delta": [0.02, 0, 0], "duration": 1.2}
       )
       print("Reached", result["reached"], "position error (m)", result["position_error_m"])
       print("Final joint positions", client.state().joint_positions)
   finally:
       client.close()
   PYTHON
   ```

4. Select **Idle** to end agent control. Click **Stop** in the Agent API row to shut down the listener. Stopping the main stack also stops the API.

Call `client.observe()` for paired RGB and native depth arrays with calibration. An agent on another machine can reach the default loopback listener through an SSH tunnel such as `ssh -L 8767:127.0.0.1:8767 triskel`. The listener binds only to `127.0.0.1`. Any process on that host can access it while it is running. Set `HUMANOID_CONTROL_PORT` in the dashboard environment to change the listener port. For a different local port, pass `AgentControlEndpoint("http://127.0.0.1:<port>")` to the client.

## HTTP contract

POST bodies use `Content-Type: application/json`.

| Endpoint | Response |
| --- | --- |
| `GET /v1/state` | Current session ID, operator mode, joint feedback and feedback age in seconds |
| `GET /v1/camera` | Paired RGB and depth PNGs in base64, intrinsics per image, frame IDs, depth scale, timestamps and age |
| `POST /v1/commands` | HTTP 202 with the command ID and current status |
| `GET /v1/commands/<id>` | Command status and, on completion, measured result |
| `POST /v1/commands/<id>/cancel` | Updated status for that command. Send `{}` as the body. |

Use the session ID from `/v1/state` and a unique client-generated command ID.

```json
{
  "session_id": "session-from-state",
  "command_id": "move-001",
  "operation": "move_base",
  "arguments": {"delta": [0.02, 0, 0], "duration": 1.2},
  "expires_in_seconds": 1.0
}
```

| Operation | Arguments |
| --- | --- |
| `move_tool` | Absolute `position` in metres and unit `quaternion` in x, y, z, w order. Tool targets use the configured base frame for a mobile robot and the world frame for a fixed-base arm. |
| `move_base` | `delta` as forward x (m), lateral y (m), and yaw (rad), relative to the mobile base at command start. |
| `set_gripper` | `positions` in configured gripper-joint order, using each joint's native units (rad for Triskel). |
| `hold` | Interrupt the active motion and hold the measured pose while retaining the gripper target. |

All operations accept `duration` in seconds, default 1 and maximum 2. One command can be pending or running at a time. Overlapping motion gets HTTP 409. Other 409 responses include stale feedback, missing fresh camera frames and an inactive control mode. Cross-origin browser requests return 403 and malformed commands return 400. The policy also checks workspace, displacement and joint limits against feedback on the execution tick.

Poll the command ID until its state leaves `queued` or `running`. Terminal states are `completed`, `rejected`, `cancelled`, `expired` and `failed`. A completed command reports `reached` and the measured position, orientation and gripper errors. The duration can elapse before the robot reaches its target. Leaving Autonomy interrupts outstanding commands.

If a submission response is lost, resend the **same JSON body**, including the session ID, command ID and deadline. The service returns the existing status and measured result for that ID. A retry neither repeats the movement nor extends its start deadline. Reusing the ID for different arguments fails. After a service restart, read the new state and reconcile the robot's position before deciding on another movement. Each call to the Python client's `execute()` creates a new command ID. Use the HTTP contract directly when the caller needs to retain its own IDs.
