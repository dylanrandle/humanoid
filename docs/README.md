# Documentation

Humanoid controls robots through the same stack in simulation and on hardware.
Start with setup, then use the guides for the task at hand. Commands assume the
repository root unless a guide says otherwise.

## Get started

- [Getting started](getting-started.md): install dependencies and run a simulation.
- [Operating the robot](operation.md): console, control inputs, viewer, and recordings.
- [Troubleshooting](troubleshooting.md): common issues and recovery steps.

## Run tasks

- [Agent control](agents.md): use the ChatGPT panel or Python robot client.
- [Agent trials](trials.md): explore, evaluate, and review toy cleanup runs.
- [Deployment](deployment.md): install on Triskel and connect a wireless headset.
- [Local HTTPS gateway](gateway.md): persistent dashboard and Quest access.
- [Networking](networking.md): LCM buffers, remote access, and camera transport.
- [Diagnostics](diagnostics.md): inspect controller tracking and grasp performance.

## Understand the system

- [Architecture](architecture.md): major components and configuration ownership.
- [Robot backend](api.md): shared API, service boundaries, and motion sessions.
- [Simulation](simulation.md): MuJoCo scenes, cameras, and model limitations.
- [State estimation](state-estimation.md): how wheel feedback becomes base pose.
- [Hardware](hardware/README.md): actuator, camera, and teleoperation device integration.
- [Feetech actuators](hardware/feetech.md): servo setup and maintenance.
- [Cameras](hardware/cameras.md): RealSense setup and depth interpretation.
- [Quest headset](hardware/quest.md): setup, transport architecture, and controller assets.
- [NanoSAM](models/nanosam.md) and [camera depth models](models/cdm.md): model setup and rebuilding.

## Develop

- [Development](development.md): verification and generated code.
- [Contributor instructions](../AGENTS.md): repository conventions and documentation rules.

## See also

- [Project overview](../README.md)
