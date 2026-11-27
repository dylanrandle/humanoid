# Networking

LCM connects robot processes; HTTP connects the console and agent clients.
Remote video also needs a direct WebRTC media path.

## LCM transport

LCM uses host-local multiport UDP (`mpudpm`, `ttl=0`). Each socket requests an
8 MiB receive buffer through `recv_buf_size`.

- **Jetson** deployment applies the buffer settings automatically.
- On **other Linux hosts**, copy the buffer configuration into `/etc/sysctl.d/`
  and run `sudo sysctl -p /etc/sysctl.d/90-humanoid-lcm.conf`. [Configuration file](../deploy/90-humanoid-lcm.conf)
- On **macOS**, if `sysctl -n kern.ipc.maxsockbuf` is below 8388608, run
  `sudo sysctl -w kern.ipc.maxsockbuf=8388608` for the current boot.

`uv run viz --host triskel` streams state and commands over SSH. Use matching robot
assets on both hosts. Set `HUMANOID_ROBOT_DIR` for a custom remote installation.
If you override `LCM_DEFAULT_URL` for the robot service,
set the same URL in the robot's SSH environment.

For direct multicast across a subnet, set this on each host before launch.

```bash
export LCM_DEFAULT_URL='mpudpm://239.255.76.67:7667?ttl=1&recv_buf_size=8388608'
```

## Remote API access

The HTTP listener binds to loopback. Use the port forwarding workflow in
[Deployment](deployment.md), or an SSH tunnel for a remote agent client:

```bash
ssh -L 8765:127.0.0.1:8765 triskel
```

`HUMANOID_API_URL` selects the Python client's endpoint. An explicit `ApiEndpoint`
can override it. Reuse the forwarded port for both the console and clients.

## Camera streaming

The console uses WebRTC for RGB and colorized native-depth video. The browser
needs a direct media path to the backend, such as the same host or LAN. The
backend uses local ICE candidates without external discovery or relay servers.
An SSH HTTP tunnel carries API calls and signaling, but does not tunnel video.

For metric snapshots rather than video, use the [robot client](agents.md), which
works over the HTTP tunnel. For display controls, see [Operating the robot](operation.md).

## See also

- [Documentation index](README.md)
- [Troubleshooting](troubleshooting.md)
- [Backend configuration](../src/humanoid/config/api.py)
