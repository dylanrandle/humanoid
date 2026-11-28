# Local HTTPS gateway

The gateway exposes the operator dashboard and Quest viewer over HTTPS on the
local network. It runs Caddy as `humanoid-gateway.service` alongside the backend,
for both hardware and simulation.

## Service responsibilities

`humanoid.service` runs `BackendApplication`, including the dashboard API and
Quest server. `humanoid-gateway.service` manages local TLS certificates and proxies
HTTP and WebSocket connections to those listeners on loopback. Separate services
allow the gateway to restart without restarting the robot backend, and keep
permission to bind HTTPS port 443 with Caddy.

The deployed Triskel addresses are:

| Interface | HTTPS address | Backend listener |
| --- | --- | --- |
| Operator dashboard | `https://triskel.local` | `127.0.0.1:8765` |
| Quest viewer | `https://triskel.local:8443/connect` | `127.0.0.1:8012` |

WebRTC camera media travels directly between the backend and browser. The gateway
carries signaling, rather than the video itself. See
[Networking](networking.md#camera-streaming) for the media path.

## Install and configure

Follow the [deployment workflow](deployment.md#deploy-the-stack). It installs the
backend, gateway, and Avahi for `.local` discovery in one step. No separate gateway
installer or router configuration is needed. Clients and the host need a local
network that permits mDNS and direct peer traffic.

Deployment sets `HUMANOID_PUBLIC_HOST` in both services from the robot's
`hostname -s`, with `.local` appended. It controls Caddy's HTTPS hostname, the
backend's accepted public host and origin, and the dashboard's Quest viewer link.
Its value must be a bare `.local` hostname, such as `triskel.local`, without a
scheme or port. Leave it unset for localhost-only operation. For a manually
launched backend, `--public-host` overrides the environment variable; it does not
configure the gateway or hostname discovery. The variable is defined by
`EnvironmentVariable.PUBLIC_HOST` in the [shared enum](../src/humanoid/constants.py).

Host and port defaults are shared through the `Host` and `Port` enums in
[constants.py](../src/humanoid/constants.py):

- `Port.QUEST_GATEWAY` selects the public HTTPS listener for the headset.
- `Port.QUEST_SERVER` selects its internal HTTP listener behind the gateway.
- `Port.DASHBOARD_GATEWAY` selects the public dashboard listener; `Port.API` selects its
  internal HTTP listener.

Deployment renders the [Caddy template](../deploy/Caddyfile) from these defaults
before packaging it with the backend. To change deployed ports, update the
enums and redeploy; do not edit the generated Caddyfile separately.
Local environment overrides used for development do not change deployment defaults.
The USB and SSH forwarding helpers read their defaults from the same enums
using `uv run`; their documented environment overrides still take precedence.

This is a trusted-LAN interface without login. Do not forward its ports through the
router. Local certificates and hostname discovery do not require public DNS.
For local Mac development, use the dashboard on localhost and the
[Quest USB workflow](hardware/quest.md#mac-usb); the Mac does not need a gateway service.

## Certificates

Caddy issues certificates from a local certificate authority. On Triskel, its
public CA certificate is at
`/opt/humanoid/gateway/data/caddy/pki/authorities/local/root.crt`. Import that certificate into
clients supporting local CA trust; never copy its private key. Deployment preserves
the CA and does not change client trust stores. Clients that have not trusted this
CA may show a certificate warning. The gateway data directory contains private
keys as well as certificates; retain it across deployments and installation moves.
Changing its location requires copying the existing state before starting Caddy
there, so the headset continues to trust the same authority.

## Manage the gateway

Deployment enables `humanoid.service` and `humanoid-gateway.service` at boot and
restarts both on updates while preserving the certificate authority. These are
system services; user lingering is unnecessary. The backend starts with robot
motion stopped. See [Operating the robot](operation.md) for dashboard controls.

SSH into Triskel to inspect or restart the gateway:

```bash
ssh triskel
systemctl status humanoid.service humanoid-gateway.service
sudo systemctl restart humanoid-gateway.service
journalctl -u humanoid-gateway.service -f
```

Caddy and its state live under the deployment directory, which defaults to
`/opt/humanoid` and follows `HUMANOID_ROBOT_DIR` when overridden:

| Content | Path under the deployment directory |
| --- | --- |
| Executable | `bin/caddy` |
| Gateway configuration | `gateway/Caddyfile` |
| Certificates, CA, and private keys | `gateway/data/caddy/` |
| Caddy's saved configuration | `gateway/config/caddy/` |

The systemd unit remains at `/etc/systemd/system/humanoid-gateway.service` and runs
as the deployment user. Its `XDG_DATA_HOME` and `XDG_CONFIG_HOME` select the
[Caddy storage locations](https://caddyserver.com/docs/conventions#file-locations);
the enclosing data and config directories are accessible only to that user and root.
Deployment downloads a pinned, checksum-verified release only when the binary is
missing and preserves existing state. Gateway health alone does not prove that the
backend is ready; check both services. For failed `.local` lookup, see
[Troubleshooting](troubleshooting.md#quest-cannot-resolve-the-hostname).

## See also

- [Documentation index](README.md)
- [Deployment](deployment.md)
- [Networking](networking.md)
- [Operating the robot](operation.md)
- [Gateway configuration](../deploy/Caddyfile)
- [Gateway service template](../deploy/humanoid-gateway.service)
- [Deployment script](../scripts/deploy)
