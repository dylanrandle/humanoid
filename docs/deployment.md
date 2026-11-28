# Deployment

Install the current checkout on Triskel and operate it through the local console.

## Deploy the stack

Package the current local code and deploy it to `/opt/humanoid`:

```bash
./scripts/deploy
```

Deployment installs locked dependencies, syncs TensorRT engines from `models/` via
`rsync`, and configures the backend and local HTTPS gateway. It enables and restarts
`humanoid.service` and `humanoid-gateway.service`, and enables Avahi for `.local` discovery.
If the engines are Git LFS pointers, run `git lfs pull --include='models/**/*.plan'` first.
The robot needs no source checkout. The stack remains stopped until started from the UI.
Deployment may prompt for `sudo`; use `--dry-run` to build without deploying.
If activation fails, deployment retains its staging directory and prints a command
to finish from an interactive terminal. The package is already installed at that
point; the service update and restart still need to complete. Remove the retained
staging directory after successful activation.
`HUMANOID_ROBOT_HOST` selects the target host; `HUMANOID_ROBOT_DIR` selects the
installation directory.

## Connect and manage the service

Open `https://triskel.local` on the same network. See
[Local HTTPS gateway](gateway.md) for certificates and gateway management.
Alternatively, forward the dashboard port from another local terminal:

```bash
./scripts/forward-ports
```

Open the [dashboard](http://127.0.0.1:8765). Run `./scripts/forward-ports --help` for port
overrides. See [Operating the robot](operation.md) for console controls. Recordings and the dashboard's
`humanoid.log` application log are preserved in `/opt/humanoid/logs`.

SSH into Triskel to manage the service:

```bash
ssh triskel
systemctl status humanoid.service
sudo systemctl restart humanoid.service
journalctl -u humanoid.service -f
```

## Local HTTPS gateway

The [gateway guide](gateway.md) covers the persistent dashboard and
wireless Quest viewer installed by `./scripts/deploy`. The gateway configuration
and service template live in `deploy/`; there is no separate installation
command or bundle. Deployment renders gateway ports from the backend's shared
[port defaults](gateway.md#install-and-configure). Updates preserve the existing Caddy binary and certificate
authority under the installation directory; see the
[gateway layout](gateway.md#manage-the-gateway).
On a Mac, run the dashboard on localhost and use the [Quest USB workflow](hardware/quest.md#mac-usb).

## See also

- [Documentation index](README.md)
- [Operating the robot](operation.md)
- [Networking and remote video](networking.md)
- [NanoSAM](models/nanosam.md) and [camera depth models](models/cdm.md)
- [Deployment script](../scripts/deploy)
