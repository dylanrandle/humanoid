# Deployment

Install the current checkout on Triskel and operate it through the local console.

## Deploy the stack

Package the current local code and deploy it to `/opt/humanoid`:

```bash
./scripts/deploy
```

Deployment installs locked dependencies, syncs TensorRT engines from `models/` via
`rsync`, enables `humanoid.service`, and restarts the console. If the engines are
Git LFS pointers, run `git lfs pull --include='models/**/*.plan'` first.
The robot needs no source checkout. The stack remains stopped until started from the UI.
Deployment may prompt for `sudo`; use `--dry-run` to build without deploying.
`HUMANOID_ROBOT_HOST` selects the target host; `HUMANOID_ROBOT_DIR` selects the
installation directory.

## Connect and manage the service

Forward the dashboard port from another local terminal:

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

## Wireless Oculus

Enable Developer Mode and USB debugging, connect the Quest over USB once, and run:

```bash
adb tcpip 5555
adb shell ip route
```

Use the address shown after `src`, then SSH into Triskel and connect to it (replace the
example IP):

```bash
ssh triskel
adb connect 192.168.1.42:5555
adb devices -l
sudo systemctl edit humanoid.service
```

Add this override, then restart the service:

```ini
[Service]
Environment=HUMANOID_OCULUS_IP=192.168.1.42
```

```bash
sudo systemctl restart humanoid.service
```

Oculus uses USB discovery by default. After a headset reboot, you may need to
re-enable wireless ADB. See [Operating the robot](operation.md#console-and-control-inputs)
for clutch and tracking behavior.

## See also

- [Documentation index](README.md)
- [Operating the robot](operation.md)
- [Networking and remote video](networking.md)
- [NanoSAM](models/nanosam.md) and [camera depth models](models/cdm.md)
- [Deployment script](../scripts/deploy)
