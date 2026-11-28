# Troubleshooting

Start with the symptom below, then follow the linked guide for the underlying setup.

## LCM self-test fails on macOS

If the LCM self-test fails, multicast traffic may need to be routed through loopback:

```bash
sudo route -nv delete 224.0.0.0/4
sudo route -nv add -net 224.0.0.0/4 -interface lo0
```

For background, see the upstream reports in
[Drake](https://github.com/RobotLocomotion/drake/issues/22322) and
[LCM](https://github.com/lcm-proj/lcm/issues/476).
See [Networking](networking.md#lcm-transport) for receive-buffer settings.

## Remote console opens but video does not

Check the [WebRTC media path](networking.md#camera-streaming). An HTTP SSH forward
or ADB reverse mapping alone does not carry UDP video. Metric camera snapshots remain available over the API.

## Quest cannot resolve the hostname

`DNS_PROBE_FINISHED_NXDOMAIN` means hostname lookup failed before HTTPS connected.
Confirm the Quest is on the host's Wi-Fi network, close its browser, toggle Wi-Fi
off and on, then open the address in a new tab. This restored `.local` lookup during
headset testing; it does not resolve certificate warnings. If lookup still fails,
check Avahi on the host and whether the network blocks multicast or isolates clients.
See [Local HTTPS gateway](gateway.md) for addresses and service checks.

## Cameras fail to start

For physical cameras, check [SDK and USB setup](hardware/cameras.md#hardware-setup).
For headless simulation, check the [OpenGL backend](simulation.md#headless-rendering).

## See also

- [Documentation index](README.md)
- [Console logs and health](operation.md#console-and-control-inputs)
