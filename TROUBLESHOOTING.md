# Troubleshooting

## LCM Self-Test Failed on macOS

If you encounter an LCM self-test failure on macOS, you may need to configure multicast routing. Run the following commands:

```bash
sudo route -nv delete 224.0.0.0/4
sudo route -nv add -net 224.0.0.0/4 -interface lo0
```

This configures the system to route multicast traffic (224.0.0.0/4) through the loopback interface, which is required for LCM to function properly on macOS.

**References:**

- https://github.com/RobotLocomotion/drake/issues/22322
- https://github.com/lcm-proj/lcm/issues/476
- https://github.com/lcm-proj/lcm/issues/60
