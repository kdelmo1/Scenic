from metadrive import MetaDriveEnv

env = MetaDriveEnv(config={"use_render": False, "num_scenarios": 1})
env.reset()

vehicle = env.agent
print("max_steering:", vehicle.MAX_STEERING)         # degrees
print("vehicle config:", vehicle.config)              # full config dump
print(dir(vehicle))             # may also be here
print(vehicle.FRONT_WHEELBASE) # prints 1.05234
print(vehicle.REAR_WHEELBASE) # prints 1.4166

env.reset()
vehicle = env.agent

# Step multiple times to let steering actuate fully
for _ in range(100):
    env.step([1.0, 0.0])  # full right steering, no throttle

for i, wheel in enumerate(vehicle.wheels):
    print(f"wheel {i} steering: {wheel.get_steering()}")

# Also try reading steering directly from vehicle
print("vehicle.steering:", vehicle.steering)  # normalized -1 to 1
print("max_steering config:", vehicle.config['max_steering'])

env.reset()
vehicle = env.agent

for i, wheel in enumerate(vehicle.wheels):
    print(f"wheel {i}: is_front_wheel={wheel.is_front_wheel()}, steering={wheel.get_steering()}")
    
env.close()
