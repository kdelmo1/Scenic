'''
To run this file using the MetaDrive simulator:
    scenic examples/driving/arya.scenic --2d --model scenic.simulators.metadrive.model --simulate

To run this file using the Carla simulator:
    scenic examples/driving/arya.scenic --2d --model scenic.simulators.carla.model --simulate


TODO:
    Make some kinda waypoints that the car can follow 
    make the car follow the waypoints using pure pursuit.
'''

param map = localPath('../../assets/maps/CARLA/Town05.xodr')
param carla_map = 'Town05'
param time_step = 1.0/10

model scenic.domains.driving.model

behavior DriveWithPurePursuit():
    lon_ctrl, lat_ctrl = simulation().getPurePursuitControllers(self)
    do FollowLaneBehavior(lon_controller=lon_ctrl, lat_controller=lat_ctrl, target_speed=15)

# lane = Uniform(*network.lanes)
lane = Uniform(*filter(lambda l: len(l.centerline.points) > 4, network.lanes))

start_pt = new OrientedPoint on lane.centerline

ego = new Car at start_pt,
    with behavior DriveWithPurePursuit()

monitor MyMonitor():
    while True:
        current_lane = ego.lane
        if current_lane is not None:
            cte = current_lane.centerline.signedDistanceTo(ego.position)
            on_road = ego.position in network.roadRegion
            print(f"speed={ego.speed:.2f} cte={cte:.3f} on_road={on_road} lane={current_lane}")
        else:
            print(f"speed={ego.speed:.2f} cte=N/A on_road=N/A lane=None")

        assert ego._lane is not None
        wait

require monitor MyMonitor()

require (distance to intersection) < 20

terminate after 60 seconds

# new scenario at fixed point and add a break point to see what's going on when it reaches the intersection (where cte starts to diverge)
# can compare target_speeds (7 vs 15)
    # print out the steering angle that is used 
    # if same, suggests that fixed steering angle does not produce the same trajectory

# run in newtonian (zoom out)
# ask eric abt speed dependency