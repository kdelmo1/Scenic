param map = localPath('../../assets/maps/CARLA/Town05.xodr')
param carla_map = 'Town05'
param time_step = 1.0/10

param lookahead_gain = 0.5
param alpha = 0.1
param max_steering_deg = 40.0

model scenic.domains.driving.model

behavior DriveWithPurePursuit():
    lon_ctrl, lat_ctrl = simulation().getPurePursuitControllers(self)
    do FollowLaneBehavior(lon_controller=lon_ctrl, lat_controller=lat_ctrl, target_speed=15)

lane = Uniform(*filter(lambda l: len(l.centerline.points) > 4, network.lanes))
start_pt = new OrientedPoint on lane.centerline

ego = new Car at start_pt,
    with behavior DriveWithPurePursuit()

record (distance from ego to ego.lane.centerline 
        if ego.lane is not None else 999) as cte

require (distance to intersection) < 20
terminate after 60 seconds