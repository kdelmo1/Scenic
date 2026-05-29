"""Low-level controllers useful for vehicles.

The Lateral/Longitudinal PID controllers are adapted from `CARLA`_'s PID controllers,
which are licensed under the following terms:

    Copyright (c) 2018-2020 CVC.

    This work is licensed under the terms of the MIT license.
    For a copy, see <https://opensource.org/licenses/MIT>.


.. _CARLA: https://carla.org/
"""

from collections import deque
import math

import numpy as np
from shapely.geometry import LineString, MultiPoint, Point as ShapelyPoint

from scenic.core.regions import CircularRegion, PolylineRegion
from scenic.domains.driving.actions import *
from scenic.domains.driving.roads import ManeuverType


class PIDLongitudinalController:
    """Longitudinal control using a PID to reach a target speed.

    Arguments:
        K_P: Proportional gain
        K_D: Derivative gain
        K_I: Integral gain
        dt: time step
    """

    def __init__(self, K_P=0.5, K_D=0.1, K_I=0.2, dt=0.1):
        self._k_p = K_P
        self._k_d = K_D
        self._k_i = K_I
        self._dt = dt
        self._error_buffer = deque(maxlen=10)

    def run_step(self, speed_error):
        """Estimate the throttle/brake of the vehicle based on the PID equations.

        Arguments:
            speed_error: target speed minus current speed

        Returns:
            a signal between -1 and 1, with negative values indicating braking.
        """
        error = speed_error
        self._error_buffer.append(error)

        if len(self._error_buffer) >= 2:
            _de = (self._error_buffer[-1] - self._error_buffer[-2]) / self._dt
            _ie = sum(self._error_buffer) * self._dt
        else:
            _de = 0.0
            _ie = 0.0

        return np.clip(
            (self._k_p * error) + (self._k_d * _de) + (self._k_i * _ie), -1.0, 1.0
        )


class PIDLateralController:
    """Lateral control using a PID to track a trajectory.

    Arguments:
        K_P: Proportional gain
        K_D: Derivative gain
        K_I: Integral gain
        dt: time step
    """

    def __init__(self, K_P=0.3, K_D=0.2, K_I=0, dt=0.1):
        self.Kp = K_P
        self.Kd = K_D
        self.Ki = K_I
        self.PTerm = 0
        self.ITerm = 0
        self.DTerm = 0
        self.dt = dt
        self.last_error = 0
        self.windup_guard = 20.0
        self.output = 0

    def run_step(self, input_trajectory, ego, opposite_traffic):
        """Estimate the steering angle of the vehicle based on the PID equations.

        Arguments:
            cte: cross-track error (distance to right of desired trajectory)

        Returns:
            a signal between -1 and 1, with -1 meaning maximum steering to the left.
        """

        nearest_line_points = input_trajectory.nearestSegmentTo(ego.position)
        nearest_line_segment = PolylineRegion(nearest_line_points)
        cte = nearest_line_segment.signedDistanceTo(ego.position)

        if opposite_traffic:
            cte = -1 * cte

        error = cte
        delta_error = error - self.last_error
        self.PTerm = self.Kp * error
        self.ITerm += error * self.dt

        if self.ITerm < -self.windup_guard:
            self.ITerm = -self.windup_guard
        elif self.ITerm > self.windup_guard:
            self.ITerm = self.windup_guard

        self.DTerm = delta_error / self.dt

        # Remember last error for next calculation
        self.last_error = error

        self.output = self.PTerm + (self.Ki * self.ITerm) + (self.Kd * self.DTerm)

        return np.clip(self.output, -1, 1)


class PurePursuitLateralController:
    """
    Pure Pursuit lateral controller.

    The lookahead distance is computed dynamically as:
        ld = clip(K_dd * speed, min_ld, max_ld)

    Arguments:
        cl:     car length (metres)
        ld:     initial / fallback lookahead distance (metres)
        dt:     time step (seconds)
        clwbr:  car-length-to-wheelbase ratio  (wb = cl * clwbr)
        K_dd:   speed-to-lookahead gain
        min_ld: minimum lookahead distance (metres)
        max_ld: maximum lookahead distance (metres)
    """

    def __init__(self, cl, ld=7.0, dt=0.1, clwbr=0.72,
                 K_dd=0.5, min_ld=3.0, max_ld=15.0):
        self.dt     = dt
        self.wb     = cl * clwbr          # wheelbase
        self.clwbr  = clwbr
        self.ld     = ld                  # current lookahead (updated each step)
        self.K_dd   = K_dd
        self.min_ld = min_ld
        self.max_ld = max_ld
        self.past_cte = 0.0
        # Maximum steering angle (radians) — used for normalisation
        self.max_steering_angle = np.radians(35)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _split_trajectory(self, line, coords, distance):
        """
        Split *coords* into (behind, ahead) at the point on *line* that is
        *distance* along it.  Returns a list of two LineStrings, or [] on
        failure.
        """
        for j, p in enumerate(coords):
            pd = line.project(ShapelyPoint(p[0], p[1]))

            if pd == distance:
                front = coords[j:]
                back  = coords[:j + 1]
                if len(front) < 2 or len(back) < 2:
                    break
                return [LineString(back), LineString(front)]

            if pd > distance:
                cp    = line.interpolate(distance)
                front = [(cp.x, cp.y, 0)] + coords[j:]
                back  = coords[:j] + [(cp.x, cp.y, 0)]
                if len(front) < 2 or len(back) < 2:
                    break
                return [LineString(back), LineString(front)]

        return []

    def _fallback_steer(self):
        """Return a steering signal based on the last known CTE."""
        angle = np.arctan((2 * self.wb * math.sin(self.past_cte)) / self.ld)
        return float(np.clip(angle / self.max_steering_angle, -1.0, 1.0))

    # ------------------------------------------------------------------
    # main step
    # ------------------------------------------------------------------

    def run_step(self, input_trajectory, ego, opposite_traffic, speed=None):
        """
        Estimate the steering angle using the Pure Pursuit algorithm.

        Arguments:
            input_trajectory: PolylineRegion representing the path to follow
            ego:              the agent object (needs .position and .heading)
            opposite_traffic: if True, flip the sign of the lateral error
            speed:            current speed in m/s; used to set lookahead distance
                              dynamically.  Pass None to keep the last ld value.

        Returns:
            A signal in [-1, 1]; -1 = maximum left, +1 = maximum right.
        """

        # ── 1. Update lookahead distance from speed ────────────────────────
        if speed is not None:
            self.ld = float(np.clip(self.K_dd * speed, self.min_ld, self.max_ld))

        # ── 2. Project ego onto trajectory, collect coords ─────────────────
        line = input_trajectory.lineString

        ego_pt   = ShapelyPoint(ego.position.coordinates[0],
                                ego.position.coordinates[1])
        distance = line.project(ego_pt)

        coords = []
        try:
            coords = list(line.coords)                # normal LineString
        except NotImplementedError:
            for geom in line.geoms:                   # MultiLineString
                coords.extend(list(geom.coords))

        # ── 3. Split trajectory into behind / ahead ────────────────────────
        split = self._split_trajectory(line, coords, distance)
        if len(split) < 2:
            return self._fallback_steer()

        ahead_line = split[1]

        # ── 4. Find intersection of lookahead circle with ahead path ────────
        circle_boundary = CircularRegion(
            ego.position, self.ld, resolution=64
        ).boundary.lineString

        intersection = circle_boundary.intersection(ahead_line)

        if isinstance(intersection, ShapelyPoint):
            candidates = [intersection]
        elif isinstance(intersection, MultiPoint):
            candidates = list(intersection.geoms)
        else:
            candidates = []

        # ── 5. Choose the lookahead point ──────────────────────────────────
        if not candidates:
            # No intersection found — use past CTE (self-correcting)
            cte = self.past_cte
        else:
            # Pick the candidate *furthest along the ahead path* (not the
            # closest one).  The original code used candidates[0] after
            # sorting ascending by arc-length, which gave the nearest
            # intersection — i.e. a point barely in front of the car.
            # That caused over-steering on curves.  Using the furthest
            # candidate within ld gives smoother, more anticipatory steering.
            candidates.sort(
                key=lambda pt: ahead_line.project(ShapelyPoint(pt.x, pt.y))
            )
            lookahead_pt = candidates[-1]   # ← furthest within circle

            # Angle from ego to lookahead point in world frame
            theta = math.atan2(
                lookahead_pt.y - ego.position.coordinates[1],
                lookahead_pt.x - ego.position.coordinates[0],
            )
            # cte expressed as heading error (rad)
            cte = (ego.heading + math.pi / 2) - theta
            self.past_cte = cte

        if opposite_traffic:
            cte = -cte

        # ── 6. Pure Pursuit formula → steering signal ──────────────────────
        steering_angle = np.arctan((2.0 * self.wb * math.sin(cte)) / self.ld)
        return float(np.clip(steering_angle / self.max_steering_angle, -1.0, 1.0))