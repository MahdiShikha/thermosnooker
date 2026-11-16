"""Conservative quadtree broad phase for event-driven collision searches.

Current neighbours alone are insufficient: a distant, fast ball may be the next
one to collide. Each node therefore bounds both positions and velocities, so a
query can exclude it only if no contact is possible before the best event time.
The tree is rebuilt at each event. Its search is still quadratic in the worst
case, even though well-separated systems can require far fewer exact checks.
"""

import math


class _Node:
    """A quadrant with tight bounds on the particles it contains."""

    __slots__ = (
        "xmin", "xmax", "ymin", "ymax", "vxmin", "vxmax", "vymin",
        "vymax", "radius", "max_index", "children", "records",
    )

    def __init__(self, records, capacity, depth):
        first = records[0]
        _, self.xmin, self.ymin, self.vxmin, self.vymin, self.radius = first
        self.xmax, self.ymax = self.xmin, self.ymin
        self.vxmax, self.vymax = self.vxmin, self.vymin
        self.max_index = first[0]
        for index, x, y, vx, vy, radius in records[1:]:
            self.xmin = min(self.xmin, x)
            self.xmax = max(self.xmax, x)
            self.ymin = min(self.ymin, y)
            self.ymax = max(self.ymax, y)
            self.vxmin = min(self.vxmin, vx)
            self.vxmax = max(self.vxmax, vx)
            self.vymin = min(self.vymin, vy)
            self.vymax = max(self.vymax, vy)
            self.radius = max(self.radius, radius)
            self.max_index = max(self.max_index, index)

        self.children = ()
        self.records = records
        if len(records) <= capacity or depth == 0:
            return
        if self.xmin == self.xmax and self.ymin == self.ymax:
            return

        mid_x = (self.xmin + self.xmax) / 2
        mid_y = (self.ymin + self.ymax) / 2
        quadrants = [[], [], [], []]
        for record in records:
            quadrant = (record[1] >= mid_x) + 2 * (record[2] >= mid_y)
            quadrants[quadrant].append(record)
        # Rounding can prevent a useful split for almost coincident centres.
        if any(len(group) == len(records) for group in quadrants):
            return
        self.children = tuple(
            _Node(group, capacity, depth - 1) for group in quadrants if group
        )
        self.records = ()

    def can_collide(self, particle, horizon):
        """Could any contained particle contact this one in [0, horizon]?"""
        _, x, y, vx, vy, radius = particle
        reach = radius + self.radius
        # Bounds in the query particle's moving frame. At any future contact,
        # both relative coordinates must lie within +/- the radius sum.
        xmin = self.xmin - x + min(0.0, (self.vxmin - vx) * horizon)
        xmax = self.xmax - x + max(0.0, (self.vxmax - vx) * horizon)
        ymin = self.ymin - y + min(0.0, (self.vymin - vy) * horizon)
        ymax = self.ymax - y + max(0.0, (self.vymax - vy) * horizon)
        # A small conservative margin protects contact at the time boundary
        # from floating-point rounding in the bounding calculations.
        tolerance = 1e-12 * max(
            1.0, reach, abs(xmin), abs(xmax), abs(ymin), abs(ymax)
        )
        reach += tolerance
        return xmin <= reach and xmax >= -reach and ymin <= reach and ymax >= -reach


class CollisionQuadtree:
    """Find exact pair events after pruning impossible quadrants.

    Events use (time, kind, first_index, second_index), with kind 0 for walls
    and 1 for ball pairs. This provides deterministic ties without comparing
    Ball objects. A finite wall event supplies the initial search horizon.
    """

    def __init__(self, balls, leaf_capacity=8, max_depth=20):
        self._balls = balls
        self._records = [
            (i, float(b.pos()[0]), float(b.pos()[1]),
             float(b.vel()[0]), float(b.vel()[1]), b.radius())
            for i, b in enumerate(balls)
        ]
        self._root = (
            _Node(self._records, leaf_capacity, max_depth) if self._records else None
        )

    def next_event(self, best_event):
        """Return the earlier of best_event and all possible ball-pair events.

        The caller handles absent finite wall events using brute force. Infinite
        horizons would make velocity bounds such as 0 * infinity undefined.
        """
        if best_event is None or not math.isfinite(best_event[0]):
            raise ValueError("A finite collision horizon is required.")
        if self._root is None:
            return best_event
        for particle in self._records:
            i = particle[0]
            stack = [self._root]
            while stack:
                node = stack.pop()
                if node.max_index <= i or not node.can_collide(particle, best_event[0]):
                    continue
                if node.children:
                    # The matching quadrant is visited first, often tightening
                    # the horizon before more distant quadrants are searched.
                    x, y = particle[1:3]
                    children = sorted(
                        node.children,
                        key=lambda child: (
                            max(child.xmin - x, 0.0, x - child.xmax) ** 2
                            + max(child.ymin - y, 0.0, y - child.ymax) ** 2
                        ),
                        reverse=True,
                    )
                    stack.extend(children)
                else:
                    for other in node.records:
                        j = other[0]
                        if j <= i:
                            continue
                        t = self._balls[i].time_to_collision(self._balls[j])
                        if t is not None:
                            event = (t, 1, i, j)
                            if event < best_event:
                                best_event = event
        return best_event


def next_quadtree_collision(balls, container, wall_events):
    """Return (time, first_object, second_object) for the original event loop.

    Reuse its wall predictions to bound the tree search, keeping the simulation's
    movement and measurement code shared with the original all-pairs method.
    """
    wall = min(wall_events, key=lambda event: event[0], default=None)
    best_event = None if wall is None else (wall[0], 0, balls.index(wall[2]), -1)
    if best_event is not None and math.isfinite(best_event[0]):
        best_event = CollisionQuadtree(balls).next_event(best_event)
    else:
        # Without a finite horizon, exact all-pairs checks are the safe fallback.
        for i, ball in enumerate(balls):
            for j in range(i + 1, len(balls)):
                t = ball.time_to_collision(balls[j])
                if t is not None:
                    event = (t, 1, i, j)
                    if best_event is None or event < best_event:
                        best_event = event
    if best_event is None:
        raise ValueError("No future collisions are available.")
    t, kind, i, j = best_event
    if kind == 0:
        return t, container, balls[i]
    return t, balls[i], balls[j]
