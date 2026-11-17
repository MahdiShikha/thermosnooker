"""Correctness checks for the selectable collision searches.

Run from the repository root with::

    PYTHONPATH=src python -m unittest discover -s tests -v

These tests compare physical results rather than imposing machine-dependent
wall-clock performance thresholds.
"""

import unittest
from unittest.mock import patch

import numpy as np

from thermosnooker.balls import Ball, Container
from thermosnooker.simulations import MultiBallSimulation


class CollisionSearchTests(unittest.TestCase):
    """The broad phase must preserve the exact collision calculation."""

    def assert_same_state(self, reference, accelerated):
        """Compare the full physical state, including the recoiling wall."""
        self.assertAlmostEqual(reference.time(), accelerated.time(), places=10)
        for first, second in zip(
            [reference.container(), *reference.balls()],
            [accelerated.container(), *accelerated.balls()],
        ):
            np.testing.assert_allclose(first.pos(), second.pos(), rtol=1e-11, atol=1e-10)
            np.testing.assert_allclose(first.vel(), second.vel(), rtol=1e-11, atol=1e-10)
        np.testing.assert_allclose(
            reference.momentum(), accelerated.momentum(), rtol=1e-11, atol=1e-10
        )
        self.assertAlmostEqual(
            reference.kinetic_energy(), accelerated.kinetic_energy(), places=8
        )
        self.assertAlmostEqual(reference.pressure(), accelerated.pressure(), places=10)

    @staticmethod
    def custom_simulation(method, specifications, container_radius=100):
        """Use explicit initial states to exercise geometry absent from rings."""
        simulation = MultiBallSimulation(
            c_radius=container_radius, collision_search=method, seed=42
        )
        simulation._balls = [
            Ball(pos=position, vel=velocity, radius=radius, mass=mass)
            for position, velocity, radius, mass in specifications
        ]
        return simulation

    def test_seeded_trajectories_and_conservation_match(self):
        reference = MultiBallSimulation(collision_search="brute_force", seed=123)
        accelerated = MultiBallSimulation(collision_search="quadtree", seed=123)
        initial_energy = reference.kinetic_energy()
        initial_momentum = reference.momentum()
        self.assert_same_state(reference, accelerated)

        for event in range(80):
            with self.subTest(event=event):
                self.assertAlmostEqual(
                    reference.next_collision(), accelerated.next_collision(), places=11
                )
                self.assert_same_state(reference, accelerated)
                self.assertAlmostEqual(
                    accelerated.kinetic_energy(), initial_energy, places=8
                )
                np.testing.assert_allclose(
                    accelerated.momentum(), initial_momentum, rtol=1e-11, atol=1e-10
                )

    def test_distant_fast_balls_are_not_pruned_as_current_non_neighbours(self):
        specifications = [
            ([-30, 0], [100, 0], 1, 1),
            ([30, 0], [-100, 0], 1, 1),
        ]
        specifications.extend(
            ([x, y], [0, 0], 1, 1)
            for x in (-60, -40, -20, 0, 20, 40, 60)
            for y in (-40, -20, 20, 40)
        )
        reference = self.custom_simulation("brute_force", specifications)
        accelerated = self.custom_simulation("quadtree", specifications)

        self.assertAlmostEqual(reference.next_collision(), 0.29)
        self.assertAlmostEqual(accelerated.next_collision(), 0.29)
        self.assert_same_state(reference, accelerated)
        np.testing.assert_allclose(accelerated.balls()[0].vel(), [-100, 0])
        np.testing.assert_allclose(accelerated.balls()[1].vel(), [100, 0])

    def test_random_mixed_radius_trajectories_match_with_moving_container(self):
        rng = np.random.default_rng(481)
        specifications = []
        while len(specifications) < 48:
            position = rng.uniform(-20, 20, size=2)
            radius = rng.uniform(0.2, 0.8)
            if any(
                np.linalg.norm(position - other_position) <= radius + other_radius
                for other_position, _, other_radius, _ in specifications
            ):
                continue
            specifications.append(
                (position, rng.uniform(-30, 30, size=2), radius, rng.uniform(0.5, 3))
            )
        reference = self.custom_simulation("brute_force", specifications, 40)
        accelerated = self.custom_simulation("quadtree", specifications, 40)
        for simulation in (reference, accelerated):
            simulation.container().set_vel([0.8, -0.4])

        for event in range(40):
            with self.subTest(event=event):
                self.assertAlmostEqual(
                    reference.next_collision(), accelerated.next_collision(), places=10
                )
                self.assert_same_state(reference, accelerated)

    def test_moving_wall_collision_is_retained(self):
        specifications = [([0, 0], [0, 0], 1, 1)]
        reference = self.custom_simulation("brute_force", specifications, 10)
        accelerated = self.custom_simulation("quadtree", specifications, 10)
        for simulation in (reference, accelerated):
            simulation.container().set_vel([2, 0])

        self.assertAlmostEqual(reference.next_collision(), 4.5)
        self.assertAlmostEqual(accelerated.next_collision(), 4.5)
        self.assert_same_state(reference, accelerated)

    def test_equal_event_times_do_not_compare_ball_objects(self):
        specifications = [([-5, 0], [-1, 0], 1, 1), ([5, 0], [1, 0], 1, 1)]
        reference = self.custom_simulation("brute_force", specifications, 10)
        accelerated = self.custom_simulation("quadtree", specifications, 10)

        self.assertAlmostEqual(reference.next_collision(), 4)
        self.assertAlmostEqual(accelerated.next_collision(), 4)
        self.assert_same_state(reference, accelerated)

    def test_quadtree_prunes_exact_pair_checks_for_short_horizon(self):
        specifications = [([98.5, 0], [10, 0], 1, 1)]
        specifications.extend(
            ([x, y], [0, 0], 1, 1)
            for x in (-30, -20, -10, 0, 10, 20, 30)
            for y in (-30, -20, -10, 0, 10, 20, 30)
        )
        original = Ball.time_to_collision
        counts = {}
        states = {}
        for method in ("brute_force", "quadtree"):
            pair_checks = 0

            def count_pair_checks(first, second):
                nonlocal pair_checks
                if not isinstance(first, Container) and not isinstance(second, Container):
                    pair_checks += 1
                return original(first, second)

            simulation = self.custom_simulation(method, specifications)
            with patch.object(Ball, "time_to_collision", count_pair_checks):
                self.assertAlmostEqual(simulation.next_collision(), 0.05)
            counts[method] = pair_checks
            states[method] = simulation

        self.assertGreater(counts["brute_force"], 0)
        self.assertLess(counts["quadtree"], counts["brute_force"])
        self.assert_same_state(states["brute_force"], states["quadtree"])

    def test_stationary_system_reports_no_future_collision(self):
        for method in ("brute_force", "quadtree"):
            with self.subTest(method=method):
                simulation = MultiBallSimulation(b_speed=0, collision_search=method)
                with self.assertRaises(ValueError):
                    simulation.next_collision()
                self.assertEqual(simulation.time(), 0)


class ConfigurationAndTimingTests(unittest.TestCase):
    """Check the public configuration and backwards-compatible return value."""

    def test_exact_ball_count_and_reproducible_seed(self):
        for count in (1, 12, 37):
            with self.subTest(count=count):
                first = MultiBallSimulation(num_balls=count, seed=79)
                second = MultiBallSimulation(num_balls=count, seed=79)
                self.assertEqual(len(first.balls()), count)
                self.assertEqual(len(second.balls()), count)
                np.testing.assert_array_equal(
                    [ball.pos() for ball in first.balls()],
                    [ball.pos() for ball in second.balls()],
                )
                np.testing.assert_array_equal(
                    [ball.vel() for ball in first.balls()],
                    [ball.vel() for ball in second.balls()],
                )

    def test_invalid_search_method_is_rejected(self):
        with self.assertRaises(ValueError):
            MultiBallSimulation(collision_search="octree")

    def test_invalid_ball_counts_are_rejected(self):
        for count in (0, -1, 1.5, True, 38):
            with self.subTest(count=count):
                with self.assertRaises((TypeError, ValueError)):
                    MultiBallSimulation(num_balls=count)

    def test_run_returns_elapsed_seconds_only_when_requested(self):
        simulation = MultiBallSimulation(num_balls=1, seed=5)
        self.assertIsNone(simulation.run(1))
        elapsed = simulation.run(2, output_time_taken=True)
        self.assertIsInstance(elapsed, float)
        self.assertGreaterEqual(elapsed, 0)
        self.assertGreater(simulation.time(), 0)


if __name__ == "__main__":
    unittest.main()
