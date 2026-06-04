import unittest
import math
import numpy as np
from src.uncertainty_networks.spatial import Point, Rot, Frame 

class TestSpatialMath(unittest.TestCase):

    def test_point_initialization_and_properties(self):
        """Verify Point initializes as a [3, 1] vector and properties work."""
        p = Point(1.0, 2.0, 3.0)
        
        # Check internal shape
        self.assertEqual(p.vec.shape, (3, 1))
        
        # Check values
        self.assertAlmostEqual(p.x, 1.0)
        self.assertAlmostEqual(p.y, 2.0)
        self.assertAlmostEqual(p.z, 3.0)

    def test_point_addition(self):
        """Verify p3 = p1 + p2 works correctly."""
        p1 = Point(1.0, 2.0, 3.0)
        p2 = Point(4.0, 5.0, 6.0)
        p3 = p1 + p2
        
        self.assertAlmostEqual(p3.x, 5.0)
        self.assertAlmostEqual(p3.y, 7.0)
        self.assertAlmostEqual(p3.z, 9.0)
        
        # Ensure type errors are raised if adding invalid types
        with self.assertRaises(TypeError):
            _ = p1 + [4.0, 5.0, 6.0]

    def test_rotation_matrix_generation(self):
        """Verify Rot correctly builds primary rotation matrices."""
        # 90 degrees around Z axis
        R_z = Rot(axis='z', angle=math.pi / 2)
        expected_z = np.array([
            [0, -1, 0],
            [1,  0, 0],
            [0,  0, 1]
        ], dtype=float)
        np.testing.assert_allclose(R_z.matrix, expected_z, atol=1e-7)

    def test_rotation_times_point(self):
        """Verify p2 = R * p1 shifts coordinates correctly."""
        # Rotate a point on X axis by 90 degrees around Z axis -> should land on Y axis
        p1 = Point(1.0, 0.0, 0.0)
        R = Rot(axis='z', angle=math.pi / 2)
        p2 = R * p1
        
        self.assertAlmostEqual(p2.x, 0.0)
        self.assertAlmostEqual(p2.y, 1.0)
        self.assertAlmostEqual(p2.z, 0.0)

    def test_rotation_times_rotation(self):
        """Verify R3 = R1 * R2 via matrix multiplication."""
        # 90 deg around X then 90 deg around Y
        R1 = Rot(axis='x', angle=math.pi / 2)
        R2 = Rot(axis='y', angle=math.pi / 2)
        R3 = R1 * R2
        
        expected_matrix = R1.matrix @ R2.matrix
        np.testing.assert_allclose(R3.matrix, expected_matrix, atol=1e-7)

    def test_frame_times_point(self):
            """Verify p2 = F * p1 basic transformation formula (R * p1 + p)."""
            # Frame translated by (0, 10, 0) and rotated 90 deg around Z
            R = Rot(axis='z', angle=math.pi / 2)
            p_trans = Point(0.0, 10.0, 0.0)
            F = Frame(R, p_trans)
            
            p1 = Point(1.0, 0.0, 0.0)
            p2 = F * p1
        
            self.assertIsInstance(p2, Point, "The result of Frame * Point must be a Point object")
            self.assertEqual(p2.vec.shape, (3, 1), "The internal vector must maintain a (3, 1) shape")

            # R * p1 gives (0, 1, 0). Adding p_trans (0, 10, 0) gives (0, 11, 0)
            self.assertAlmostEqual(p2.x, 0.0)
            self.assertAlmostEqual(p2.y, 11.0)
            self.assertAlmostEqual(p2.z, 0.0)

    def test_frame_times_frame(self):
        """Verify compounding frames: F3 = F2 * F1."""
        R1 = Rot(axis='z', angle=math.pi / 2)
        p1 = Point(1.0, 0.0, 0.0)
        F1 = Frame(R1, p1)

        R2 = Rot(axis='x', angle=math.pi / 2)
        p2 = Point(0.0, 5.0, 0.0)
        F2 = Frame(R2, p2)

        F3 = F2 * F1
        
        # Validate composite rotation matrix
        expected_R = R2.matrix @ R1.matrix
        np.testing.assert_allclose(F3.R.matrix, expected_R, atol=1e-7)
        
        # Validate composite translation vector (F2.R * F1.p + F2.p)
        expected_p = (R2 * p1) + p2
        self.assertAlmostEqual(F3.p.x, expected_p.x)
        self.assertAlmostEqual(F3.p.y, expected_p.y)
        self.assertAlmostEqual(F3.p.z, expected_p.z)

if __name__ == '__main__':
    unittest.main()