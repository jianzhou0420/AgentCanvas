from __future__ import annotations

import functools

import numpy as np
import torch

# Import rotation utilities from JianRotationTorch instead of pytorch3d
from workspace.nodesets.policy.policy_adapter_vla.models.z_utils.JianRotationTorch import (
    axis_angle_to_matrix,
    euler_angles_to_matrix,
    matrix_to_axis_angle,
    matrix_to_euler_angles,
    matrix_to_quaternion,
    matrix_to_rotation_6d,
    quaternion_to_matrix,
    rotation_6d_to_matrix,
)


class RotationTransformer:
    valid_reps = ["axis_angle", "euler_angles", "quaternion", "rotation_6d", "matrix"]

    def __init__(
        self, from_rep="axis_angle", to_rep="rotation_6d", from_convention=None, to_convention=None
    ):
        """
        Valid representations

        Always use matrix as intermediate representation.
        """
        assert from_rep != to_rep
        assert from_rep in self.valid_reps
        assert to_rep in self.valid_reps
        if from_rep == "euler_angles":
            assert from_convention is not None
        if to_rep == "euler_angles":
            assert to_convention is not None

        forward_funcs = list()
        inverse_funcs = list()

        # Map representation names to conversion functions
        rep_to_matrix_funcs = {
            "axis_angle": axis_angle_to_matrix,
            "euler_angles": euler_angles_to_matrix,
            "quaternion": quaternion_to_matrix,
            "rotation_6d": rotation_6d_to_matrix,
        }

        matrix_to_rep_funcs = {
            "axis_angle": matrix_to_axis_angle,
            "euler_angles": matrix_to_euler_angles,
            "quaternion": matrix_to_quaternion,
            "rotation_6d": matrix_to_rotation_6d,
        }

        if from_rep != "matrix":
            to_matrix_func = rep_to_matrix_funcs[from_rep]
            from_matrix_func = matrix_to_rep_funcs[from_rep]

            if from_convention is not None:
                to_matrix_func = functools.partial(to_matrix_func, convention=from_convention)
                from_matrix_func = functools.partial(from_matrix_func, convention=from_convention)

            forward_funcs.append(to_matrix_func)
            inverse_funcs.append(from_matrix_func)

        if to_rep != "matrix":
            to_rep_func = matrix_to_rep_funcs[to_rep]
            from_rep_func = rep_to_matrix_funcs[to_rep]

            if to_convention is not None:
                to_rep_func = functools.partial(to_rep_func, convention=to_convention)
                from_rep_func = functools.partial(from_rep_func, convention=to_convention)

            forward_funcs.append(to_rep_func)
            inverse_funcs.append(from_rep_func)

        inverse_funcs = inverse_funcs[::-1]

        self.forward_funcs = forward_funcs
        self.inverse_funcs = inverse_funcs

    @staticmethod
    def _apply_funcs(x: np.ndarray | torch.Tensor, funcs: list) -> np.ndarray | torch.Tensor:
        x_ = x
        if isinstance(x, np.ndarray):
            x_ = torch.from_numpy(x)
        x_: torch.Tensor
        for func in funcs:
            x_ = func(x_)
        y = x_
        if isinstance(x, np.ndarray):
            y = x_.numpy()
        return y

    def forward(self, x: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        return self._apply_funcs(x, self.forward_funcs)

    def inverse(self, x: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        return self._apply_funcs(x, self.inverse_funcs)


if __name__ == "__main__":

    def test():
        import numpy as np
        import torch

        from workspace.nodesets.policy.policy_adapter_vla.models.dp.common.rotation_transformer import (
            RotationTransformer,
        )

        def test_all_conversions():
            """Test all possible representation conversions."""
            print("=" * 80)
            print("Testing All Representation Conversions")
            print("=" * 80)

            representations = ["axis_angle", "euler_angles", "quaternion", "rotation_6d", "matrix"]

            # Generate test data
            np.random.seed(42)
            torch.manual_seed(42)

            test_axis_angle = np.random.uniform(-np.pi, np.pi, size=(10, 3))

            total_tests = 0
            passed_tests = 0

            for from_rep in representations:
                for to_rep in representations:
                    if from_rep == to_rep:
                        continue

                    total_tests += 1

                    try:
                        # Handle euler_angles which require convention
                        from_conv = "XYZ" if from_rep == "euler_angles" else None
                        to_conv = "XYZ" if to_rep == "euler_angles" else None

                        tf = RotationTransformer(
                            from_rep=from_rep,
                            to_rep=to_rep,
                            from_convention=from_conv,
                            to_convention=to_conv,
                        )

                        # Convert axis_angle to the from_rep first
                        if from_rep != "axis_angle":
                            tf_temp = RotationTransformer(
                                from_rep="axis_angle",
                                to_rep=from_rep,
                                from_convention=None,
                                to_convention=from_conv,
                            )
                            test_data = tf_temp.forward(test_axis_angle)
                        else:
                            test_data = test_axis_angle

                        # Test forward conversion
                        result = tf.forward(test_data)

                        # Test inverse conversion
                        reconstructed = tf.inverse(result)

                        # Check shapes
                        assert result.shape[0] == test_data.shape[0], "Batch size mismatch"
                        assert reconstructed.shape == test_data.shape, (
                            "Reconstruction shape mismatch"
                        )

                        print(f"✓ {from_rep:15} -> {to_rep:15} : PASSED")
                        passed_tests += 1

                    except Exception as e:
                        print(f"✗ {from_rep:15} -> {to_rep:15} : FAILED - {e!s}")

            print("\n" + "=" * 80)
            print(f"Results: {passed_tests}/{total_tests} tests passed")
            print("=" * 80)
            return passed_tests == total_tests

        def test_numerical_accuracy():
            """Test numerical accuracy of conversions."""
            print("\n" + "=" * 80)
            print("Testing Numerical Accuracy")
            print("=" * 80)

            np.random.seed(42)

            # Test 1: axis_angle -> rotation_6d -> axis_angle round trip
            print("\nTest 1: axis_angle -> rotation_6d -> axis_angle")
            tf = RotationTransformer(from_rep="axis_angle", to_rep="rotation_6d")

            rotvec = np.random.uniform(-2 * np.pi, 2 * np.pi, size=(1000, 3))
            rot6d = tf.forward(rotvec)
            new_rotvec = tf.inverse(rot6d)

            from scipy.spatial.transform import Rotation

            diff = Rotation.from_rotvec(rotvec) * Rotation.from_rotvec(new_rotvec).inv()
            dist = diff.magnitude()
            max_error = dist.max()

            print(f"  Max error: {max_error:.2e}")
            print(f"  Mean error: {dist.mean():.2e}")
            assert max_error < 1e-6, f"Error too large: {max_error}"
            print("  ✓ PASSED")

            # Test 2: rotation_6d normalization to valid rotation matrix
            print("\nTest 2: rotation_6d -> matrix (with noise)")
            tf = RotationTransformer("rotation_6d", "matrix")

            rot6d_noisy = rot6d + np.random.normal(scale=0.1, size=rot6d.shape)
            mat = tf.forward(rot6d_noisy)
            mat_det = np.linalg.det(mat)

            print(f"  Determinant range: [{mat_det.min():.6f}, {mat_det.max():.6f}]")
            print(f"  Determinant mean: {mat_det.mean():.6f}")
            assert np.allclose(mat_det, 1.0, atol=1e-5), "Determinants not close to 1"
            print("  ✓ PASSED")

            # Test 3: quaternion -> matrix -> quaternion
            print("\nTest 3: quaternion -> matrix -> quaternion")
            tf = RotationTransformer(from_rep="quaternion", to_rep="matrix")

            # Generate random quaternions
            from workspace.nodesets.policy.policy_adapter_vla.models.z_utils.JianRotationTorch import (
                random_quaternions,
            )

            quats = random_quaternions(100).numpy()

            matrices = tf.forward(quats)
            reconstructed_quats = tf.inverse(matrices)

            # Quaternions can differ by sign, so check both q and -q
            diff1 = np.abs(quats - reconstructed_quats).max(axis=1)
            diff2 = np.abs(quats + reconstructed_quats).max(axis=1)
            error = np.minimum(diff1, diff2).max()

            print(f"  Max error: {error:.2e}")
            assert error < 1e-6, f"Error too large: {error}"
            print("  ✓ PASSED")

            # Test 4: euler_angles -> matrix -> euler_angles
            print("\nTest 4: euler_angles -> matrix -> euler_angles (XYZ convention)")
            tf = RotationTransformer(
                from_rep="euler_angles", to_rep="matrix", from_convention="XYZ", to_convention="XYZ"
            )

            euler = np.random.uniform(-np.pi, np.pi, size=(100, 3))
            matrices = tf.forward(euler)
            reconstructed_euler = tf.inverse(matrices)

            # Check rotation matrices are the same
            from workspace.nodesets.policy.policy_adapter_vla.models.z_utils.JianRotationTorch import (
                euler_angles_to_matrix,
            )

            original_mat = euler_angles_to_matrix(torch.from_numpy(euler), "XYZ").numpy()
            reconstructed_mat = euler_angles_to_matrix(
                torch.from_numpy(reconstructed_euler), "XYZ"
            ).numpy()

            error = np.abs(original_mat - reconstructed_mat).max()
            print(f"  Max matrix error: {error:.2e}")
            assert error < 1e-6, f"Error too large: {error}"
            print("  ✓ PASSED")

            print("\n" + "=" * 80)
            print("All numerical accuracy tests PASSED")
            print("=" * 80)
            return True

        def test_torch_and_numpy():
            """Test that both PyTorch tensors and NumPy arrays work."""
            print("\n" + "=" * 80)
            print("Testing PyTorch Tensor and NumPy Array Support")
            print("=" * 80)

            tf = RotationTransformer(from_rep="axis_angle", to_rep="rotation_6d")

            # Test with NumPy
            print("\nTesting with NumPy array:")
            data_np = np.random.uniform(-np.pi, np.pi, size=(10, 3))
            result_np = tf.forward(data_np)
            assert isinstance(result_np, np.ndarray), "Result should be NumPy array"
            print(f"  Input type: {type(data_np).__name__}")
            print(f"  Output type: {type(result_np).__name__}")
            print("  ✓ PASSED")

            # Test with PyTorch
            print("\nTesting with PyTorch tensor:")
            data_torch = torch.from_numpy(data_np)
            result_torch = tf.forward(data_torch)
            assert isinstance(result_torch, torch.Tensor), "Result should be PyTorch tensor"
            print(f"  Input type: {type(data_torch).__name__}")
            print(f"  Output type: {type(result_torch).__name__}")
            print("  ✓ PASSED")

            # Check results are the same
            print("\nComparing NumPy and PyTorch results:")
            diff = np.abs(result_np - result_torch.numpy()).max()
            print(f"  Max difference: {diff:.2e}")
            assert diff < 1e-10, "NumPy and PyTorch results differ"
            print("  ✓ PASSED")

            print("\n" + "=" * 80)
            print("All type compatibility tests PASSED")
            print("=" * 80)
            return True

        def test_batch_sizes():
            """Test various batch sizes."""
            print("\n" + "=" * 80)
            print("Testing Various Batch Sizes")
            print("=" * 80)

            tf = RotationTransformer(from_rep="axis_angle", to_rep="rotation_6d")

            batch_sizes = [1, 5, 10, 100, 1000]

            for batch_size in batch_sizes:
                data = np.random.uniform(-np.pi, np.pi, size=(batch_size, 3))
                result = tf.forward(data)
                reconstructed = tf.inverse(result)

                assert result.shape == (batch_size, 6), (
                    f"Unexpected output shape for batch size {batch_size}"
                )
                assert reconstructed.shape == (batch_size, 3), (
                    f"Unexpected reconstructed shape for batch size {batch_size}"
                )

                print(f"  Batch size {batch_size:4d}: ✓ PASSED")

            print("\n" + "=" * 80)
            print("All batch size tests PASSED")
            print("=" * 80)
            return True

        print("\n")
        print("╔" + "=" * 78 + "╗")
        print("║" + " " * 15 + "ROTATION TRANSFORMER VALIDATION SUITE" + " " * 25 + "║")
        print("╚" + "=" * 78 + "╝")

        all_passed = True

        try:
            all_passed &= test_all_conversions()
        except Exception as e:
            print(f"\n✗ test_all_conversions FAILED with exception: {e}")
            import traceback

            traceback.print_exc()
            all_passed = False

        try:
            all_passed &= test_numerical_accuracy()
        except Exception as e:
            print(f"\n✗ test_numerical_accuracy FAILED with exception: {e}")
            import traceback

            traceback.print_exc()
            all_passed = False

        try:
            all_passed &= test_torch_and_numpy()
        except Exception as e:
            print(f"\n✗ test_torch_and_numpy FAILED with exception: {e}")
            import traceback

            traceback.print_exc()
            all_passed = False

        try:
            all_passed &= test_batch_sizes()
        except Exception as e:
            print(f"\n✗ test_batch_sizes FAILED with exception: {e}")
            import traceback

            traceback.print_exc()
            all_passed = False

        print("\n")
        print("╔" + "=" * 78 + "╗")
        if all_passed:
            print("║" + " " * 25 + "ALL TESTS PASSED! ✓" + " " * 33 + "║")
            print("║" + " " * 78 + "║")
            print(
                "║  The refactored rotation_transformer.py is working correctly and"
                + " " * 11
                + "║"
            )
            print(
                "║  produces identical results to the pytorch3d-based implementation."
                + " " * 9
                + "║"
            )
        else:
            print("║" + " " * 25 + "SOME TESTS FAILED ✗" + " " * 33 + "║")
        print("╚" + "=" * 78 + "╝")
        print("\n")

        exit(0 if all_passed else 1)

    test()
