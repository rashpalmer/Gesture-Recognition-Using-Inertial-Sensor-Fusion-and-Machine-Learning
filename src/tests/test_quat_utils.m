%% TEST_QUAT_UTILS - Unit tests for quaternion utility functions
% Run with: runtests('test_quat_utils')
%
% Tests cover:
%   - Quaternion normalization
%   - Quaternion multiplication
%   - Quaternion rotation
%   - Euler angle conversions
%   - Axis-angle conversions
%   - SLERP interpolation
%   - Edge cases and numerical stability
%
% Author: Sensor Fusion Demo (Test Suite)
% Date: 2026

classdef test_quat_utils < matlab.unittest.TestCase

    properties
        qu  % Quaternion utilities struct
        tol % Numerical tolerance
    end

    methods(TestMethodSetup)
        function setup(testCase)
            testCase.qu = quat_utils();
            testCase.tol = 1e-10;
        end
    end

    %% ==================== NORMALIZATION TESTS ====================

    methods(Test)
        function test_normalize_unit_quaternion(testCase)
            % Unit quaternion should remain unchanged
            q = [1; 0; 0; 0];
            qn = testCase.qu.normalize(q);
            testCase.verifyEqual(qn, q, 'AbsTol', testCase.tol);
        end

        function test_normalize_non_unit_quaternion(testCase)
            % Non-unit quaternion should have norm = 1 after normalization
            q = [2; 1; 0; 0];
            qn = testCase.qu.normalize(q);
            testCase.verifyEqual(norm(qn), 1, 'AbsTol', testCase.tol);
        end

        function test_normalize_batch(testCase)
            % Batch normalization (Nx4)
            Q = [2 1 0 0; 0 1 1 0; 1 1 1 1];
            Qn = testCase.qu.normalize(Q);
            norms = sqrt(sum(Qn.^2, 2));
            testCase.verifyEqual(norms, ones(3,1), 'AbsTol', testCase.tol);
        end

        function test_normalize_zero_quaternion(testCase)
            % Zero quaternion should return identity
            q = [0; 0; 0; 0];
            qn = testCase.qu.normalize(q);
            testCase.verifyEqual(qn, [1; 0; 0; 0], 'AbsTol', testCase.tol);
        end
    end

    %% ==================== MULTIPLICATION TESTS ====================

    methods(Test)
        function test_multiply_identity(testCase)
            % Multiplication by identity should not change quaternion
            q = testCase.qu.normalize([1; 2; 3; 4]);
            qi = [1; 0; 0; 0];
            result = testCase.qu.multiply(q, qi);
            testCase.verifyEqual(result, q, 'AbsTol', testCase.tol);
        end

        function test_multiply_inverse(testCase)
            % q * q^(-1) should give identity
            q = testCase.qu.normalize([1; 2; 3; 4]);
            q_inv = testCase.qu.inverse(q);
            result = testCase.qu.multiply(q, q_inv);
            expected = [1; 0; 0; 0];
            testCase.verifyEqual(result, expected, 'AbsTol', testCase.tol);
        end

        function test_multiply_non_commutative(testCase)
            % Quaternion multiplication is non-commutative
            q1 = testCase.qu.normalize([1; 1; 0; 0]);
            q2 = testCase.qu.normalize([1; 0; 1; 0]);
            r1 = testCase.qu.multiply(q1, q2);
            r2 = testCase.qu.multiply(q2, q1);
            testCase.verifyNotEqual(r1, r2);
        end

        function test_multiply_batch(testCase)
            % Batch multiplication
            Q1 = [1 0 0 0; 0 1 0 0];
            Q2 = [1 0 0 0; 0 0 1 0];
            result = testCase.qu.multiply(Q1, Q2);
            testCase.verifySize(result, [2, 4]);
        end
    end

    %% ==================== ROTATION TESTS ====================

    methods(Test)
        function test_rotate_identity(testCase)
            % Identity quaternion should not change vector
            q = [1; 0; 0; 0];
            v = [1; 2; 3];
            v_rot = testCase.qu.rotate(q, v);
            testCase.verifyEqual(v_rot, v, 'AbsTol', testCase.tol);
        end

        function test_rotate_90deg_z(testCase)
            % 90 degree rotation about Z axis
            q = testCase.qu.fromAxisAngle([0; 0; 1], pi/2);
            v = [1; 0; 0];
            v_rot = testCase.qu.rotate(q, v);
            expected = [0; 1; 0];
            testCase.verifyEqual(v_rot, expected, 'AbsTol', testCase.tol);
        end

        function test_rotate_preserves_magnitude(testCase)
            % Rotation should preserve vector magnitude
            q = testCase.qu.normalize([1; 2; 3; 4]);
            v = [1; 2; 3];
            v_rot = testCase.qu.rotate(q, v);
            testCase.verifyEqual(norm(v_rot), norm(v), 'AbsTol', testCase.tol);
        end

        function test_rotate_batch(testCase)
            % Batch rotation
            q = testCase.qu.fromAxisAngle([0; 0; 1], pi/4);
            V = [1 0 0; 0 1 0; 0 0 1]';  % 3xN
            V_rot = testCase.qu.rotateBatch(q, V);
            testCase.verifySize(V_rot, [3, 3]);

            % Each rotated vector should have same magnitude
            for i = 1:3
                testCase.verifyEqual(norm(V_rot(:,i)), norm(V(:,i)), 'AbsTol', testCase.tol);
            end
        end
    end

    %% ==================== EULER ANGLE TESTS ====================

    methods(Test)
        function test_euler_roundtrip(testCase)
            % Converting to Euler and back should give same quaternion
            q_orig = testCase.qu.normalize([0.9; 0.1; 0.2; 0.3]);

            [roll, pitch, yaw] = testCase.qu.toEuler(q_orig);
            q_back = testCase.qu.fromEuler(roll, pitch, yaw);

            % Quaternions may differ by sign
            if q_orig(1)*q_back(1) < 0
                q_back = -q_back;
            end

            testCase.verifyEqual(q_back, q_orig, 'AbsTol', 1e-6);
        end

        function test_euler_zero_angles(testCase)
            % Zero Euler angles should give identity quaternion
            q = testCase.qu.fromEuler(0, 0, 0);
            expected = [1; 0; 0; 0];
            testCase.verifyEqual(q, expected, 'AbsTol', testCase.tol);
        end

        function test_euler_gimbal_lock(testCase)
            % Test near gimbal lock (pitch = ±90°)
            q = testCase.qu.fromEuler(0.1, pi/2 - 0.01, 0.2);
            [roll, pitch, yaw] = testCase.qu.toEuler(q);

            % Should not produce NaN
            testCase.verifyFalse(isnan(roll));
            testCase.verifyFalse(isnan(pitch));
            testCase.verifyFalse(isnan(yaw));
        end
    end

    %% ==================== AXIS-ANGLE TESTS ====================

    methods(Test)
        function test_axis_angle_identity(testCase)
            % Zero angle should give identity
            q = testCase.qu.fromAxisAngle([0; 0; 1], 0);
            expected = [1; 0; 0; 0];
            testCase.verifyEqual(q, expected, 'AbsTol', testCase.tol);
        end

        function test_axis_angle_360(testCase)
            % 360 degree rotation should be equivalent to identity
            q = testCase.qu.fromAxisAngle([1; 0; 0], 2*pi);
            % q should be approximately [-1; 0; 0; 0] or [1; 0; 0; 0]
            testCase.verifyEqual(abs(q(1)), 1, 'AbsTol', 1e-6);
        end

        function test_from_omega(testCase)
            % Test quaternion from angular velocity
            omega = [0; 0; 1];  % 1 rad/s about Z
            dt = 0.1;
            q = testCase.qu.fromOmega(omega, dt);

            % Should be small rotation about Z
            expected = testCase.qu.fromAxisAngle([0; 0; 1], 0.1);
            testCase.verifyEqual(q, expected, 'AbsTol', 1e-6);
        end
    end

    %% ==================== SLERP TESTS ====================

    methods(Test)
        function test_slerp_endpoints(testCase)
            % SLERP at t=0 and t=1 should give endpoints
            q1 = testCase.qu.normalize([1; 0; 0; 0]);
            q2 = testCase.qu.normalize([0; 1; 0; 0]);

            r0 = testCase.qu.slerp(q1, q2, 0);
            r1 = testCase.qu.slerp(q1, q2, 1);

            testCase.verifyEqual(r0, q1, 'AbsTol', testCase.tol);
            testCase.verifyEqual(r1, q2, 'AbsTol', testCase.tol);
        end

        function test_slerp_midpoint(testCase)
            % SLERP at t=0.5 should be unit quaternion
            q1 = testCase.qu.normalize([1; 0; 0; 0]);
            q2 = testCase.qu.normalize([0; 1; 0; 0]);

            r = testCase.qu.slerp(q1, q2, 0.5);
            testCase.verifyEqual(norm(r), 1, 'AbsTol', testCase.tol);
        end

        function test_slerp_identical(testCase)
            % SLERP between identical quaternions
            q = testCase.qu.normalize([1; 2; 3; 4]);
            r = testCase.qu.slerp(q, q, 0.5);
            testCase.verifyEqual(r, q, 'AbsTol', testCase.tol);
        end
    end

    %% ==================== ROTATION MATRIX TESTS ====================

    methods(Test)
        function test_rotmat_identity(testCase)
            % Identity quaternion should give identity matrix
            q = [1; 0; 0; 0];
            R = testCase.qu.toRotMat(q);
            testCase.verifyEqual(R, eye(3), 'AbsTol', testCase.tol);
        end

        function test_rotmat_orthogonal(testCase)
            % Rotation matrix should be orthogonal
            q = testCase.qu.normalize([1; 2; 3; 4]);
            R = testCase.qu.toRotMat(q);

            testCase.verifyEqual(R * R', eye(3), 'AbsTol', testCase.tol);
            testCase.verifyEqual(det(R), 1, 'AbsTol', testCase.tol);
        end

        function test_rotmat_roundtrip(testCase)
            % Converting to matrix and back should preserve quaternion
            q_orig = testCase.qu.normalize([0.9; 0.1; 0.2; 0.3]);
            R = testCase.qu.toRotMat(q_orig);
            q_back = testCase.qu.fromRotMat(R);

            % Quaternions may differ by sign
            if q_orig(1)*q_back(1) < 0
                q_back = -q_back;
            end

            testCase.verifyEqual(q_back, q_orig, 'AbsTol', 1e-6);
        end
    end

    %% ==================== DERIVATIVE TESTS ====================

    methods(Test)
        function test_derivative_zero_omega(testCase)
            % Zero angular velocity should give zero derivative
            q = testCase.qu.normalize([1; 2; 3; 4]);
            omega = [0; 0; 0];
            q_dot = testCase.qu.derivative(q, omega);

            testCase.verifyEqual(q_dot, zeros(4,1), 'AbsTol', testCase.tol);
        end

        function test_derivative_magnitude(testCase)
            % Derivative magnitude should be proportional to angular velocity
            q = [1; 0; 0; 0];
            omega1 = [1; 0; 0];
            omega2 = [2; 0; 0];

            q_dot1 = testCase.qu.derivative(q, omega1);
            q_dot2 = testCase.qu.derivative(q, omega2);

            testCase.verifyEqual(norm(q_dot2), 2*norm(q_dot1), 'AbsTol', testCase.tol);
        end
    end

end
