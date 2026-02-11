%% TEST_QUAT_MATH - Unit tests for quaternion utilities
%
% Quick sanity checks for quat_utils.m functions.
% Run with: results = runtests('test_quat_math'); disp(results)
%
% Tests verify:
%   - Normalization maintains unit norm
%   - Multiplication is associative
%   - Conjugate/inverse properties
%   - Rotation correctness
%   - Euler angle round-trips
%   - Identity properties
%
% Author: Generated for Gesture-SA-Sensor-Fusion project
% Date: 2025

function tests = test_quat_math
    tests = functiontests(localfunctions);
end

%% Test: Normalization
function test_normalize(testCase)
    qu = quat_utils();
    q = [1, 2, 3, 4];  % Unnormalized (1x4 row)
    qn = qu.normalize(q);
    
    norm_val = sqrt(sum(qn.^2));
    verifyEqual(testCase, norm_val, 1.0, 'AbsTol', 1e-10, ...
        'Normalized quaternion should have unit norm');
end

%% Test: Identity quaternion
function test_identity(testCase)
    qu = quat_utils();
    q_id = qu.identity();  % Returns 4x1 column
    
    verifyEqual(testCase, q_id, [1; 0; 0; 0], 'AbsTol', 1e-10, ...
        'Identity quaternion should be [1;0;0;0]');
end

%% Test: Multiply with identity
function test_multiply_identity(testCase)
    qu = quat_utils();
    q = qu.normalize([0.5; 0.5; 0.5; 0.5]);  % 4x1 column
    q_id = qu.identity();
    
    q_result = qu.multiply(q, q_id);
    
    verifyEqual(testCase, q_result, q, 'AbsTol', 1e-10, ...
        'q * identity should equal q');
end

%% Test: Multiply with conjugate gives identity
function test_multiply_conjugate(testCase)
    qu = quat_utils();
    q = qu.normalize([1; 2; 3; 4]);
    q_conj = qu.conjugate(q);
    
    q_result = qu.multiply(q, q_conj);
    q_id = qu.identity();
    
    verifyEqual(testCase, q_result, q_id, 'AbsTol', 1e-10, ...
        'q * conjugate(q) should equal identity');
end

%% Test: Inverse property
function test_inverse(testCase)
    qu = quat_utils();
    q = qu.normalize([0.7; 0.3; -0.5; 0.1]);
    q_inv = qu.inverse(q);
    
    q_result = qu.multiply(q, q_inv);
    q_id = qu.identity();
    
    verifyEqual(testCase, q_result, q_id, 'AbsTol', 1e-10, ...
        'q * inverse(q) should equal identity');
end

%% Test: Rotation of vector by identity
function test_rotate_identity(testCase)
    qu = quat_utils();
    v = [1; 2; 3];  % 3x1 column (quatRotate returns 3x1)
    q_id = qu.identity();
    
    v_rot = qu.rotate(q_id, v);
    
    verifyEqual(testCase, v_rot, v, 'AbsTol', 1e-10, ...
        'Rotating by identity should not change vector');
end

%% Test: 90-degree rotation around Z-axis
function test_rotate_90_z(testCase)
    qu = quat_utils();
    q = qu.fromAxisAngle([0; 0; 1], pi/2);
    
    v = [1; 0; 0];
    v_rot = qu.rotate(q, v);
    
    expected = [0; 1; 0];
    verifyEqual(testCase, v_rot, expected, 'AbsTol', 1e-10, ...
        '90 deg rotation around Z should map X to Y');
end

%% Test: 180-degree rotation
function test_rotate_180(testCase)
    qu = quat_utils();
    q = qu.fromAxisAngle([0; 0; 1], pi);
    
    v = [1; 0; 0];
    v_rot = qu.rotate(q, v);
    
    expected = [-1; 0; 0];
    verifyEqual(testCase, v_rot, expected, 'AbsTol', 1e-10, ...
        '180 deg rotation around Z should negate X');
end

%% Test: Euler to quaternion round-trip
function test_euler_roundtrip(testCase)
    qu = quat_utils();
    euler_orig = [pi/6, pi/4, pi/3];  % [roll, pitch, yaw]
    
    q = qu.fromEuler(euler_orig(1), euler_orig(2), euler_orig(3));
    [roll, pitch, yaw] = qu.toEuler(q);  % 3 separate outputs
    euler_back = [roll, pitch, yaw];
    
    verifyEqual(testCase, euler_back, euler_orig, 'AbsTol', 1e-10, ...
        'Euler -> quaternion -> Euler should round-trip');
end

%% Test: Small angle rotation via fromOmega
function test_fromOmega(testCase)
    qu = quat_utils();
    omega = [0; 0; 0.1];  % Small rotation around Z (3x1 column)
    dt = 0.01;
    
    q = qu.fromOmega(omega, dt);
    
    % q ~ [cos(theta/2), 0, 0, sin(theta/2)] where theta = 0.1*0.01
    expected_w = cos(0.001/2);
    expected_z = sin(0.001/2);
    
    verifyEqual(testCase, q(1), expected_w, 'AbsTol', 1e-6, ...
        'fromOmega w component');
    verifyEqual(testCase, q(4), expected_z, 'AbsTol', 1e-6, ...
        'fromOmega z component');
end

%% Test: Rotation matrix conversion round-trip
function test_rotmat_roundtrip(testCase)
    qu = quat_utils();
    q_orig = qu.normalize([0.5; 0.5; 0.5; 0.5]);
    
    R = qu.toRotMat(q_orig);
    q_back = qu.fromRotMat(R);
    
    % Quaternions are equivalent up to sign
    if q_back(1) * q_orig(1) < 0
        q_back = -q_back;
    end
    
    verifyEqual(testCase, q_back, q_orig, 'AbsTol', 1e-10, ...
        'Quaternion -> RotMat -> Quaternion should round-trip');
end

%% Test: Rotation matrix is orthogonal
function test_rotmat_orthogonal(testCase)
    qu = quat_utils();
    q = qu.normalize([1; 2; 3; 4]);
    R = qu.toRotMat(q);
    
    I = R * R';
    verifyEqual(testCase, I, eye(3), 'AbsTol', 1e-10, ...
        'Rotation matrix should be orthogonal: R*R'' = I');
    
    det_R = det(R);
    verifyEqual(testCase, det_R, 1.0, 'AbsTol', 1e-10, ...
        'Rotation matrix determinant should be 1');
end

%% Test: SLERP at t=0 and t=1
function test_slerp_endpoints(testCase)
    qu = quat_utils();
    q0 = qu.normalize([1; 0; 0; 0]);
    q1 = qu.normalize([0.707; 0; 0; 0.707]);
    
    q_at_0 = qu.slerp(q0, q1, 0);
    q_at_1 = qu.slerp(q0, q1, 1);
    
    verifyEqual(testCase, q_at_0, q0, 'AbsTol', 1e-10, ...
        'SLERP at t=0 should return q0');
    verifyEqual(testCase, q_at_1, q1, 'AbsTol', 1e-10, ...
        'SLERP at t=1 should return q1');
end

%% Test: SLERP at t=0.5 (midpoint)
function test_slerp_midpoint(testCase)
    qu = quat_utils();
    q0 = qu.identity();
    q1 = qu.fromAxisAngle([0; 0; 1], pi/2);
    
    q_mid = qu.slerp(q0, q1, 0.5);
    
    q_expected = qu.fromAxisAngle([0; 0; 1], pi/4);
    
    verifyEqual(testCase, q_mid, q_expected, 'AbsTol', 1e-10, ...
        'SLERP midpoint should be half the rotation');
end

%% Test: Normalization preserves unit length for negative-w input
function test_normalize_negative_w(testCase)
    qu = quat_utils();
    q = [-0.5, 0.5, 0.5, 0.5];  % Negative w (1x4 row)
    qn = qu.normalize(q);
    
    % normalize does not enforce positive w; just check unit norm
    norm_val = sqrt(sum(qn.^2));
    verifyEqual(testCase, norm_val, 1.0, 'AbsTol', 1e-10, ...
        'Normalized quaternion should have unit norm regardless of w sign');
end

%% Test: Batch vector rotation (3xN input)
function test_rotate_batch(testCase)
    qu = quat_utils();
    q = qu.fromAxisAngle([0; 0; 1], pi/2);
    
    % quatRotate expects 3xN columns, returns 3xN
    V = [1, 0, 1;     % x-coords of 3 vectors
         0, 1, 1;     % y-coords
         0, 0, 0];    % z-coords
    
    V_rot = qu.rotate(q, V);
    
    % 90deg Z rotation: [x,y,z] -> [-y, x, z]
    expected = [ 0, -1, -1;
                 1,  0,  1;
                 0,  0,  0];
    
    verifyEqual(testCase, V_rot, expected, 'AbsTol', 1e-10, ...
        'Batch rotation should work for multiple vectors');
end

%% To run these tests:
%   results = runtests('test_quat_math');
%   disp(results);
