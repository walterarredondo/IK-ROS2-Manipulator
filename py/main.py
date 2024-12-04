import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import pandas as pd
import itertools
import matplotlib.pyplot as plt
from numpy import cos, sin, pi
from sympy import symbols, sin, cos
import csv
import os
import hashlib
import pickle
import serial
import time
import sys
import os
import argparse

class RobotArmIK:
    def __init__(self, dh_params):
        """
        Initialize the robot arm inverse kinematics solver.

        :param dh_params: List of DH parameters [d, theta, a, alpha].
                           Use sympy symbols for joint variables (e.g., sp.Symbol('q1')).
        """
        self.DH_params = dh_params
        self.DOF = len([param for param in dh_params if isinstance(param[1], sp.Symbol)])

        # Create symbolic variables for joints
        self.joint_symbols = [param[1] for param in dh_params if isinstance(param[1], sp.Symbol)]

        # Precompute or load symbolic jacobian
        self.jacobian_symbolic = self._load_or_compute_jacobian()

    def _generate_dh_params_hash(self):
        """
        Generate a unique hash for the DH parameters to use in filename.
        
        :return: Hash string representing the DH parameters
        """
        # Convert DH params to a hashable representation
        hashable_params = str([(list(map(str, param)) if isinstance(param[1], sp.Symbol) else list(map(float, param))) for param in self.DH_params])
        return hashlib.md5(hashable_params.encode()).hexdigest()[:10]

    def _load_jacobian_from_file(self, filename):
        """
        Load Jacobian matrix from a file.
        
        :param filename: Path to the file
        :return: Loaded Jacobian matrix or None if loading fails
        """
        try:
            with open(filename, 'rb') as file:
                loaded_jacobian = pickle.load(file)
                print(f"Jacobian loaded from {filename}")
                return loaded_jacobian
        except (FileNotFoundError, pickle.UnpicklingError) as e:
            print(f"Error loading Jacobian: {e}")
            return None

    def _export_jacobian_to_file(self, jacobian, filename):
        """
        Export Jacobian matrix to a file.
        
        :param jacobian: Symbolic Jacobian matrix to export
        :param filename: Path to save the file
        """
        try:
            with open(filename, 'wb') as file:
                pickle.dump(jacobian, file)
            print(f"Jacobian exported to {filename}")
        except Exception as e:
            print(f"Error exporting Jacobian: {e}")

    def _load_or_compute_jacobian(self):
        """
        Load Jacobian from file if exists, otherwise compute and export.
        
        :return: Symbolic Jacobian matrix
        """
        # Create cache directory if it doesn't exist
        cache_dir = 'jacobian_cache'
        os.makedirs(cache_dir, exist_ok=True)

        # Generate unique filename based on DH parameters
        dh_hash = self._generate_dh_params_hash()
        cache_filename = os.path.join(cache_dir, f'jacobian_{dh_hash}.pkl')

        # Try to load existing Jacobian
        cached_jacobian = self._load_jacobian_from_file(cache_filename)
        if cached_jacobian is not None:
            return cached_jacobian

        # If not found, compute Jacobian
        print("Computing Jacobian...")
        computed_jacobian = self._compute_jacobian()

        # Export newly computed Jacobian
        self._export_jacobian_to_file(computed_jacobian, cache_filename)

        return computed_jacobian

    def _dh_trans_matrix(self, params):
        """
        Generate transformation matrix from DH parameters.

        :param params: [d, theta, a, alpha]
        :return: Symbolic transformation matrix
        """
        d, theta, a, alpha = params
        return sp.Matrix([
            [cos(theta), -sin(theta) * cos(alpha), sin(theta) * sin(alpha), a * cos(theta)],
            [sin(theta), cos(theta) * cos(alpha), -cos(theta) * sin(alpha), a * sin(theta)],
            [0, sin(alpha), cos(alpha), d],
            [0, 0, 0, 1]
        ])

    def _compute_joint_transforms(self, joints=[]):
        """
        Compute transformations from origin to each joint.

        :param joints: If provided, substitute joint angles into the matrices.
        :return: List of transformation matrices
        """
        transforms = [sp.eye(4)]  # Start with identity matrix

        for params in self.DH_params:
            current_params = list(params)  # Create a copy to avoid modifying original params
            if len(joints) > 0:
                for i, symbol in enumerate(self.joint_symbols):
                    if current_params[1] == symbol:
                        current_params[1] = joints[i]
            transforms.append(self._dh_trans_matrix(current_params))

        return transforms

    def _compute_jacobian(self):
        """
        Compute the symbolic Jacobian.

        :return: Symbolic Jacobian matrix
        """
        transforms = self._compute_joint_transforms()

        # Compute total transformation to end effector
        trans_EF = transforms[0]
        for mat in transforms[1:]:
            trans_EF = trans_EF * mat

        pos_EF = trans_EF[0:3, 3]

        # Initialize Jacobian
        J = sp.zeros(6, self.DOF)

        for joint in range(self.DOF):
            # Transformation to current joint
            trans_joint = transforms[0]
            for i in range(joint + 1):
                if isinstance(self.DH_params[i][1], sp.Symbol):
                    symbolic_trans_matrix = self._dh_trans_matrix(self.DH_params[i])
                    trans_joint = trans_joint * symbolic_trans_matrix
                else:
                    trans_joint = trans_joint * self._dh_trans_matrix(self.DH_params[i])

            z_axis = trans_joint[0:3, 2]
            pos_joint = trans_joint[0:3, 3]

            Jv = z_axis.cross(pos_EF - pos_joint)
            Jw = z_axis

            J[0:3, joint] = Jv
            J[3:6, joint] = Jw

        return sp.simplify(J)

    def _forward_kinematics(self, joints):
        """
        Compute end effector position and orientation.

        :param joints: Joint angles
        :return: Transformation matrix (numpy array)
        """
        transforms = self._compute_joint_transforms(joints)
        trans_EF = np.eye(4)
        for mat in transforms[1:]:
            trans_EF = trans_EF @ np.array(mat).astype(np.float64)

        return trans_EF

    def inverse_kinematics(self, target, initial_joints,
                            no_rotation=True,
                            joint_limits=None,
                            max_iterations=1000,
                            position_tolerance=1e-3,
                            rotation_tolerance = 1e-3,
                            lambda_damping=0.1,
                            alpha=1.0):
        """
        Solve inverse kinematics using Damped Least Squares method with advanced modifications.

        :param target: Target transformation matrix (numpy array).
        :param initial_joints: Initial joint configuration.
        :param no_rotation: Ignore end effector orientation.
        :param joint_limits: List of tuples specifying (lower, upper) joint limits for each DOF.
        :param max_iterations: Maximum number of iterations.
        :param position_tolerance: Position convergence threshold.
        :param rotation_tolerance: Rotation convergence threshold.
        :param lambda_damping: Damping factor for stability.
        :param alpha: Step size factor.
        :return: Optimized joint angles (numpy array)
        """
        # Optimize numpy operations
        joints = np.asarray(initial_joints, dtype=np.float64)
        target_pos = target[0:3, 3]
        target_rot = target[0:3, 0:3]

        best_error = float('inf')
        best_joints = joints.copy()
        error_history = []
        
        # Change: start from first joint (index 0)
        randomized_joint_index = 0
        randomization_count = 0

        # Set default joint limits if not provided
        if joint_limits is None:
            joint_limits = [(-np.pi, np.pi) for _ in range(len(joints))]

        # Preallocate some arrays to reduce memory allocations
        error = np.zeros(6, dtype=np.float64)
        delta_joints = np.zeros_like(joints, dtype=np.float64)

        for iteration in range(max_iterations):
            # Compute current transformation more efficiently
            current_trans = self._forward_kinematics(joints)
            current_pos = current_trans[0:3, 3]
            current_rot = current_trans[0:3, 0:3]

            # Position error
            error[0:3] = target_pos - current_pos

            # Rotation error computation
            R_error = target_rot @ current_rot.T
            v = np.arccos(np.clip((np.trace(R_error) - 1) / 2, -1.0, 1.0))  # Use clip to handle numerical errors
                
            if np.abs(v) > 1e-6:  # Check for significant rotation error
                r_axis = (1 / (2 * np.sin(v))) * np.array([R_error[2, 1] - R_error[1, 2],
                                                            R_error[0, 2] - R_error[2, 0],
                                                            R_error[1, 0] - R_error[0, 1]])
                error[3:6] = v * r_axis
            else:
                error[3:6] = 0
                    
            # Error norm computation
            error_norm = np.linalg.norm(error)

            # Save best configuration
            if error_norm < best_error:
                best_error = error_norm
                best_joints = joints.copy()

            # Error tracking
            error_history.append(error_norm)
            if len(error_history) > 10:
                error_history.pop(0)

            # Convergence check with reset mechanism
            epsilon = 1e-2
            m = 1
            if len(error_history) == 10 and abs(sum(error_history) / len(error_history) - error_history[-1]) < epsilon:
                randomization_count += 1
                
                # Change: Use global best configuration for randomization
                joints = best_joints.copy()
                
                if randomization_count < m:  # Randomize same joint multiple times
                    print(f"Resetting to best configuration and randomizing joint {randomized_joint_index + 1} (attempt {randomization_count}) at iteration {iteration + 1}")
                    joints[randomized_joint_index] = np.random.uniform(
                        joint_limits[randomized_joint_index][0], 
                        joint_limits[randomized_joint_index][1]
                    )
                else:  # Move to next joint
                    print(f"Randomized joint {randomized_joint_index + 1} for {m} times. Moving to next joint.")
                    randomized_joint_index += 1
                    randomization_count = 0
                    
                    if randomized_joint_index >= len(joints):  # Reset to first joint if all joints have been tried
                        randomized_joint_index = 0
                    
                    print(f"Resetting to best configuration and randomizing joint {randomized_joint_index + 1} (attempt {randomization_count + 1}) at iteration {iteration + 1}")
                    joints[randomized_joint_index] = np.random.uniform(
                        joint_limits[randomized_joint_index][0], 
                        joint_limits[randomized_joint_index][1]
                    )

                error_history = []  # Clear error history

            # Convergence check
            if error_norm < position_tolerance and (no_rotation or np.linalg.norm(error[3:6]) < rotation_tolerance):
                print(f"Converged after {iteration + 1} iterations.")
                break

            # Compute Jacobian (use just-in-time computation)
            jac = np.array(self.jacobian_symbolic.subs(list(zip(self.joint_symbols, joints)))).astype(np.float64)

            # Use efficient pseudoinverse computation
            damped_jac = jac.T @ np.linalg.inv(jac @ jac.T + (lambda_damping ** 2) * np.eye(6))

            # Compute joint velocity
            np.multiply(alpha, damped_jac @ error, out=delta_joints)

            # Update joints and apply limits
            joints += delta_joints
            joints = self._apply_joint_limits(joints, joint_limits)

            # Simplified logging
            if iteration % 10 == 0:  # Log every 10 iterations to reduce overhead
                print(f"Iteration {iteration + 1}: Error = {error_norm:.4f}")

        if iteration == max_iterations - 1:
            print("Maximum iterations reached without convergence.")

        return joints

    def _apply_joint_limits(self, joints, limits):
        """
        Apply joint angle limits.

        :param joints: Current joint angles.
        :param limits: List of tuples specifying (lower, upper) joint limits for each DOF.
        :return: Joint angles within limits.
        """
        limited_joints = np.array(joints)
        for i, (joint, (lower, upper)) in enumerate(zip(joints, limits)):
            limited_joints[i] = max(min(joint, upper), lower)
        return limited_joints

    def plot_pose(self, joints):
        """
        Plot robot arm configuration.

        :param joints: Joint angles.
        """
        transforms = self._compute_joint_transforms(joints)

        xs, ys, zs = [], [], []

        # Track joint positions
        current_transform = np.eye(4)
        for transform in transforms[1:]:
            current_transform = current_transform @ np.array(transform).astype(np.float64)
            xs.append(current_transform[0, 3])
            ys.append(current_transform[1, 3])
            zs.append(current_transform[2, 3])

        # Plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(xs, ys, zs, marker='o')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        ax.set_xlim([-50, 150])  # Adjust limits as needed
        ax.set_ylim([-100, 100])
        ax.set_zlim([0, 200])

        plt.show()


def generate_joint_configurations(joint_limits, delta_angle):
    """
    Generate all possible joint configurations within given limits.
    
    :param joint_limits: List of tuples with (lower, upper) limits for each joint
    :param delta_angle: Step size in radians
    :return: List of joint configurations
    """
    # Convert delta to radians if given in degrees
    if delta_angle > 2*pi:
        delta_angle = np.deg2rad(delta_angle)
    
    # Generate ranges for each joint
    joint_ranges = [
        np.arange(limits[0], limits[1] + delta_angle, delta_angle) 
        for limits in joint_limits
    ]
    
    # Create all combinations
    return list(itertools.product(*joint_ranges))

def export_joint_configurations(robot_ik, joint_limits, delta_angle, target):
    """
    Export joint configurations and their end effector positions to CSV.
    
    :param robot_ik: RobotArmIK instance
    :param joint_limits: Joint angle limits
    :param delta_angle: Step size for joint angles
    :param target: Target transformation matrix
    :return: Path to the generated CSV file
    """
    # Generate configurations
    configurations = generate_joint_configurations(joint_limits, delta_angle)
    
    # Prepare CSV file
    csv_filename = 'joint_configurations.csv'
    
    with open(csv_filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        # Write header
        csvwriter.writerow(['q1', 'q2', 'q3', 'q4', 'x', 'y', 'z', 
                            'roll', 'pitch', 'yaw'])
        
        # Compute and write end effector positions
        for config in configurations:
            try:
                # Compute forward kinematics
                trans_matrix = robot_ik._forward_kinematics(config)
                
                # Extract position
                pos = trans_matrix[0:3, 3]
                
                # Extract rotation (convert to Euler angles)
                R = trans_matrix[0:3, 0:3]
                sy = np.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
                singular = sy < 1e-6
                
                if not singular:
                    roll = np.arctan2(R[2,1] , R[2,2])
                    pitch = np.arctan2(-R[2,0], sy)
                    yaw = np.arctan2(R[1,0], R[0,0])
                else:
                    roll = np.arctan2(-R[1,2], R[1,1])
                    pitch = np.arctan2(-R[2,0], sy)
                    yaw = 0
                
                # Write to CSV
                csvwriter.writerow([*config, *pos, roll, pitch, yaw])
                
            except Exception as e:
                print(f"Error processing configuration {config}: {e}")
    
    return csv_filename

def find_best_initial_position(robot_ik, target, joint_limits, delta_angle, no_rotation=False):
    """
    Find best initial position by generating or loading CSV.
    
    :param robot_ik: RobotArmIK instance
    :param target: Target transformation matrix
    :param joint_limits: Joint angle limits
    :param delta_angle: Step size for joint angles
    :param no_rotation: Flag to ignore rotation error
    :return: Best initial joint configuration
    """
    csv_filename = 'joint_configurations.csv'
    
    # Only generate CSV if it doesn't exist
    if not os.path.exists(csv_filename):
        csv_filename = export_joint_configurations(robot_ik, joint_limits, delta_angle, target)
    
    # Read CSV
    df = pd.read_csv(csv_filename)
    
    # Compute errors for each configuration
    def compute_error(row):
        joints = row[['q1', 'q2', 'q3', 'q4']].values
        try:
            current_trans = robot_ik._forward_kinematics(joints)
            
            # Position error
            pos_error = np.linalg.norm(target[0:3, 3] - current_trans[0:3, 3])
            
            # Rotation error (only if no_rotation is False)
            if not no_rotation:
                R_error = target[0:3, 0:3] @ current_trans[0:3, 0:3].T
                v = np.arccos(np.clip((np.trace(R_error) - 1) / 2, -1.0, 1.0))
                return pos_error + v  # Combined error metric
            else:
                return pos_error
        except Exception:
            return np.inf
    
    # Add error column
    df['error'] = df.apply(compute_error, axis=1)
    
    # Find configuration with minimal error
    best_config = df.loc[df['error'].idxmin()]
    
    print("Best Initial Configuration:")
    print(best_config[['q1', 'q2', 'q3', 'q4', 'error']])
    
    return best_config[['q1', 'q2', 'q3', 'q4']].values


def convert_joint_to_servo_angle(joint_angle, joint_index):
    """
    Convert radians from robot arm IK to servo angles (0-180 degrees).
    
    :param joint_angle: Joint angle in radians
    :param joint_index: Index of the joint (0-based)
    :return: Servo angle (0-180 degrees)
    """
    # Specific adjustments for each joint based on servo mounting and arm geometry
    joint_mappings = [
        # Joint 1 (Base): Map from [-π/2, π/2] to [0, 180]
        lambda angle: np.rad2deg(angle) + 90,
        
        # Joint 2: Map from [-π/2, π/2] to [0, 180]
        lambda angle: np.rad2deg(angle) + 90,
        
        # Joint 3: More complex mapping, might need calibration
        lambda angle: np.rad2deg(angle) + 90,
        
        # Joint 4: Similar to others
        lambda angle: np.rad2deg(angle) + 90
    ]
    
    # Clip the result to 0-180 range
    return max(0, min(180, int(joint_mappings[joint_index](joint_angle))))

def send_joint_config_to_arduino(final_joints, port='/dev/ttyUSB0', baud_rate=115200, timeout=2):
    """
    Send joint configurations to Arduino via serial communication.
    
    :param final_joints: NumPy array of joint angles in radians
    :param port: Serial port name (default '/dev/ttyUSB0')
    :param baud_rate: Serial communication baud rate
    :param timeout: Connection timeout in seconds
    """
    try:
        # Open serial connection
        ser = serial.Serial(port, baud_rate, timeout=timeout)
        
        # Wait for Arduino to initialize
        time.sleep(2)
        
        # Convert and send each joint angle
        for joint_index, joint_angle in enumerate(final_joints, 1):
            servo_angle = convert_joint_to_servo_angle(joint_angle, joint_index - 1)
            
            # Format: "joint,angle\n"
            command = f"{joint_index},{servo_angle}\n"
            ser.write(command.encode('utf-8'))
            
            # Small delay to ensure Arduino processes each command
            time.sleep(0.5)
            
            # Optional: Read and print Arduino's response
            if ser.in_waiting:
                response = ser.readline().decode('utf-8').strip()
                print(f"Arduino response: {response}")
        
        print("Joint configuration sent successfully!")
    
    except serial.SerialException as e:
        print(f"Serial communication error: {e}")
        print("Ensure:")
        print("1. Arduino is connected")
        print("2. Correct port is specified")
        print("3. No other program is using the port")
    
    except Exception as e:
        print(f"Unexpected error: {e}")
    
    finally:
        # Close serial connection if it was opened
        if 'ser' in locals() and ser.is_open:
            ser.close()

def main():
    # DH Parameters (same as in the original script)
    DH_params = [
        [20, 0, 0, 0],
        [0, sp.Symbol('q1'), 0, 0],
        [35, pi/2, 0, pi/2],
        [0, sp.Symbol('q2'), 0, 0],
        [0, 0, 90, 0],
        [0, pi/2, 30, 0],
        [0, sp.Symbol('q3'), 0, 0],
        [0, -pi/2, 90, 0],
        [0, sp.Symbol('q4'), 0, 0],
        [0, 0, 90, 0]
    ]

    # Joint limits
    joint_limits = [
        (-pi/2, pi/2),  # Limits for q1
        (-pi/2, pi/2),  # Limits for q2
        (-pi/2, pi/2),  # Limits for q3
        (-pi/2, pi/2)   # Limits for q4
    ]

    # Create argument parser
    parser = argparse.ArgumentParser(description='Robot Arm Position Controller')
    parser.add_argument('x', type=float, help='X coordinate')
    parser.add_argument('y', type=float, help='Y coordinate')
    parser.add_argument('z', type=float, help='Z coordinate')
    parser.add_argument('--port', type=str, default='/dev/ttyUSB0', 
                        help='Arduino serial port (default: /dev/ttyUSB0)')
    parser.add_argument('--no-send', action='store_true', 
                        help='Calculate joint angles without sending to Arduino')
    parser.add_argument('--plot', action='store_true', 
                        help='Plot the robot arm configuration')

    # Parse arguments
    args = parser.parse_args()

    # Create Robot Arm solver
    robot_ik = RobotArmIK(DH_params)

    # Create target transformation matrix
    target = np.array([
        [1, 0, 0, args.x],
        [0, 1, 0, args.y],
        [0, 0, 1, args.z],
        [0, 0, 0, 1]
    ])

    # Delta angle (72 degrees)
    delta_angle = np.deg2rad(72)

    try:
        # Find best initial position
        initial_joints = find_best_initial_position(
            robot_ik, 
            target=target, 
            joint_limits=joint_limits, 
            delta_angle=delta_angle, 
            no_rotation=True
        )

        # Solve Inverse Kinematics
        final_joints = robot_ik.inverse_kinematics(
            target,
            initial_joints,
            joint_limits=joint_limits,
            no_rotation=True,
            max_iterations=10000,
            position_tolerance=10,
            rotation_tolerance=1e-1,
            lambda_damping=0.1,
            alpha=1
        )

        # Print calculated joint angles
        print("Calculated Joint Angles (radians):")
        for i, angle in enumerate(final_joints, 1):
            print(f"Joint {i}: {angle:.4f} rad")
        
        # Plot if requested
        if args.plot:
            robot_ik.plot_pose(final_joints)

        # Send to Arduino unless no-send flag is set
        if not args.no_send:
            send_joint_config_to_arduino(final_joints, port=args.port)

    except Exception as e:
        print(f"Error calculating or sending joint configuration: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Ensure numpy and sympy use of pi is consistent
    from numpy import pi
   
