#!/usr/bin/env python3
import os
import sys

# Ensure local venv site-packages is in sys.path when executed via ROS 2 entrypoint
venv_site = os.path.expanduser("~/mujoco_ws/.venv/lib/python3.12/site-packages")
if os.path.exists(venv_site) and venv_site not in sys.path:
    sys.path.insert(0, venv_site)

import rclpy
from rclpy.node import Node
import numpy as np

from sensor_msgs.msg import JointState, Imu
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray
from stable_baselines3 import PPO


class QuadrupedInferenceNode(Node):
    """
    ROS 2 Inference Node running at 50 Hz for 12-DOF Spider / Crawler Quadruped.
    Subscribes:
      - /joint_states (sensor_msgs/JointState)
      - /imu/data (sensor_msgs/Imu)
      - /cmd_vel (geometry_msgs/Twist) [optional teleop]
    Publishes:
      - /leg_position_controller/commands (std_msgs/Float64MultiArray)
    """

    def __init__(self):
        super().__init__('quadruped_policy_inference')

        # Declarations & Parameters
        self.declare_parameter('model_path', '')
        self.declare_parameter('policy_rate', 50.0)  # Hz
        self.declare_parameter('target_velocity_x', 0.4)
        self.declare_parameter('action_scale', 0.30)

        self.model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.rate = self.get_parameter('policy_rate').get_parameter_value().double_value
        self.cmd_vx = self.get_parameter('target_velocity_x').get_parameter_value().double_value
        self.cmd_vy = 0.0
        self.cmd_wz = 0.0
        self.action_scale = self.get_parameter('action_scale').get_parameter_value().double_value

        # Canonical Joint Order (12-DOF: 3 per leg: hip yaw, thigh pitch, knee pitch)
        self.joint_names = [
            'fl_hip_joint', 'fl_thigh_joint', 'fl_knee_joint',
            'fr_hip_joint', 'fr_thigh_joint', 'fr_knee_joint',
            'rl_hip_joint', 'rl_thigh_joint', 'rl_knee_joint',
            'rr_hip_joint', 'rr_thigh_joint', 'rr_knee_joint'
        ]

        # Nominal Spider Standing Stance (12-DOF, CAD calibrated)
        # Hips at 0.0 rad, thighs at 0.45 rad down, knees at 0.95 rad down
        self.default_dof_pos = np.array([
            0.0, 0.45, 0.95,  # FL
            0.0, 0.45, 0.95,  # FR
            0.0, 0.45, 0.95,  # RL
            0.0, 0.45, 0.95   # RR
        ], dtype=np.float32)

        self.dof_lower = np.array([
            -0.80, -0.60, -0.40,
            -0.80, -0.60, -0.40,
            -0.80, -0.60, -0.40,
            -0.80, -0.60, -0.40
        ], dtype=np.float32)

        self.dof_upper = np.array([
             0.80,  1.30,  2.00,
             0.80,  1.30,  2.00,
             0.80,  1.30,  2.00,
             0.80,  1.30,  2.00
        ], dtype=np.float32)

        # State Variables
        self.current_q = self.default_dof_pos.copy()
        self.current_qd = np.zeros(12, dtype=np.float32)
        self.quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # [w, x, y, z]
        self.ang_vel = np.zeros(3, dtype=np.float32)
        self.prev_action = np.zeros(12, dtype=np.float32)

        self.has_joint_states = False
        self.has_imu = False

        # Load Policy Checkpoint
        self.model = None
        if self.model_path and os.path.exists(self.model_path):
            self.get_logger().info(f"Loading trained policy from: {self.model_path}")
            self.model = PPO.load(self.model_path, device="cpu")
            self.get_logger().info("Policy loaded successfully!")
        else:
            self.get_logger().warn(
                f"No valid model_path provided ('{self.model_path}'). "
                "Running in nominal spider standing fallback mode."
            )

        # Subscribers
        self.sub_joints = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10
        )
        self.sub_imu = self.create_subscription(
            Imu, '/imu/data', self.imu_callback, 10
        )
        self.sub_cmd_vel = self.create_subscription(
            Twist, '/cmd_vel', self.cmd_vel_callback, 10
        )

        # Publisher for ros2_control JointGroupPositionController
        self.pub_commands = self.create_publisher(
            Float64MultiArray, '/leg_position_controller/commands', 10
        )

        # Policy inference timer loop at 50 Hz
        timer_period = 1.0 / self.rate
        self.timer = self.create_timer(timer_period, self.control_loop)
        self.get_logger().info(f"Spider Quadruped 12-DOF Policy Inference Node started at {self.rate} Hz.")

    def joint_state_callback(self, msg: JointState):
        name_to_idx = {name: i for i, name in enumerate(msg.name)}
        for i, target_joint in enumerate(self.joint_names):
            if target_joint in name_to_idx:
                idx = name_to_idx[target_joint]
                self.current_q[i] = msg.position[idx]
                if msg.velocity and len(msg.velocity) > idx:
                    self.current_qd[i] = msg.velocity[idx]
        self.has_joint_states = True

    def imu_callback(self, msg: Imu):
        # ROS 2 quat: [x, y, z, w] -> convert to [w, x, y, z]
        self.quat = np.array([
            msg.orientation.w,
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z
        ], dtype=np.float32)
        self.ang_vel = np.array([
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z
        ], dtype=np.float32)
        self.has_imu = True

    def cmd_vel_callback(self, msg: Twist):
        self.cmd_vx = msg.linear.x
        self.cmd_vy = msg.linear.y
        self.cmd_wz = msg.angular.z

    def _quat_to_rot_mat(self, q):
        w, x, y, z = q[0], q[1], q[2], q[3]
        return np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w,     2*x*z + 2*y*w],
            [2*x*y + 2*z*w,     1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w,     2*y*z + 2*x*w,     1 - 2*x*x - 2*y*y]
        ], dtype=np.float32)

    def control_loop(self):
        # 1. Base orientation: projected gravity vector in body frame
        rot_mat = self._quat_to_rot_mat(self.quat)
        projected_gravity = -rot_mat[2, :]

        # 2. Base angular velocity in body frame
        ang_vel_body = rot_mat.T @ self.ang_vel

        # 3. Joint position error from default nominal standing pose (12)
        joint_pos_err = (self.current_q - self.default_dof_pos).astype(np.float32)

        # 4. Commands (3)
        commands = np.array([self.cmd_vx, self.cmd_vy, self.cmd_wz], dtype=np.float32)

        # 5. Assemble Observation Vector (45 dims)
        obs = np.concatenate([
            projected_gravity,
            ang_vel_body,
            joint_pos_err,
            self.current_qd,
            self.prev_action,
            commands
        ]).astype(np.float32)

        # 6. Policy Inference
        if self.model is not None:
            action, _ = self.model.predict(obs, deterministic=True)
            action = np.clip(action, -1.0, 1.0)
        else:
            action = np.zeros(12, dtype=np.float32)

        self.prev_action = action.copy()

        # 7. Compute target joint angles
        target_q = self.default_dof_pos + action * self.action_scale
        target_q = np.clip(target_q, self.dof_lower, self.dof_upper)

        # 8. Publish command to /leg_position_controller/commands
        cmd_msg = Float64MultiArray()
        cmd_msg.data = target_q.tolist()
        self.pub_commands.publish(cmd_msg)


def main(args=None):
    rclpy.init(args=args)
    node = QuadrupedInferenceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
