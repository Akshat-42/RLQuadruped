import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco


class QuadrupedEnv(gym.Env):
    """
    Gymnasium environment for 12-DOF Spider / Crawler Quadruped Robot using native MuJoCo.
    Kinematics: Radial outward legs in X-pattern (+/- 45 deg, +/- 135 deg).
      - Hip Yaw: Rotates horizontally around vertical Z (+/- 0.80 rad)
      - Thigh Pitch: Rotates outward and downward around transverse Y (-0.60 to 1.30 rad)
      - Knee Pitch: Bends downward to ground around transverse Y (-0.40 to 2.00 rad)
    Decoupled high-speed rollout:
      - Physics Rate: 500 Hz (timestep = 0.002s)
      - Policy Step Rate: 50 Hz (decimation = 10)
    Actuators: 12x MG996R hobby servo simulation with PD control, deadband,
               speed limit (~6 rad/s), and torque saturation (1.0 Nm).
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        xml_path=None,
        render_mode=None,
        target_velocity=0.4,  # m/s forward
        action_scale=0.30,    # rad target offset scaling
        max_episode_steps=1000,
    ):
        super().__init__()

        # Resolve XML path
        if xml_path is None:
            ws_path = os.path.expanduser("~/mujoco_ws/src/quadruped_description/mujoco/quadruped.xml")
            if os.path.exists(ws_path):
                xml_path = ws_path
            else:
                xml_path = os.path.join(os.path.dirname(__file__), "quadruped.xml")

        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"MuJoCo XML file not found at: {xml_path}")

        self.xml_path = xml_path
        self.render_mode = render_mode
        self.target_velocity = target_velocity
        self.action_scale = action_scale
        self.max_episode_steps = max_episode_steps
        self.step_count = 0

        # Load MuJoCo Model and Data
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)

        # Simulation timing: 500 Hz physics, 10 decimation steps = 50 Hz policy
        self.physics_dt = self.model.opt.timestep  # 0.002s
        self.decimation = 10
        self.dt = self.physics_dt * self.decimation  # 0.02s (50 Hz)

        # MG996R Actuator Parameters
        self.servo_kp = 18.0
        self.servo_kd = 0.6
        self.servo_deadband = 0.015       # ~5 microseconds pulse deadband in rad
        self.servo_max_velocity = 6.0     # rad/s (~0.17 s/60 deg)
        self.servo_max_torque = 1.0       # Nm (~10 kg.cm stall torque)

        # Joint Configuration (12-DOF: FL, FR, RL, RR: hip yaw, thigh pitch, knee pitch)
        # Nominal Spider Standing Stance (CAD calibrated):
        # Hips at 0.0 rad, thighs at 0.45 rad down, knees at 0.95 rad down
        self.default_dof_pos = np.array([
            0.0, 0.75, 0.5,  # FL hip, thigh, knee
            0.0, 0.75, 0.5,  # FR hip, thigh, knee
            0.0, 0.75, 0.5,  # RL hip, thigh, knee
            0.0, 0.75, 0.5,  # RR hip, thigh, knee
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

        # 12-DOF Action space: normalized [-1, 1] mapped to [default - scale, default + scale]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)

        # Observation space (45 dimensions):
        # - Projected gravity vector in base frame (3)
        # - Base angular velocity (IMU gyro) (3)
        # - Joint positions error from default (12)
        # - Joint velocities (12)
        # - Previous actions (12)
        # - Target velocity command [vx_cmd, vy_cmd, wz_cmd] (3)
        self.obs_dim = 3 + 3 + 12 + 12 + 12 + 3
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )

        self.prev_action = np.zeros(12, dtype=np.float32)
        self.last_torques = np.zeros(12, dtype=np.float32)
        self.commands = np.array([self.target_velocity, 0.0, 0.0], dtype=np.float32)

        # Sensor and Body IDs
        self.base_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        self.foot_site_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, f"{leg}_foot_site")
            for leg in ["fl", "fr", "rl", "rr"]
        ]

        self.viewer = None

    def _quat_to_rot_mat(self, quat):
        mat = np.zeros(9, dtype=np.float64)
        mujoco.mju_quat2Mat(mat, quat)
        return mat.reshape(3, 3)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.prev_action = np.zeros(12, dtype=np.float32)
        self.last_torques = np.zeros(12, dtype=np.float32)

        # Reset MuJoCo state
        mujoco.mj_resetData(self.model, self.data)

        # Base position: z = 0.12m with small random perturbation
        self.data.qpos[0] = 0.0  # x
        self.data.qpos[1] = 0.0  # y
        self.data.qpos[2] = 0.5 + self.np_random.uniform(-0.005, 0.005)  # z
        # Base quaternion: [w, x, y, z]
        self.data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])

        # Joints: default spider stance + small initial perturbation
        joint_jitter = self.np_random.uniform(-0.03, 0.03, size=12)
        self.data.qpos[7:19] = self.default_dof_pos + joint_jitter

        # Velocities: zero
        self.data.qvel[:] = 0.0

        # Step forward kinematics
        mujoco.mj_forward(self.model, self.data)

        # Command velocity
        self.commands = np.array([self.target_velocity, 0.0, 0.0], dtype=np.float32)

        obs = self._get_obs()
        info = {}
        return obs, info

    def _get_obs(self):
        # 1. Base orientation: projected gravity vector in body frame
        quat = self.data.qpos[3:7]  # [w, x, y, z]
        rot_mat = self._quat_to_rot_mat(quat)
        projected_gravity = -rot_mat[2, :]

        # 2. Base angular velocity in body frame (gyro)
        ang_vel_world = self.data.qvel[3:6]
        ang_vel_body = rot_mat.T @ ang_vel_world

        # 3. Joint positions error from default
        joint_pos = self.data.qpos[7:19].copy()
        joint_pos_err = (joint_pos - self.default_dof_pos).astype(np.float32)

        # 4. Joint velocities
        joint_vel = self.data.qvel[6:18].copy().astype(np.float32)

        # 5. Assemble observation vector (45 dims)
        obs = np.concatenate([
            projected_gravity.astype(np.float32),  # 3
            ang_vel_body.astype(np.float32),       # 3
            joint_pos_err,                         # 12
            joint_vel,                             # 12
            self.prev_action,                      # 12
            self.commands                          # 3
        ])
        return obs

    def step(self, action):
        self.step_count += 1
        action = np.clip(action, -1.0, 1.0)

        # Compute desired joint angles from policy action
        target_q = self.default_dof_pos + action * self.action_scale
        target_q = np.clip(target_q, self.dof_lower, self.dof_upper)

        accum_power = 0.0
        self.last_torques = np.zeros(12, dtype=np.float32)

        # Run physics at 500 Hz for decimation = 10 steps
        for _ in range(self.decimation):
            current_q = self.data.qpos[7:19]
            current_qd = self.data.qvel[6:18]

            # MG996R servo PD control with deadband and limits
            error = target_q - current_q
            active_mask = np.abs(error) > self.servo_deadband

            tau = np.zeros(12, dtype=np.float32)
            tau[active_mask] = (
                self.servo_kp * error[active_mask] - self.servo_kd * current_qd[active_mask]
            )

            # Velocity saturation / speed limit
            speed_ratio = np.abs(current_qd) / self.servo_max_velocity
            tau = tau * np.clip(1.0 - speed_ratio, 0.0, 1.0)

            # Torque saturation (MG996R ~1.0 Nm stall)
            tau = np.clip(tau, -self.servo_max_torque, self.servo_max_torque)

            # Apply controls
            self.data.ctrl[:] = target_q
            self.data.qfrc_applied[6:18] = tau
            self.last_torques = tau

            accum_power += np.sum(np.abs(tau * current_qd)) * self.physics_dt

            # Step MuJoCo physics
            mujoco.mj_step(self.model, self.data)

        # Calculate reward and termination
        obs = self._get_obs()
        reward, reward_info = self._compute_reward(action, accum_power)
        terminated = self._is_terminated()
        truncated = self.step_count >= self.max_episode_steps

        self.prev_action = action.copy()

        info = {"reward_details": reward_info}
        return obs, reward, terminated, truncated, info

    def _compute_reward(self, action, mechanical_power):
        base_linvel = self.data.qvel[0:3]
        vx, vy = base_linvel[0], base_linvel[1]

        # 1. Forward velocity tracking reward
        target_vx = self.commands[0]
        r_forward = np.exp(-1.5 * (vx - target_vx)**2)

        # 2. Lateral drift penalty
        r_lateral = -1.0 * (vy**2)

        # 3. Base orientation penalty (keep chassis flat)
        quat = self.data.qpos[3:7]
        rot_mat = self._quat_to_rot_mat(quat)
        proj_gravity_z = rot_mat[2, 2]
        r_orientation = -2.0 * ((1.0 - proj_gravity_z)**2)

        # 4. Action smoothness / rate-of-change penalty
        action_diff = action - self.prev_action
        r_action_rate = -0.05 * np.sum(action_diff**2)

        # 5. Mechanical power / torque penalty
        r_energy = -0.02 * mechanical_power

        # 6. Base height maintenance (target ~0.12m for spider stance)
        base_z = self.data.qpos[2]
        r_height = -5.0 * ((base_z - 0.12)**2)

        # 7. Alive bonus
        r_alive = 0.5

        total_reward = (
            r_forward
            + r_lateral
            + r_orientation
            + r_action_rate
            + r_energy
            + r_height
            + r_alive
        )

        reward_info = {
            "r_forward": float(r_forward),
            "r_lateral": float(r_lateral),
            "r_orientation": float(r_orientation),
            "r_action_rate": float(r_action_rate),
            "r_energy": float(r_energy),
            "r_height": float(r_height),
        }
        return float(total_reward), reward_info

    def _is_terminated(self):
        base_z = self.data.qpos[2]
        if base_z < 0.05:
            return True

        quat = self.data.qpos[3:7]
        rot_mat = self._quat_to_rot_mat(quat)
        proj_gravity_z = rot_mat[2, 2]
        if proj_gravity_z < 0.60:
            return True

        return False

    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                import mujoco.viewer
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.sync()
        elif self.render_mode == "rgb_array":
            renderer = mujoco.Renderer(self.model, 480, 640)
            renderer.update_scene(self.data)
            return renderer.render()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
