import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, Shutdown
from launch.substitutions import (
    Command,
    FindExecutable,
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue, ParameterFile
from launch_ros.substitutions import FindPackageShare


def launch_setup(context, *args, **kwargs):
    pkg_quadruped_description = FindPackageShare("quadruped_description")
    pkg_quadruped_bringup = FindPackageShare("quadruped_bringup")

    # Build robot description from xacro
    robot_description_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            PathJoinSubstitution([pkg_quadruped_description, "urdf", "quadruped.urdf.xacro"]),
        ]
    )

    robot_description_str = robot_description_content.perform(context)
    robot_description = {"robot_description": ParameterValue(value=robot_description_str, value_type=str)}

    controllers_file = PathJoinSubstitution([pkg_quadruped_bringup, "config", "ros2_controllers.yaml"])

    nodes = []

    # Robot state publisher
    nodes.append(
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            output="both",
            parameters=[robot_description, {"use_sim_time": True}],
        )
    )

    # MuJoCo ros2_control simulation node
    nodes.append(
        Node(
            package="mujoco_ros2_control",
            executable="ros2_control_node",
            emulate_tty=True,
            output="both",
            parameters=[
                robot_description,
                {"use_sim_time": True},
                ParameterFile(controllers_file),
            ],
            on_exit=Shutdown(),
        )
    )

    # Controller Spawners
    controllers = ["joint_state_broadcaster", "leg_position_controller"]
    for controller in controllers:
        nodes.append(
            Node(
                package="controller_manager",
                executable="spawner",
                arguments=[controller, "--param-file", controllers_file],
                output="both",
            )
        )

    return nodes


def generate_launch_description():
    headless = DeclareLaunchArgument(
        "headless",
        default_value="false",
        description="Run simulation without visualization window",
    )

    return LaunchDescription(
        [
            headless,
            OpaqueFunction(function=launch_setup),
        ]
    )

