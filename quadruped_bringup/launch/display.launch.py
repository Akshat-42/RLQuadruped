import os
from ament_index_python.packages import get_packages_with_prefixes
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_quadruped_description = FindPackageShare('quadruped_description')

    gui = LaunchConfiguration('gui')
    declare_gui = DeclareLaunchArgument(
        'gui',
        default_value='true',
        description='Whether to use joint_state_publisher_gui if available',
    )

    # Check if joint_state_publisher_gui package is installed
    has_jsp_gui = 'joint_state_publisher_gui' in get_packages_with_prefixes()

    # Process Xacro
    robot_description_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name='xacro')]),
            ' ',
            PathJoinSubstitution([pkg_quadruped_description, 'urdf', 'quadruped.urdf.xacro']),
        ]
    )
    robot_description = {
        'robot_description': ParameterValue(robot_description_content, value_type=str)
    }

    # Robot State Publisher
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='both',
        parameters=[robot_description],
    )

    nodes = [declare_gui, robot_state_publisher_node]

    # Joint State Publisher (GUI or standard fallback)
    if has_jsp_gui:
        nodes.append(
            Node(
                package='joint_state_publisher_gui',
                executable='joint_state_publisher_gui',
                name='joint_state_publisher_gui',
                output='screen',
                condition=IfCondition(gui),
            )
        )
        nodes.append(
            Node(
                package='joint_state_publisher',
                executable='joint_state_publisher',
                name='joint_state_publisher',
                output='screen',
                condition=UnlessCondition(gui),
            )
        )
    else:
        nodes.append(
            Node(
                package='joint_state_publisher',
                executable='joint_state_publisher',
                name='joint_state_publisher',
                output='screen',
            )
        )

    # RViz2
    rviz_config_file = PathJoinSubstitution([FindPackageShare('quadruped_bringup'), 'config', 'display.rviz'])
    nodes.append(
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', rviz_config_file],
        )
    )

    return LaunchDescription(nodes)
