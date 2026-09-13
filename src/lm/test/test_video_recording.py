"""Recording and presentation checks without starting a robot or ROS nodes."""

from pathlib import Path
from unittest.mock import Mock
import runpy

import cv2
import mujoco
import numpy as np
import pytest
from launch import LaunchContext
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch_ros.utilities import evaluate_parameters, normalize_parameters

from lm.video_recording import VideoRecording

SRC = Path(__file__).resolve().parents[2]
SCENES = SRC / "crl-humanoid-ros/crl_humanoid_commons/data/robots/g1_description"


@pytest.fixture(autouse=True)
def ros_logs_in_temp(monkeypatch, tmp_path):
    monkeypatch.setenv("ROS_LOG_DIR", str(tmp_path / "ros_logs"))


@pytest.mark.parametrize("kind", ["box", "bucket"])
def test_white_scene_changes_presentation_not_object_physics(kind):
    normal = mujoco.MjModel.from_xml_path(str(SCENES / f"scene_crl_with_{kind}.xml"))
    video = mujoco.MjModel.from_xml_path(str(SCENES / f"scene_crl_with_{kind}_video.xml"))
    for field in ("body_mass", "body_inertia", "geom_size", "geom_friction", "geom_solref", "geom_solimp",
                  "geom_condim", "geom_contype", "geom_conaffinity", "jnt_range"):
        np.testing.assert_array_equal(getattr(normal, field), getattr(video, field))
    assert video.camera("experiment_video").mode == mujoco.mjtCamLight.mjCAMLIGHT_FIXED
    data = mujoco.MjData(video)
    mujoco.mj_forward(video, data)
    camera = data.cam_xpos.copy()
    data.qpos[0] = -2.
    mujoco.mj_forward(video, data)
    np.testing.assert_array_equal(data.cam_xpos, camera)
    ground_start = video.tex_adr[video.texture("groundplane").id]
    assert np.all(video.tex_data[:ground_start] == 255)  # white sky remains
    assert np.min(video.tex_data[ground_start:]) < 255  # grey grid edges
    assert np.max(video.tex_data[ground_start:]) == 255  # white tile interiors


def test_recording_never_overwrites_and_preserves_wall_time(tmp_path):
    path = tmp_path / "demo.mp4"
    recording = VideoRecording(path, 10., 64, 48)
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    recording.write(frame, now=10.)
    recording.write(frame, now=10.05)  # same output slot
    recording.write(frame, now=10.35)  # two late slots filled
    assert recording.frames == 4
    recording.close()
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        VideoRecording(path, 10., 64, 48)
    assert path.read_bytes() == original
    reader = cv2.VideoCapture(str(path))
    assert reader.isOpened()
    assert reader.get(cv2.CAP_PROP_FRAME_COUNT) == 4
    assert reader.get(cv2.CAP_PROP_FPS) == 10.
    reader.release()


def test_recording_waits_for_both_state_streams_and_can_skip_ros_images():
    from lm.scene_camera import SceneCameraNode
    from threading import Lock
    node = SceneCameraNode.__new__(SceneCameraNode)
    node._recording = Mock()
    node._have_monitor = True
    node._have_object_pose = False
    node._renderer = Mock()
    node._publish_mujoco_image()
    node._renderer.render.assert_not_called()
    node._recording.write.assert_not_called()


@pytest.mark.parametrize("kind", ["box", "bucket"])
def test_video_mirrors_robot_goal_without_changing_actual_robot_or_object(kind):
    from geometry_msgs.msg import PoseArray, Pose
    from lm.scene_camera import SceneCameraNode
    from threading import Lock
    node = SceneCameraNode.__new__(SceneCameraNode)
    node._model = mujoco.MjModel.from_xml_path(str(SCENES / f"scene_crl_with_{kind}_video.xml"))
    node._data = mujoco.MjData(node._model)
    node._lock = Lock()
    node._goal_mocap_ids = [int(node._model.body(f"target_kp_{i:02d}").mocapid[0]) for i in range(14)]
    original_qpos = node._data.qpos.copy()
    object_id = node._model.body("target_object").mocapid[0]
    original_object_target = node._data.mocap_pos[object_id].copy()
    msg = PoseArray()
    for i in range(15):  # 14 robot poses followed by one object pose
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = float(i), .2, .8
        msg.poses.append(pose)
    node._on_robot_goal(msg)
    np.testing.assert_allclose(node._data.mocap_pos[node._goal_mocap_ids], [[float(i), .2, .8] for i in range(14)])
    np.testing.assert_array_equal(node._data.qpos, original_qpos)
    np.testing.assert_array_equal(node._data.mocap_pos[object_id], original_object_target)
    # New previews update the same points; malformed/missing poses clear them.
    msg.poses[0].position.x = 3.
    msg.poses[1].position.x = float("nan")
    msg.poses = msg.poses[:2]
    node._on_robot_goal(msg)
    np.testing.assert_allclose(node._data.mocap_pos[node._goal_mocap_ids[0]], [3., .2, .8])
    np.testing.assert_allclose(node._data.mocap_pos[node._goal_mocap_ids[1:]], [[0., 0., -10.]] * 13)


@pytest.mark.parametrize("kind", ["box", "bucket"])
def test_video_launch_is_sim_only_and_does_not_start_a_task(monkeypatch, kind):
    import ament_index_python.packages
    original = ament_index_python.packages.get_package_share_directory
    monkeypatch.setattr(ament_index_python.packages, "get_package_share_directory",
                        lambda name: str(SRC / "lm") if name == "lm" else original(name))
    module = runpy.run_path(str(SRC / "lm/launch/vlm_video_launch.py"))
    context = LaunchContext()
    context.launch_configurations["scene_object"] = kind
    entities = module["generate_launch_description"]().entities
    for action in entities:
        if isinstance(action, DeclareLaunchArgument):
            action.execute(context)
    assert context.launch_configurations["mode"] == "sim"
    assert context.launch_configurations["scene_object"] == kind
    assert context.launch_configurations["sim_scene_xml"].endswith(f"scene_crl_with_{kind}_video.xml")
    assert context.launch_configurations["camera_robot_xml"].endswith(f"scene_crl_with_{kind}_video.xml")
    assert context.launch_configurations["camera_object_joint_name"] == f"{kind}_freejoint"
    assert context.launch_configurations["task_text"].startswith(f"Pick up the {kind}")
    assert context.launch_configurations["initial_root_pos"] == "[-2.0, 0.0, 0.8]"
    assert context.launch_configurations["camera_name"] == "experiment_video"
    assert context.launch_configurations["record_video"] == "true"
    assert context.launch_configurations["video_width"] == "1920"
    assert context.launch_configurations["video_height"] == "1080"
    assert context.launch_configurations["video_show_robot_goal"] == "true"
    assert not any(getattr(action, "node_executable", "") == "vlm_client" for action in entities)
    include = next(action for action in entities if isinstance(action, IncludeLaunchDescription))
    include.launch_description_source.get_launch_description(context)
    assert include.launch_description_source.location.endswith("vlm_experiment_launch.py")


@pytest.mark.parametrize("kind", ["box", "bucket"])
def test_sim_root_and_fixed_camera_overrides_reach_all_scene_consumers(monkeypatch, kind):
    path = SRC / "crl-humanoid-ros/crl_g1_goalcontroller_py/crl_g1_goalcontroller/launch/g1_keyframe_sim.py"
    module = runpy.run_path(str(path))
    context = LaunchContext()
    context.launch_configurations.update(initial_root_pos="[-2.0, 0.0, 0.8]", scene_object=kind,
                                         sim_scene_xml=f"g1_description/scene_crl_with_{kind}_video.xml",
                                         monitor_camera_name="experiment_video")
    for action in module["generate_launch_description"]().entities:
        if isinstance(action, DeclareLaunchArgument):
            action.execute(context)
    launch_nodes = module["_launch_nodes"]
    monkeypatch.setitem(launch_nodes.__globals__, "Node", lambda **kwargs: kwargs)
    simulator, monitor, controller = launch_nodes(context)
    def params(node):
        result = {}
        for item in node["parameters"]:
            if isinstance(item, dict):
                result.update(evaluate_parameters(context, normalize_parameters([item]))[0])
        return result
    assert list(params(simulator)["initial_root_pos"]) == [-2., 0., .8]
    assert params(simulator)["robot_xml_file"] == params(monitor)["robot_xml_file"]
    assert params(simulator)["object_joint_name"] == f"{kind}_freejoint"
    if kind == "bucket":
        assert params(monitor)["object_joint_name"] == "bucket_freejoint"
    assert params(monitor)["camera_name"] == "experiment_video"
    assert params(controller)["robot_xml"].endswith(f"scene_crl_with_{kind}_video.xml")
