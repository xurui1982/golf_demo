"""指标计算模块 - 计算挥杆相关指标"""

import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass

from .pose_analyzer import PoseResult, PoseLandmark, calculate_angle


@dataclass
class SwingMetrics:
    """挥杆指标"""
    # 姿态指标
    spine_angle_setup: Optional[float] = None    # 准备时脊柱角度
    spine_angle_impact: Optional[float] = None   # 击球时脊柱角度
    spine_angle_stability: Optional[float] = None  # 脊柱角度稳定性（标准差）

    x_factor_top: Optional[float] = None  # 顶点时 X-Factor
    x_factor_max: Optional[float] = None  # 最大 X-Factor

    head_movement: Optional[float] = None  # 头部移动量（归一化）

    left_arm_angle_top: Optional[float] = None  # 顶点时左臂角度

    # 节奏指标
    backswing_time: Optional[float] = None  # 上杆时间
    downswing_time: Optional[float] = None  # 下杆时间
    tempo_ratio: Optional[float] = None  # 节奏比

    # 球数据
    ball_speed: Optional[float] = None  # 球初速（像素/秒）
    launch_angle: Optional[float] = None  # 起飞角度
    launch_direction: Optional[float] = None  # 起飞方向


class MetricsCalculator:
    """指标计算器"""

    def __init__(self, fps: float = 240.0):
        self.fps = fps

    def calculate_all(
        self,
        pose_results: List[PoseResult],
        phases_frames: Dict[str, int],  # 各阶段关键帧
        ball_speed: Optional[float] = None,
        launch_angle: Optional[float] = None,
        launch_direction: Optional[float] = None
    ) -> SwingMetrics:
        """计算所有指标"""

        metrics = SwingMetrics()

        # 获取各阶段的姿态
        setup_pose = self._get_pose_at_frame(pose_results, phases_frames.get("setup"))
        top_pose = self._get_pose_at_frame(pose_results, phases_frames.get("top"))
        impact_pose = self._get_pose_at_frame(pose_results, phases_frames.get("impact"))

        # 计算脊柱角度
        if setup_pose:
            metrics.spine_angle_setup = self._calculate_spine_angle(setup_pose)
        if impact_pose:
            metrics.spine_angle_impact = self._calculate_spine_angle(impact_pose)

        # 计算脊柱角度稳定性
        spine_angles = [
            self._calculate_spine_angle(pr)
            for pr in pose_results
            if self._calculate_spine_angle(pr) is not None
        ]
        if spine_angles:
            metrics.spine_angle_stability = np.std(spine_angles)

        # 计算 X-Factor
        x_factors = []
        for pr in pose_results:
            xf = self._calculate_x_factor(pr)
            if xf is not None:
                x_factors.append((pr.frame_index, xf))

        if x_factors:
            metrics.x_factor_max = max(xf for _, xf in x_factors)

        if top_pose:
            metrics.x_factor_top = self._calculate_x_factor(top_pose)
            metrics.left_arm_angle_top = self._calculate_left_arm_angle(top_pose)

        # 计算头部移动
        metrics.head_movement = self._calculate_head_movement(pose_results)

        # 节奏指标
        if "backswing" in phases_frames and "top" in phases_frames:
            backswing_start = phases_frames["backswing"]
            top_frame = phases_frames["top"]
            metrics.backswing_time = (top_frame - backswing_start) / self.fps

        if "top" in phases_frames and "impact" in phases_frames:
            top_frame = phases_frames["top"]
            impact_frame = phases_frames["impact"]
            metrics.downswing_time = (impact_frame - top_frame) / self.fps

        if metrics.backswing_time and metrics.downswing_time:
            metrics.tempo_ratio = metrics.backswing_time / metrics.downswing_time

        # 球数据
        metrics.ball_speed = ball_speed
        metrics.launch_angle = launch_angle
        metrics.launch_direction = launch_direction

        return metrics

    def _get_pose_at_frame(
        self,
        pose_results: List[PoseResult],
        frame: Optional[int]
    ) -> Optional[PoseResult]:
        """获取指定帧的姿态"""
        if frame is None:
            return None

        for pr in pose_results:
            if pr.frame_index == frame:
                return pr
        return None

    def _calculate_spine_angle(self, pose: PoseResult) -> Optional[float]:
        """
        计算脊柱角度（前倾角度）

        使用肩部中点和髋部中点的连线与垂直线的夹角
        """
        landmarks = pose.landmarks

        left_shoulder = landmarks.get("left_shoulder")
        right_shoulder = landmarks.get("right_shoulder")
        left_hip = landmarks.get("left_hip")
        right_hip = landmarks.get("right_hip")

        if not all([left_shoulder, right_shoulder, left_hip, right_hip]):
            return None

        # 肩部中点
        shoulder_mid = (
            (left_shoulder.x + right_shoulder.x) / 2,
            (left_shoulder.y + right_shoulder.y) / 2
        )

        # 髋部中点
        hip_mid = (
            (left_hip.x + right_hip.x) / 2,
            (left_hip.y + right_hip.y) / 2
        )

        # 脊柱向量
        spine_vec = (shoulder_mid[0] - hip_mid[0], shoulder_mid[1] - hip_mid[1])

        # 与垂直向量的夹角
        vertical = (0, -1)  # 向上
        dot = spine_vec[0] * vertical[0] + spine_vec[1] * vertical[1]
        mag1 = np.sqrt(spine_vec[0]**2 + spine_vec[1]**2)
        mag2 = 1.0

        if mag1 == 0:
            return None

        cos_angle = dot / (mag1 * mag2)
        angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

        return angle

    def _calculate_x_factor(self, pose: PoseResult) -> Optional[float]:
        """
        计算 X-Factor（髋肩分离角度）

        肩部连线与髋部连线的角度差
        """
        landmarks = pose.landmarks

        left_shoulder = landmarks.get("left_shoulder")
        right_shoulder = landmarks.get("right_shoulder")
        left_hip = landmarks.get("left_hip")
        right_hip = landmarks.get("right_hip")

        if not all([left_shoulder, right_shoulder, left_hip, right_hip]):
            return None

        # 肩部角度
        shoulder_angle = np.degrees(np.arctan2(
            right_shoulder.y - left_shoulder.y,
            right_shoulder.x - left_shoulder.x
        ))

        # 髋部角度
        hip_angle = np.degrees(np.arctan2(
            right_hip.y - left_hip.y,
            right_hip.x - left_hip.x
        ))

        return abs(shoulder_angle - hip_angle)

    def _calculate_left_arm_angle(self, pose: PoseResult) -> Optional[float]:
        """计算左臂角度（肩-肘-腕的角度）"""
        landmarks = pose.landmarks

        shoulder = landmarks.get("left_shoulder")
        elbow = landmarks.get("left_elbow")
        wrist = landmarks.get("left_wrist")

        if not all([shoulder, elbow, wrist]):
            return None

        return calculate_angle(shoulder, elbow, wrist)

    def _calculate_head_movement(
        self,
        pose_results: List[PoseResult]
    ) -> Optional[float]:
        """
        计算头部移动量

        返回鼻子位置的最大移动范围（归一化坐标）
        """
        nose_positions = []
        for pr in pose_results:
            nose = pr.landmarks.get("nose")
            if nose and nose.visibility > 0.5:
                nose_positions.append((nose.x, nose.y))

        if len(nose_positions) < 2:
            return None

        x_coords = [p[0] for p in nose_positions]
        y_coords = [p[1] for p in nose_positions]

        x_range = max(x_coords) - min(x_coords)
        y_range = max(y_coords) - min(y_coords)

        return np.sqrt(x_range**2 + y_range**2)


def format_metrics(metrics: SwingMetrics) -> Dict[str, str]:
    """格式化指标为可显示的字符串"""
    result = {}

    if metrics.spine_angle_setup is not None:
        result["Spine Angle (Setup)"] = f"{metrics.spine_angle_setup:.1f}°"

    if metrics.spine_angle_impact is not None:
        result["Spine Angle (Impact)"] = f"{metrics.spine_angle_impact:.1f}°"

    if metrics.x_factor_top is not None:
        result["X-Factor (Top)"] = f"{metrics.x_factor_top:.1f}°"

    if metrics.x_factor_max is not None:
        result["X-Factor (Max)"] = f"{metrics.x_factor_max:.1f}°"

    if metrics.left_arm_angle_top is not None:
        result["Left Arm Angle (Top)"] = f"{metrics.left_arm_angle_top:.1f}°"

    if metrics.head_movement is not None:
        result["Head Movement"] = f"{metrics.head_movement:.3f}"

    if metrics.backswing_time is not None:
        result["Backswing Time"] = f"{metrics.backswing_time:.2f}s"

    if metrics.downswing_time is not None:
        result["Downswing Time"] = f"{metrics.downswing_time:.2f}s"

    if metrics.tempo_ratio is not None:
        result["Tempo Ratio"] = f"{metrics.tempo_ratio:.1f}:1"

    if metrics.launch_angle is not None:
        result["Launch Angle"] = f"{metrics.launch_angle:.1f}°"

    if metrics.launch_direction is not None:
        direction = "Right" if metrics.launch_direction > 0 else "Left"
        result["Launch Direction"] = f"{abs(metrics.launch_direction):.1f}° {direction}"

    return result
