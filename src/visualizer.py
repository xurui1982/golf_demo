"""可视化模块 - 生成骨骼叠加视频和图表"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple, Dict
from pathlib import Path

from .pose_analyzer import PoseResult, PoseLandmark
from .ball_tracker import BallPosition
from .swing_detector import SwingPhase


# 骨骼连接定义
POSE_CONNECTIONS = [
    # 躯干
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    # 左臂
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    # 右臂
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    # 左腿
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    # 右腿
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
]

# 颜色定义 (BGR)
COLORS = {
    "skeleton": (0, 255, 0),      # 绿色
    "joint": (0, 255, 255),       # 黄色
    "ball": (0, 0, 255),          # 红色
    "ball_trail": (255, 100, 100), # 浅蓝
    "text": (255, 255, 255),      # 白色
    "phase": (255, 200, 0),       # 青色
}


class Visualizer:
    """可视化生成器"""

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height

    def draw_skeleton(
        self,
        frame: np.ndarray,
        pose_result: PoseResult,
        color: Tuple[int, int, int] = COLORS["skeleton"],
        joint_color: Tuple[int, int, int] = COLORS["joint"],
        thickness: int = 2,
        joint_radius: int = 4
    ) -> np.ndarray:
        """
        在帧上绘制骨骼

        Args:
            frame: 输入图像
            pose_result: 姿态结果
            color: 骨骼颜色
            joint_color: 关节颜色
            thickness: 线条粗细
            joint_radius: 关节点半径

        Returns:
            绘制后的图像
        """
        output = frame.copy()
        landmarks = pose_result.landmarks

        # 绘制骨骼连接
        for start_name, end_name in POSE_CONNECTIONS:
            start = landmarks.get(start_name)
            end = landmarks.get(end_name)

            if start and end and start.visibility > 0.5 and end.visibility > 0.5:
                start_point = self._to_pixel(start)
                end_point = self._to_pixel(end)
                cv2.line(output, start_point, end_point, color, thickness)

        # 绘制关节点
        for name, landmark in landmarks.items():
            if landmark.visibility > 0.5:
                point = self._to_pixel(landmark)
                cv2.circle(output, point, joint_radius, joint_color, -1)

        return output

    def draw_ball(
        self,
        frame: np.ndarray,
        ball_position: BallPosition,
        color: Tuple[int, int, int] = COLORS["ball"],
        thickness: int = 2
    ) -> np.ndarray:
        """在帧上标记球的位置"""
        output = frame.copy()
        cv2.circle(
            output,
            (ball_position.x, ball_position.y),
            ball_position.radius + 5,
            color,
            thickness
        )
        return output

    def draw_ball_trail(
        self,
        frame: np.ndarray,
        positions: List[BallPosition],
        max_trail: int = 20,
        color: Tuple[int, int, int] = COLORS["ball_trail"]
    ) -> np.ndarray:
        """绘制球的轨迹线"""
        output = frame.copy()

        if len(positions) < 2:
            return output

        # 只显示最近的轨迹
        recent = positions[-max_trail:]

        points = [(p.x, p.y) for p in recent]
        for i in range(1, len(points)):
            # 渐变透明度
            alpha = i / len(points)
            thickness = max(1, int(3 * alpha))
            cv2.line(output, points[i-1], points[i], color, thickness)

        return output

    def draw_phase_label(
        self,
        frame: np.ndarray,
        phase: SwingPhase,
        position: Tuple[int, int] = (30, 50),
        font_scale: float = 1.2
    ) -> np.ndarray:
        """在帧上显示当前阶段"""
        output = frame.copy()

        # 阶段名称映射
        phase_names = {
            SwingPhase.SETUP: "Setup",
            SwingPhase.BACKSWING: "Backswing",
            SwingPhase.TOP: "Top",
            SwingPhase.DOWNSWING: "Downswing",
            SwingPhase.IMPACT: "Impact",
            SwingPhase.FOLLOW_THROUGH: "Follow Through",
            SwingPhase.FINISH: "Finish",
        }

        text = phase_names.get(phase, "")

        # 绘制背景
        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
        cv2.rectangle(
            output,
            (position[0] - 10, position[1] - h - 10),
            (position[0] + w + 10, position[1] + 10),
            (0, 0, 0),
            -1
        )

        # 绘制文字
        cv2.putText(
            output,
            text,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            COLORS["phase"],
            2
        )

        return output

    def draw_metrics(
        self,
        frame: np.ndarray,
        metrics: Dict[str, str],
        start_y: int = 100
    ) -> np.ndarray:
        """在帧上显示指标数据"""
        output = frame.copy()
        y = start_y

        for name, value in metrics.items():
            text = f"{name}: {value}"
            cv2.putText(
                output,
                text,
                (30, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                COLORS["text"],
                2
            )
            y += 30

        return output

    def _to_pixel(self, landmark: PoseLandmark) -> Tuple[int, int]:
        """将归一化坐标转换为像素坐标"""
        return (
            int(landmark.x * self.width),
            int(landmark.y * self.height)
        )


def generate_annotated_video(
    input_path: str,
    output_path: str,
    pose_results: List[PoseResult],
    ball_positions: Optional[List[BallPosition]] = None,
    phase_info: Optional[Dict[int, SwingPhase]] = None,
    fps: Optional[float] = None
):
    """
    生成带标注的视频

    Args:
        input_path: 输入视频路径
        output_path: 输出视频路径
        pose_results: 姿态结果列表
        ball_positions: 球位置列表
        phase_info: 帧到阶段的映射
        fps: 输出帧率（None 则使用原始帧率）
    """
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_fps = cap.get(cv2.CAP_PROP_FPS)

    output_fps = fps or original_fps
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, output_fps, (width, height))

    visualizer = Visualizer(width, height)

    # 建立帧索引到姿态结果的映射
    pose_map = {pr.frame_index: pr for pr in pose_results}

    # 建立帧索引到球位置的映射
    ball_map = {}
    if ball_positions:
        for bp in ball_positions:
            ball_map[bp.frame_index] = bp

    # 收集球轨迹
    ball_trail = []

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 绘制骨骼
        if frame_idx in pose_map:
            frame = visualizer.draw_skeleton(frame, pose_map[frame_idx])

        # 绘制球和轨迹
        if frame_idx in ball_map:
            ball_trail.append(ball_map[frame_idx])
            frame = visualizer.draw_ball_trail(frame, ball_trail)
            frame = visualizer.draw_ball(frame, ball_map[frame_idx])

        # 绘制阶段标签
        if phase_info and frame_idx in phase_info:
            frame = visualizer.draw_phase_label(frame, phase_info[frame_idx])

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()


def save_keyframe(
    frame: np.ndarray,
    pose_result: Optional[PoseResult],
    output_path: str,
    phase_name: str = ""
):
    """保存关键帧图片"""
    width, height = frame.shape[1], frame.shape[0]
    visualizer = Visualizer(width, height)

    output = frame.copy()

    if pose_result:
        output = visualizer.draw_skeleton(output, pose_result)

    if phase_name:
        output = visualizer.draw_phase_label(
            output,
            SwingPhase(phase_name) if phase_name in [p.value for p in SwingPhase] else SwingPhase.SETUP,
            position=(30, height - 30)
        )

    cv2.imwrite(output_path, output)


def plot_swing_metrics(
    pose_results: List[PoseResult],
    output_path: str,
    title: str = "Swing Metrics"
):
    """绘制挥杆指标曲线图"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    timestamps = [pr.timestamp for pr in pose_results]

    # 1. 手腕高度
    wrist_y = []
    for pr in pose_results:
        left = pr.landmarks.get("left_wrist")
        right = pr.landmarks.get("right_wrist")
        if left and right:
            wrist_y.append(1 - (left.y + right.y) / 2)  # 反转 y 轴
        else:
            wrist_y.append(None)

    axes[0, 0].plot(timestamps, wrist_y, 'b-', linewidth=2)
    axes[0, 0].set_title("Wrist Height")
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("Height (normalized)")
    axes[0, 0].grid(True)

    # 2. 肩部旋转（简化：肩部连线角度）
    shoulder_angle = []
    for pr in pose_results:
        left = pr.landmarks.get("left_shoulder")
        right = pr.landmarks.get("right_shoulder")
        if left and right:
            angle = np.degrees(np.arctan2(right.y - left.y, right.x - left.x))
            shoulder_angle.append(angle)
        else:
            shoulder_angle.append(None)

    axes[0, 1].plot(timestamps, shoulder_angle, 'r-', linewidth=2)
    axes[0, 1].set_title("Shoulder Rotation")
    axes[0, 1].set_xlabel("Time (s)")
    axes[0, 1].set_ylabel("Angle (degrees)")
    axes[0, 1].grid(True)

    # 3. 髋部旋转
    hip_angle = []
    for pr in pose_results:
        left = pr.landmarks.get("left_hip")
        right = pr.landmarks.get("right_hip")
        if left and right:
            angle = np.degrees(np.arctan2(right.y - left.y, right.x - left.x))
            hip_angle.append(angle)
        else:
            hip_angle.append(None)

    axes[1, 0].plot(timestamps, hip_angle, 'g-', linewidth=2)
    axes[1, 0].set_title("Hip Rotation")
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].set_ylabel("Angle (degrees)")
    axes[1, 0].grid(True)

    # 4. X-Factor (肩髋分离)
    x_factor = []
    for i in range(len(pose_results)):
        if shoulder_angle[i] is not None and hip_angle[i] is not None:
            x_factor.append(abs(shoulder_angle[i] - hip_angle[i]))
        else:
            x_factor.append(None)

    axes[1, 1].plot(timestamps, x_factor, 'm-', linewidth=2)
    axes[1, 1].set_title("X-Factor (Shoulder-Hip Separation)")
    axes[1, 1].set_xlabel("Time (s)")
    axes[1, 1].set_ylabel("Angle (degrees)")
    axes[1, 1].grid(True)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
