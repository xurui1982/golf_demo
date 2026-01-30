"""姿态分析模块 - 使用 MediaPipe 提取人体关键点"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Optional, List, Dict, Any
from dataclasses import dataclass


@dataclass
class PoseLandmark:
    """单个关键点"""
    x: float  # 归一化 x 坐标 (0-1)
    y: float  # 归一化 y 坐标 (0-1)
    z: float  # 深度（相对于髋部）
    visibility: float  # 可见度 (0-1)


@dataclass
class PoseResult:
    """单帧姿态分析结果"""
    frame_index: int
    timestamp: float  # 秒
    landmarks: Dict[str, PoseLandmark]  # 关键点字典
    raw_landmarks: Any  # MediaPipe 原始数据


# MediaPipe 关键点名称映射
LANDMARK_NAMES = {
    0: "nose",
    1: "left_eye_inner",
    2: "left_eye",
    3: "left_eye_outer",
    4: "right_eye_inner",
    5: "right_eye",
    6: "right_eye_outer",
    7: "left_ear",
    8: "right_ear",
    9: "mouth_left",
    10: "mouth_right",
    11: "left_shoulder",
    12: "right_shoulder",
    13: "left_elbow",
    14: "right_elbow",
    15: "left_wrist",
    16: "right_wrist",
    17: "left_pinky",
    18: "right_pinky",
    19: "left_index",
    20: "right_index",
    21: "left_thumb",
    22: "right_thumb",
    23: "left_hip",
    24: "right_hip",
    25: "left_knee",
    26: "right_knee",
    27: "left_ankle",
    28: "right_ankle",
    29: "left_heel",
    30: "right_heel",
    31: "left_foot_index",
    32: "right_foot_index",
}


class PoseAnalyzer:
    """使用 MediaPipe 进行姿态估计"""

    def __init__(
        self,
        static_image_mode: bool = False,
        model_complexity: int = 2,  # 0, 1, 2 - 2 最精确
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5
    ):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    def analyze_frame(
        self,
        frame: np.ndarray,
        frame_index: int = 0,
        fps: float = 30.0
    ) -> Optional[PoseResult]:
        """
        分析单帧图像

        Args:
            frame: BGR 格式的图像
            frame_index: 帧索引
            fps: 帧率（用于计算时间戳）

        Returns:
            PoseResult 或 None（如果检测失败）
        """
        # 转换为 RGB（MediaPipe 要求）
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 运行姿态估计
        results = self.pose.process(rgb_frame)

        if not results.pose_landmarks:
            return None

        # 提取关键点
        landmarks = {}
        for idx, landmark in enumerate(results.pose_landmarks.landmark):
            name = LANDMARK_NAMES.get(idx, f"landmark_{idx}")
            landmarks[name] = PoseLandmark(
                x=landmark.x,
                y=landmark.y,
                z=landmark.z,
                visibility=landmark.visibility
            )

        return PoseResult(
            frame_index=frame_index,
            timestamp=frame_index / fps,
            landmarks=landmarks,
            raw_landmarks=results.pose_landmarks
        )

    def analyze_video(
        self,
        video_processor,
        step: int = 1,
        progress_callback=None
    ) -> List[PoseResult]:
        """
        分析整个视频

        Args:
            video_processor: VideoProcessor 实例
            step: 帧间隔
            progress_callback: 进度回调函数 (current, total)

        Returns:
            PoseResult 列表
        """
        results = []
        total = video_processor.frame_count // step

        for i, (frame_idx, frame) in enumerate(video_processor.extract_frames(step=step)):
            pose_result = self.analyze_frame(frame, frame_idx, video_processor.fps)
            if pose_result:
                results.append(pose_result)

            if progress_callback:
                progress_callback(i + 1, total)

        return results

    def close(self):
        """释放资源"""
        self.pose.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def calculate_angle(p1: PoseLandmark, p2: PoseLandmark, p3: PoseLandmark) -> float:
    """
    计算三点形成的角度（以 p2 为顶点）

    Returns:
        角度（度数）
    """
    v1 = np.array([p1.x - p2.x, p1.y - p2.y])
    v2 = np.array([p3.x - p2.x, p3.y - p2.y])

    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))

    return np.degrees(angle)


def get_body_center(landmarks: Dict[str, PoseLandmark]) -> tuple:
    """获取身体中心点（髋部中点）"""
    left_hip = landmarks.get("left_hip")
    right_hip = landmarks.get("right_hip")

    if left_hip and right_hip:
        return (
            (left_hip.x + right_hip.x) / 2,
            (left_hip.y + right_hip.y) / 2
        )
    return None
