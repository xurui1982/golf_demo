"""球追踪模块 - 检测和追踪高尔夫球"""

import cv2
import numpy as np
from typing import Optional, List, Tuple
from dataclasses import dataclass


@dataclass
class BallPosition:
    """球的位置信息"""
    frame_index: int
    timestamp: float
    x: int  # 像素坐标
    y: int
    radius: int  # 检测到的球半径
    confidence: float  # 置信度


@dataclass
class BallTrackResult:
    """球追踪结果"""
    positions: List[BallPosition]
    impact_frame: Optional[int]  # 击球帧
    launch_angle: Optional[float]  # 起飞角度（度）
    launch_direction: Optional[float]  # 起飞方向（度，正为右偏）
    initial_speed: Optional[float]  # 初始速度（像素/秒）


class BallTracker:
    """高尔夫球追踪器"""

    def __init__(
        self,
        min_radius: int = 5,
        max_radius: int = 30,
        white_threshold: int = 200,
        circularity_threshold: float = 0.7
    ):
        """
        Args:
            min_radius: 最小球半径（像素）
            max_radius: 最大球半径（像素）
            white_threshold: 白色阈值（0-255）
            circularity_threshold: 圆度阈值（0-1）
        """
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.white_threshold = white_threshold
        self.circularity_threshold = circularity_threshold

    def detect_ball_in_frame(
        self,
        frame: np.ndarray,
        roi: Optional[Tuple[int, int, int, int]] = None
    ) -> Optional[Tuple[int, int, int, float]]:
        """
        在单帧中检测球

        Args:
            frame: BGR 图像
            roi: 感兴趣区域 (x, y, w, h)，可选

        Returns:
            (x, y, radius, confidence) 或 None
        """
        if roi:
            x, y, w, h = roi
            search_frame = frame[y:y+h, x:x+w]
            offset = (x, y)
        else:
            search_frame = frame
            offset = (0, 0)

        # 转换为灰度图
        gray = cv2.cvtColor(search_frame, cv2.COLOR_BGR2GRAY)

        # 白色区域提取
        _, white_mask = cv2.threshold(gray, self.white_threshold, 255, cv2.THRESH_BINARY)

        # 形态学操作去噪
        kernel = np.ones((3, 3), np.uint8)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)

        # 霍夫圆检测
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=30,
            param1=50,
            param2=30,
            minRadius=self.min_radius,
            maxRadius=self.max_radius
        )

        if circles is None:
            return None

        # 找最可能是球的圆
        best_circle = None
        best_score = 0

        for circle in circles[0]:
            cx, cy, r = circle
            cx, cy, r = int(cx), int(cy), int(r)

            # 检查圆心是否在白色区域
            if 0 <= cy < white_mask.shape[0] and 0 <= cx < white_mask.shape[1]:
                white_score = white_mask[cy, cx] / 255.0
            else:
                white_score = 0

            # 检查圆度
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.circle(mask, (cx, cy), r, 255, -1)
            intersection = cv2.bitwise_and(white_mask, mask)
            circle_area = np.pi * r * r
            white_area = np.sum(intersection > 0)
            circularity = white_area / (circle_area + 1e-6)

            # 综合得分
            score = white_score * 0.5 + min(circularity, 1.0) * 0.5

            if score > best_score and score > 0.3:
                best_score = score
                best_circle = (cx + offset[0], cy + offset[1], r, score)

        return best_circle

    def track_ball(
        self,
        video_processor,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        initial_position: Optional[Tuple[int, int]] = None
    ) -> BallTrackResult:
        """
        追踪视频中的球

        Args:
            video_processor: VideoProcessor 实例
            start_frame: 起始帧
            end_frame: 结束帧
            initial_position: 球的初始位置（可选，用于缩小搜索范围）

        Returns:
            BallTrackResult
        """
        positions = []
        last_position = initial_position
        search_radius = 100  # 搜索范围（像素）

        for frame_idx, frame in video_processor.extract_frames(start_frame, end_frame):
            # 定义搜索 ROI
            roi = None
            if last_position:
                x, y = last_position
                roi = (
                    max(0, x - search_radius),
                    max(0, y - search_radius),
                    min(search_radius * 2, video_processor.width - x + search_radius),
                    min(search_radius * 2, video_processor.height - y + search_radius)
                )

            # 检测球
            result = self.detect_ball_in_frame(frame, roi)

            if result:
                x, y, r, conf = result
                positions.append(BallPosition(
                    frame_index=frame_idx,
                    timestamp=frame_idx / video_processor.fps,
                    x=x,
                    y=y,
                    radius=r,
                    confidence=conf
                ))
                last_position = (x, y)

                # 球飞起后扩大搜索范围
                if len(positions) > 5:
                    search_radius = 150

        # 分析球的运动
        impact_frame, launch_angle, launch_dir, init_speed = self._analyze_motion(
            positions, video_processor.fps
        )

        return BallTrackResult(
            positions=positions,
            impact_frame=impact_frame,
            launch_angle=launch_angle,
            launch_direction=launch_dir,
            initial_speed=init_speed
        )

    def _analyze_motion(
        self,
        positions: List[BallPosition],
        fps: float
    ) -> Tuple[Optional[int], Optional[float], Optional[float], Optional[float]]:
        """分析球的运动，计算击球帧和起飞数据"""
        if len(positions) < 10:
            return None, None, None, None

        # 计算帧间位移
        displacements = []
        for i in range(1, len(positions)):
            dx = positions[i].x - positions[i-1].x
            dy = positions[i].y - positions[i-1].y
            dist = np.sqrt(dx*dx + dy*dy)
            displacements.append((positions[i].frame_index, dist, dx, dy))

        # 找击球帧（位移突然变大的点）
        impact_frame = None
        impact_idx = None
        for i, (frame_idx, dist, _, _) in enumerate(displacements):
            if dist > 10:  # 位移阈值
                # 检查之前几帧是否静止
                if i >= 3:
                    prev_avg = np.mean([d[1] for d in displacements[i-3:i]])
                    if prev_avg < 3 and dist > prev_avg * 3:
                        impact_frame = frame_idx
                        impact_idx = i
                        break

        if impact_frame is None or impact_idx is None:
            return None, None, None, None

        # 计算起飞角度和方向（使用击球后几帧的数据）
        post_impact = displacements[impact_idx:impact_idx+5]
        if len(post_impact) < 3:
            return impact_frame, None, None, None

        # 平均位移向量
        avg_dx = np.mean([d[2] for d in post_impact])
        avg_dy = np.mean([d[3] for d in post_impact])

        # 起飞角度（相对于水平线，向上为正）
        # 注意：图像坐标 y 向下为正，所以取负
        launch_angle = np.degrees(np.arctan2(-avg_dy, abs(avg_dx)))

        # 起飞方向（相对于正前方，右偏为正）
        launch_direction = np.degrees(np.arctan2(avg_dx, -avg_dy))

        # 初始速度（像素/秒）
        avg_dist = np.mean([d[1] for d in post_impact])
        initial_speed = avg_dist * fps

        return impact_frame, launch_angle, launch_direction, initial_speed


def find_ball_initial_position(
    frame: np.ndarray,
    expected_region: str = "bottom_center"
) -> Optional[Tuple[int, int]]:
    """
    在初始帧中找到球的位置

    Args:
        frame: 图像
        expected_region: 预期区域 ("bottom_center", "bottom_left", "bottom_right")

    Returns:
        (x, y) 或 None
    """
    tracker = BallTracker()
    h, w = frame.shape[:2]

    # 定义搜索区域
    if expected_region == "bottom_center":
        roi = (w // 4, h // 2, w // 2, h // 2)
    elif expected_region == "bottom_left":
        roi = (0, h // 2, w // 2, h // 2)
    else:
        roi = (w // 2, h // 2, w // 2, h // 2)

    result = tracker.detect_ball_in_frame(frame, roi)
    if result:
        return (result[0], result[1])
    return None
