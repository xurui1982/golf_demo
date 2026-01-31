"""挥杆阶段检测模块 - 自动识别挥杆各阶段"""

import numpy as np
from typing import List, Optional, Dict
from dataclasses import dataclass
from enum import Enum

from .pose_analyzer import PoseResult, PoseLandmark, calculate_angle


class SwingPhase(Enum):
    """挥杆阶段"""
    SETUP = "setup"           # 准备
    BACKSWING = "backswing"   # 上杆
    TOP = "top"               # 顶点
    DOWNSWING = "downswing"   # 下杆
    IMPACT = "impact"         # 击球
    FOLLOW_THROUGH = "follow_through"  # 收杆
    FINISH = "finish"         # 结束


@dataclass
class SwingPhaseInfo:
    """挥杆阶段信息"""
    phase: SwingPhase
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    key_frame: int  # 该阶段的关键帧


@dataclass
class SwingAnalysis:
    """挥杆分析结果"""
    phases: List[SwingPhaseInfo]
    backswing_duration: float  # 上杆时间（秒）
    downswing_duration: float  # 下杆时间（秒）
    tempo_ratio: float  # 节奏比（上杆/下杆）
    total_duration: float  # 总挥杆时间


class SwingDetector:
    """挥杆阶段检测器"""

    def __init__(self, fps: float = 240.0):
        self.fps = fps

    def detect_phases(
        self,
        pose_results: List[PoseResult],
        impact_frame: Optional[int] = None
    ) -> SwingAnalysis:
        """
        检测挥杆各阶段

        Args:
            pose_results: 姿态分析结果列表
            impact_frame: 击球帧（如果已知，由球追踪提供）

        Returns:
            SwingAnalysis
        """
        if len(pose_results) < 10:
            raise ValueError("姿态数据不足，无法分析挥杆阶段")

        # 提取手腕位置序列（用于检测挥杆运动）
        wrist_positions = self._extract_wrist_positions(pose_results)

        # 检测各阶段
        phases = []

        # 1. 找到顶点（手腕最高点）
        top_idx = self._find_top_position(wrist_positions)

        # 2. 找到击球点
        if impact_frame is not None:
            impact_idx = self._frame_to_index(impact_frame, pose_results)
        else:
            impact_idx = self._estimate_impact(wrist_positions, top_idx)

        # 3. 找到上杆开始
        backswing_start = self._find_backswing_start(wrist_positions, top_idx)

        # 4. 找到收杆结束
        finish_idx = self._find_finish(wrist_positions, impact_idx)

        # 构建阶段列表
        # Setup
        if backswing_start > 0:
            phases.append(self._create_phase(
                SwingPhase.SETUP, 0, backswing_start - 1, pose_results
            ))

        # Backswing
        phases.append(self._create_phase(
            SwingPhase.BACKSWING, backswing_start, top_idx - 1, pose_results
        ))

        # Top
        phases.append(self._create_phase(
            SwingPhase.TOP, top_idx, top_idx, pose_results
        ))

        # Downswing
        phases.append(self._create_phase(
            SwingPhase.DOWNSWING, top_idx + 1, impact_idx - 1, pose_results
        ))

        # Impact
        phases.append(self._create_phase(
            SwingPhase.IMPACT, impact_idx, impact_idx, pose_results
        ))

        # Follow-through
        if finish_idx > impact_idx:
            phases.append(self._create_phase(
                SwingPhase.FOLLOW_THROUGH, impact_idx + 1, finish_idx, pose_results
            ))

        # 计算时间数据
        backswing_duration = (top_idx - backswing_start) / self.fps
        downswing_duration = (impact_idx - top_idx) / self.fps
        tempo_ratio = backswing_duration / downswing_duration if downswing_duration > 0 else 0
        total_duration = (finish_idx - backswing_start) / self.fps

        return SwingAnalysis(
            phases=phases,
            backswing_duration=backswing_duration,
            downswing_duration=downswing_duration,
            tempo_ratio=tempo_ratio,
            total_duration=total_duration
        )

    def _extract_wrist_positions(
        self,
        pose_results: List[PoseResult]
    ) -> List[Dict]:
        """提取手腕位置序列"""
        positions = []
        for pr in pose_results:
            left_wrist = pr.landmarks.get("left_wrist")
            right_wrist = pr.landmarks.get("right_wrist")

            if left_wrist and right_wrist:
                # 使用双手的平均位置
                avg_x = (left_wrist.x + right_wrist.x) / 2
                avg_y = (left_wrist.y + right_wrist.y) / 2
                positions.append({
                    "frame": pr.frame_index,
                    "x": avg_x,
                    "y": avg_y,
                    "left": (left_wrist.x, left_wrist.y),
                    "right": (right_wrist.x, right_wrist.y)
                })

        return positions

    def _find_top_position(self, wrist_positions: List[Dict]) -> int:
        """
        找到挥杆顶点（第一个显著的手腕高度峰值）

        改进：不再找全局最高点，而是找第一个局部峰值，
        这样可以正确区分上杆顶点和收杆位置。
        """
        if len(wrist_positions) < 5:
            return 0

        # 提取 y 坐标序列（y 越小表示越高）
        y_values = [pos["y"] for pos in wrist_positions]

        # 简单平滑（移动平均）
        window = 5
        smoothed = []
        for i in range(len(y_values)):
            start = max(0, i - window // 2)
            end = min(len(y_values), i + window // 2 + 1)
            smoothed.append(np.mean(y_values[start:end]))

        # 找所有局部最小值（即手腕最高点）
        # 局部最小值：比前后都小
        peaks = []
        for i in range(2, len(smoothed) - 2):
            if (smoothed[i] < smoothed[i-1] and
                smoothed[i] < smoothed[i-2] and
                smoothed[i] < smoothed[i+1] and
                smoothed[i] < smoothed[i+2]):
                # 计算峰值的显著性（与周围的差异）
                prominence = min(
                    max(smoothed[max(0,i-10):i]) - smoothed[i],
                    max(smoothed[i+1:min(len(smoothed),i+11)]) - smoothed[i]
                ) if i > 0 and i < len(smoothed) - 1 else 0
                peaks.append((i, smoothed[i], prominence))

        if not peaks:
            # 如果没找到局部峰值，使用前半部分的最小值
            half = len(smoothed) // 2
            min_idx = 0
            min_val = smoothed[0]
            for i in range(half):
                if smoothed[i] < min_val:
                    min_val = smoothed[i]
                    min_idx = i
            return min_idx

        # 找第一个显著的峰值（prominence > 0.05）
        for idx, val, prominence in peaks:
            if prominence > 0.03:  # 阈值可调
                return idx

        # 如果没有显著峰值，返回第一个峰值
        return peaks[0][0]

    def _estimate_impact(
        self,
        wrist_positions: List[Dict],
        top_idx: int
    ) -> int:
        """
        估计击球点

        策略：顶点之后，手腕快速下降到接近初始高度的位置
        （击球时手腕位置接近 setup 时的高度）
        """
        if top_idx >= len(wrist_positions) - 1:
            return len(wrist_positions) - 1

        # 获取初始手腕高度（setup 位置）
        initial_y = wrist_positions[0]["y"]

        # 从顶点之后开始，找手腕下降到接近初始高度的点
        # 同时考虑下降速度
        best_idx = top_idx + 1
        best_score = float('inf')

        # 限制搜索范围：顶点后的一段时间内
        search_end = min(top_idx + len(wrist_positions) // 3, len(wrist_positions) - 1)

        for i in range(top_idx + 1, search_end):
            current_y = wrist_positions[i]["y"]

            # 计算与初始高度的差距
            height_diff = abs(current_y - initial_y)

            # 计算下降速度（从顶点到当前）
            if i > top_idx:
                dy = wrist_positions[i]["y"] - wrist_positions[i-1]["y"]
                # 正的 dy 表示向下移动（y 坐标增加）

                # 综合得分：接近初始高度 + 正在下降
                if dy > 0:  # 手腕正在下降
                    score = height_diff
                    if score < best_score:
                        best_score = score
                        best_idx = i

        # 如果没找到好的点，使用手腕速度最大的点
        if best_score == float('inf'):
            max_speed = 0
            for i in range(top_idx + 1, search_end):
                dx = wrist_positions[i]["x"] - wrist_positions[i-1]["x"]
                dy = wrist_positions[i]["y"] - wrist_positions[i-1]["y"]
                speed = np.sqrt(dx * dx + dy * dy)
                if speed > max_speed:
                    max_speed = speed
                    best_idx = i

        return best_idx

    def _find_backswing_start(
        self,
        wrist_positions: List[Dict],
        top_idx: int
    ) -> int:
        """找到上杆开始点（手腕开始向上移动）"""
        if top_idx <= 0:
            return 0

        # 从顶点往回找，找到手腕开始移动的点
        threshold = 0.005  # 移动阈值

        for i in range(top_idx - 1, 0, -1):
            dy = wrist_positions[i + 1]["y"] - wrist_positions[i]["y"]
            if abs(dy) < threshold:
                return i + 1

        return 0

    def _find_finish(
        self,
        wrist_positions: List[Dict],
        impact_idx: int
    ) -> int:
        """找到收杆结束点"""
        if impact_idx >= len(wrist_positions) - 1:
            return len(wrist_positions) - 1

        # 找到击球后手腕速度趋于稳定的点
        threshold = 0.003
        stable_count = 0

        for i in range(impact_idx + 1, len(wrist_positions) - 1):
            dx = wrist_positions[i + 1]["x"] - wrist_positions[i]["x"]
            dy = wrist_positions[i + 1]["y"] - wrist_positions[i]["y"]
            speed = np.sqrt(dx * dx + dy * dy)

            if speed < threshold:
                stable_count += 1
                if stable_count >= 5:  # 连续稳定
                    return i
            else:
                stable_count = 0

        return len(wrist_positions) - 1

    def _frame_to_index(
        self,
        frame: int,
        pose_results: List[PoseResult]
    ) -> int:
        """将帧号转换为 pose_results 中的索引"""
        for i, pr in enumerate(pose_results):
            if pr.frame_index >= frame:
                return i
        return len(pose_results) - 1

    def _create_phase(
        self,
        phase: SwingPhase,
        start_idx: int,
        end_idx: int,
        pose_results: List[PoseResult]
    ) -> SwingPhaseInfo:
        """创建阶段信息"""
        start_idx = max(0, min(start_idx, len(pose_results) - 1))
        end_idx = max(0, min(end_idx, len(pose_results) - 1))

        return SwingPhaseInfo(
            phase=phase,
            start_frame=pose_results[start_idx].frame_index,
            end_frame=pose_results[end_idx].frame_index,
            start_time=pose_results[start_idx].timestamp,
            end_time=pose_results[end_idx].timestamp,
            key_frame=pose_results[(start_idx + end_idx) // 2].frame_index
        )


def get_phase_keyframes(swing_analysis: SwingAnalysis) -> Dict[str, int]:
    """获取各阶段的关键帧"""
    return {
        phase_info.phase.value: phase_info.key_frame
        for phase_info in swing_analysis.phases
    }
