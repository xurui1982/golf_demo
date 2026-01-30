"""视频处理模块 - 帧提取和预处理"""

import cv2
import numpy as np
from pathlib import Path
from typing import Generator, Tuple, Optional


class VideoProcessor:
    """处理高尔夫挥杆视频"""

    def __init__(self, video_path: str):
        self.video_path = Path(video_path)
        self.cap = cv2.VideoCapture(str(video_path))

        if not self.cap.isOpened():
            raise ValueError(f"无法打开视频: {video_path}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration = self.frame_count / self.fps if self.fps > 0 else 0

    @property
    def info(self) -> dict:
        """返回视频基本信息"""
        return {
            "path": str(self.video_path),
            "fps": self.fps,
            "frame_count": self.frame_count,
            "width": self.width,
            "height": self.height,
            "duration": self.duration,
            "is_slow_motion": self.fps >= 120
        }

    def extract_frames(
        self,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        step: int = 1
    ) -> Generator[Tuple[int, np.ndarray], None, None]:
        """
        逐帧提取视频

        Args:
            start_frame: 起始帧
            end_frame: 结束帧（None 表示到视频结尾）
            step: 帧间隔（1 表示每帧都取）

        Yields:
            (frame_index, frame) 元组
        """
        if end_frame is None:
            end_frame = self.frame_count

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        for i in range(start_frame, end_frame, step):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = self.cap.read()
            if not ret:
                break
            yield i, frame

    def get_frame(self, frame_index: int) -> Optional[np.ndarray]:
        """获取指定帧"""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = self.cap.read()
        return frame if ret else None

    def get_frame_at_time(self, time_sec: float) -> Optional[np.ndarray]:
        """获取指定时间点的帧"""
        frame_index = int(time_sec * self.fps)
        return self.get_frame(frame_index)

    def close(self):
        """释放视频资源"""
        self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def get_video_info(video_path: str) -> dict:
    """快速获取视频信息"""
    with VideoProcessor(video_path) as vp:
        return vp.info
