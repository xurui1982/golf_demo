"""高尔夫挥杆分析器 - 主入口"""

import argparse
import json
from pathlib import Path
from typing import Optional
import sys

from .video_processor import VideoProcessor
from .pose_analyzer import PoseAnalyzer
from .ball_tracker import BallTracker, find_ball_initial_position
from .swing_detector import SwingDetector, SwingPhase, get_phase_keyframes
from .metrics import MetricsCalculator, format_metrics
from .visualizer import (
    generate_annotated_video,
    save_keyframe,
    plot_swing_metrics,
    Visualizer
)


def analyze_swing(
    video_path: str,
    output_dir: str,
    track_ball: bool = True,
    generate_video: bool = True,
    verbose: bool = True
) -> dict:
    """
    分析高尔夫挥杆视频

    Args:
        video_path: 输入视频路径
        output_dir: 输出目录
        track_ball: 是否追踪球
        generate_video: 是否生成标注视频
        verbose: 是否输出详细信息

    Returns:
        分析结果字典
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 创建子目录
    keyframes_dir = output_path / "keyframes"
    data_dir = output_path / "data"
    charts_dir = output_path / "charts"

    keyframes_dir.mkdir(exist_ok=True)
    data_dir.mkdir(exist_ok=True)
    charts_dir.mkdir(exist_ok=True)

    results = {"video_path": video_path}

    # 1. 读取视频
    if verbose:
        print(f"读取视频: {video_path}")

    with VideoProcessor(video_path) as vp:
        video_info = vp.info
        results["video_info"] = video_info

        if verbose:
            print(f"  帧率: {video_info['fps']} fps")
            print(f"  分辨率: {video_info['width']}x{video_info['height']}")
            print(f"  时长: {video_info['duration']:.2f}s")
            print(f"  总帧数: {video_info['frame_count']}")

        # 2. 姿态分析
        if verbose:
            print("\n正在分析姿态...")

        with PoseAnalyzer(model_complexity=2) as pose_analyzer:
            def progress(current, total):
                if verbose and current % 50 == 0:
                    print(f"  处理进度: {current}/{total}")

            pose_results = pose_analyzer.analyze_video(vp, progress_callback=progress)

        if verbose:
            print(f"  检测到 {len(pose_results)} 帧有效姿态")

        results["pose_count"] = len(pose_results)

        # 3. 球追踪
        ball_result = None
        if track_ball and video_info['fps'] >= 60:
            if verbose:
                print("\n正在追踪球...")

            # 找到球的初始位置
            first_frame = vp.get_frame(0)
            initial_pos = find_ball_initial_position(first_frame)

            if initial_pos:
                if verbose:
                    print(f"  初始球位置: {initial_pos}")

                tracker = BallTracker()
                ball_result = tracker.track_ball(vp, initial_position=initial_pos)

                if verbose:
                    print(f"  追踪到 {len(ball_result.positions)} 帧球位置")
                    if ball_result.impact_frame:
                        print(f"  击球帧: {ball_result.impact_frame}")
                    if ball_result.launch_angle:
                        print(f"  起飞角度: {ball_result.launch_angle:.1f}°")
            else:
                if verbose:
                    print("  未找到初始球位置")

        # 4. 挥杆阶段检测
        if verbose:
            print("\n正在检测挥杆阶段...")

        swing_detector = SwingDetector(fps=video_info['fps'])
        impact_frame = ball_result.impact_frame if ball_result else None

        try:
            swing_analysis = swing_detector.detect_phases(pose_results, impact_frame)
            phase_keyframes = get_phase_keyframes(swing_analysis)

            results["swing_analysis"] = {
                "backswing_duration": swing_analysis.backswing_duration,
                "downswing_duration": swing_analysis.downswing_duration,
                "tempo_ratio": swing_analysis.tempo_ratio,
                "total_duration": swing_analysis.total_duration,
                "phases": [
                    {
                        "phase": p.phase.value,
                        "start_frame": p.start_frame,
                        "end_frame": p.end_frame,
                        "key_frame": p.key_frame
                    }
                    for p in swing_analysis.phases
                ]
            }

            if verbose:
                print(f"  上杆时间: {swing_analysis.backswing_duration:.2f}s")
                print(f"  下杆时间: {swing_analysis.downswing_duration:.2f}s")
                print(f"  节奏比: {swing_analysis.tempo_ratio:.1f}:1")

        except ValueError as e:
            if verbose:
                print(f"  警告: {e}")
            phase_keyframes = {}
            swing_analysis = None

        # 5. 计算指标
        if verbose:
            print("\n正在计算指标...")

        calculator = MetricsCalculator(fps=video_info['fps'])
        metrics = calculator.calculate_all(
            pose_results,
            phase_keyframes,
            ball_speed=ball_result.initial_speed if ball_result else None,
            launch_angle=ball_result.launch_angle if ball_result else None,
            launch_direction=ball_result.launch_direction if ball_result else None
        )

        formatted_metrics = format_metrics(metrics)
        results["metrics"] = formatted_metrics

        if verbose:
            print("  指标:")
            for name, value in formatted_metrics.items():
                print(f"    {name}: {value}")

        # 6. 保存关键帧
        if verbose:
            print("\n正在保存关键帧...")

        for phase_name, frame_idx in phase_keyframes.items():
            frame = vp.get_frame(frame_idx)
            if frame is not None:
                pose_at_frame = None
                for pr in pose_results:
                    if pr.frame_index == frame_idx:
                        pose_at_frame = pr
                        break

                output_file = keyframes_dir / f"{phase_name}.jpg"
                save_keyframe(frame, pose_at_frame, str(output_file), phase_name)

                if verbose:
                    print(f"  保存: {output_file}")

        # 7. 生成图表
        if verbose:
            print("\n正在生成图表...")

        plot_swing_metrics(
            pose_results,
            str(charts_dir / "swing_metrics.png"),
            title="Swing Analysis"
        )

        if verbose:
            print(f"  保存: {charts_dir / 'swing_metrics.png'}")

        # 8. 生成标注视频
        if generate_video:
            if verbose:
                print("\n正在生成标注视频...")

            # 建立帧到阶段的映射
            phase_map = {}
            if swing_analysis:
                for phase_info in swing_analysis.phases:
                    for f in range(phase_info.start_frame, phase_info.end_frame + 1):
                        phase_map[f] = phase_info.phase

            output_video = output_path / "annotated.mp4"
            generate_annotated_video(
                video_path,
                str(output_video),
                pose_results,
                ball_positions=ball_result.positions if ball_result else None,
                phase_info=phase_map
            )

            if verbose:
                print(f"  保存: {output_video}")

        # 9. 保存数据
        if verbose:
            print("\n正在保存数据...")

        # 保存姿态数据
        pose_data = []
        for pr in pose_results:
            landmarks_dict = {}
            for name, lm in pr.landmarks.items():
                landmarks_dict[name] = {
                    "x": lm.x,
                    "y": lm.y,
                    "z": lm.z,
                    "visibility": lm.visibility
                }
            pose_data.append({
                "frame_index": pr.frame_index,
                "timestamp": pr.timestamp,
                "landmarks": landmarks_dict
            })

        with open(data_dir / "pose_data.json", "w") as f:
            json.dump(pose_data, f, indent=2)

        # 保存球追踪数据
        if ball_result:
            ball_data = {
                "positions": [
                    {
                        "frame_index": bp.frame_index,
                        "timestamp": bp.timestamp,
                        "x": bp.x,
                        "y": bp.y,
                        "radius": bp.radius,
                        "confidence": bp.confidence
                    }
                    for bp in ball_result.positions
                ],
                "impact_frame": ball_result.impact_frame,
                "launch_angle": ball_result.launch_angle,
                "launch_direction": ball_result.launch_direction,
                "initial_speed": ball_result.initial_speed
            }
            with open(data_dir / "ball_track.json", "w") as f:
                json.dump(ball_data, f, indent=2)

        # 保存指标数据
        with open(data_dir / "metrics.json", "w") as f:
            json.dump(formatted_metrics, f, indent=2)

        if verbose:
            print(f"\n分析完成！结果保存在: {output_path}")

    return results


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(
        description="高尔夫挥杆分析器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python -m src.main video.mp4 -o output/
  python -m src.main swing.mov -o result/ --no-ball
  python -m src.main test.mp4 -o out/ --no-video -q
        """
    )

    parser.add_argument("video", help="输入视频文件路径")
    parser.add_argument("-o", "--output", default="output",
                        help="输出目录 (默认: output)")
    parser.add_argument("--no-ball", action="store_true",
                        help="不追踪球")
    parser.add_argument("--no-video", action="store_true",
                        help="不生成标注视频")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="安静模式")

    args = parser.parse_args()

    # 检查输入文件
    if not Path(args.video).exists():
        print(f"错误: 找不到视频文件: {args.video}")
        sys.exit(1)

    # 运行分析
    analyze_swing(
        args.video,
        args.output,
        track_ball=not args.no_ball,
        generate_video=not args.no_video,
        verbose=not args.quiet
    )


if __name__ == "__main__":
    main()
