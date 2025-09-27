#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
文件名处理演示脚本
展示系统如何处理各种类型的文件名
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import FileHandler

def demo_filename_processing():
    """演示文件名处理功能"""
    print("🎭 HeyGem AI 数字人生成器 - 文件名处理演示")
    print("=" * 60)

    file_handler = FileHandler()

    # 演示文件名
    demo_files = [
        # 中文文件名
        "我的数字人视频.mp4",
        "中文音频文件.wav",
        "测试 音频 (最终版).mp3",

        # 英文文件名
        "My Digital Human Video.mp4",
        "Audio File for Testing.wav",
        "Final Version (2023).m4a",

        # 混合文件名
        "中英混合 Mixed Language 文件.mkv",
        "Project测试_Version2.mp4",

        # 特殊字符文件名
        "文件名包含特殊字符!@#$%^&*().avi",
        "Dangerous\\/:*?\"<>|Characters.mp3",
        "Too    Many    Spaces.wav",

        # 边界情况
        "",
        "   .mp4",
        ".hidden_file.mp3",
        "very_long_filename_that_might_cause_issues_in_some_systems_especially_when_dealing_with_unicode_characters_and_various_encodings.wav"
    ]

    print("📝 文件名处理演示:")
    print()

    for i, original in enumerate(demo_files, 1):
        if not original.strip():
            original = "空文件名"
            clean_name = file_handler.sanitize_filename("")
        else:
            clean_name = file_handler.sanitize_filename(original)

        print(f"{i:2d}. 原始文件名: '{original}'")
        print(f"    处理后文件名: '{clean_name}'")
        print()

    print("=" * 60)
    print("🌟 支持的文件格式:")
    print()
    print("🎵 音频格式:", ", ".join(file_handler.audio_formats))
    print("🎬 视频格式:", ", ".join(file_handler.video_formats))
    print()
    print("📁 自动创建的目录:")
    print("   • inputs/audio/  - 音频文件存储")
    print("   • inputs/video/  - 视频文件存储")
    print("   • result/        - 处理结果存储")
    print("   • temp/          - 临时文件存储")
    print()
    print("✨ 特性:")
    print("   • ✅ 完全支持中文文件名")
    print("   • ✅ 自动清理不安全字符")
    print("   • ✅ 防止文件名冲突")
    print("   • ✅ 文件完整性验证")
    print("   • ✅ 详细的上传进度显示")
    print()
    print("🚀 启动应用: ./start_app.sh")
    print("🌐 访问地址: http://localhost:7860")

if __name__ == "__main__":
    demo_filename_processing()