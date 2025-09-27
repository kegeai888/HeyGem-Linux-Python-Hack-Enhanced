#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
文件处理功能测试脚本
测试中文文件名处理、格式验证等功能
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入文件处理器
from app import FileHandler

def test_filename_sanitization():
    """测试文件名清理功能"""
    print("🔍 测试文件名清理功能...")

    file_handler = FileHandler()

    test_cases = [
        ("我的音频文件.mp3", "我的音频文件.mp3"),
        ("My Audio File.wav", "My_Audio_File.wav"),
        ("测试文件!@#$%^&*().mp4", "测试文件_@_^_().mp4"),
        ("Audio with spaces  .m4a", "Audio_with_spaces.m4a"),
        ("中文 English 123.aac", "中文_English_123.aac"),
        ("", "unnamed_file"),
        ("   .mp3", "file_"),  # 会添加时间戳
        ("特殊字符\\/:*?\"<>|.flac", "特殊字符_.flac"),  # 不安全字符会被替换为下划线
    ]

    for original, expected_pattern in test_cases:
        result = file_handler.sanitize_filename(original)
        print(f"   原始: '{original}' -> 清理后: '{result}'")

        # 验证基本规则
        if result:
            # 检查不安全字符是否被移除
            unsafe_chars = r'\\/:*?"<>|'
            assert not any(char in result for char in unsafe_chars), f"仍包含不安全字符: {result}"

            # 检查文件扩展名数量
            parts = result.split('.')
            assert len(parts) <= 2, f"文件名包含多个点: {result}"

    print("✅ 文件名清理功能测试通过")

def test_format_validation():
    """测试文件格式验证"""
    print("🔍 测试文件格式验证...")

    file_handler = FileHandler()

    # 测试音频格式
    audio_tests = [
        ("test.mp3", True),
        ("test.wav", True),
        ("test.m4a", True),
        ("test.aac", True),
        ("test.flac", True),
        ("test.ogg", True),
        ("test.txt", False),
        ("test.jpg", False),
        ("test", False),
    ]

    for filename, should_be_valid in audio_tests:
        is_valid, message = file_handler.validate_file_format(filename, 'audio')
        print(f"   音频 '{filename}': {is_valid} - {message}")
        assert is_valid == should_be_valid, f"音频格式验证失败: {filename}"

    # 测试视频格式
    video_tests = [
        ("test.mp4", True),
        ("test.avi", True),
        ("test.mov", True),
        ("test.mkv", True),
        ("test.wmv", True),
        ("test.flv", True),
        ("test.webm", True),
        ("test.txt", False),
        ("test.mp3", False),
    ]

    for filename, should_be_valid in video_tests:
        is_valid, message = file_handler.validate_file_format(filename, 'video')
        print(f"   视频 '{filename}': {is_valid} - {message}")
        assert is_valid == should_be_valid, f"视频格式验证失败: {filename}"

    print("✅ 文件格式验证测试通过")

def test_directory_creation():
    """测试目录创建功能"""
    print("🔍 测试目录创建功能...")

    # 临时测试目录
    test_base_dir = tempfile.mkdtemp(prefix="heygem_test_")
    original_cwd = os.getcwd()

    try:
        os.chdir(test_base_dir)

        # 创建文件处理器实例（会自动创建目录）
        file_handler = FileHandler()

        # 验证目录是否创建
        expected_dirs = ['inputs', 'inputs/audio', 'inputs/video', 'result', 'temp']

        for dir_path in expected_dirs:
            assert os.path.exists(dir_path), f"目录未创建: {dir_path}"
            assert os.path.isdir(dir_path), f"路径不是目录: {dir_path}"
            print(f"   ✅ 目录已创建: {dir_path}")

        print("✅ 目录创建功能测试通过")

    finally:
        os.chdir(original_cwd)
        shutil.rmtree(test_base_dir, ignore_errors=True)

def test_file_copy_and_rename():
    """测试文件复制和重命名功能"""
    print("🔍 测试文件复制和重命名功能...")

    test_base_dir = tempfile.mkdtemp(prefix="heygem_copy_test_")
    original_cwd = os.getcwd()

    try:
        os.chdir(test_base_dir)

        file_handler = FileHandler()

        # 创建测试文件
        test_content = b"test audio content"
        test_filename = "测试音频文件.mp3"
        test_file_path = os.path.join(test_base_dir, test_filename)

        with open(test_file_path, 'wb') as f:
            f.write(test_content)

        # 测试复制和重命名
        target_path, clean_name = file_handler.copy_and_rename_file(test_file_path, 'audio')

        print(f"   原始文件: {test_filename}")
        print(f"   清理后名称: {clean_name}")
        print(f"   目标路径: {target_path}")

        # 验证文件是否复制成功
        assert os.path.exists(target_path), f"目标文件未创建: {target_path}"

        # 验证文件内容
        with open(target_path, 'rb') as f:
            copied_content = f.read()

        assert copied_content == test_content, "文件内容不匹配"

        # 验证文件大小
        original_size = os.path.getsize(test_file_path)
        copied_size = os.path.getsize(target_path)
        assert original_size == copied_size, f"文件大小不匹配: {original_size} vs {copied_size}"

        print("✅ 文件复制和重命名功能测试通过")

    finally:
        os.chdir(original_cwd)
        shutil.rmtree(test_base_dir, ignore_errors=True)

def main():
    """运行所有测试"""
    print("🚀 开始测试 HeyGem 文件处理功能")
    print("=" * 50)

    try:
        test_filename_sanitization()
        print()

        test_format_validation()
        print()

        test_directory_creation()
        print()

        test_file_copy_and_rename()
        print()

        print("🎉 所有测试通过！")
        print("=" * 50)
        print("✅ 文件处理功能验证完成")

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()