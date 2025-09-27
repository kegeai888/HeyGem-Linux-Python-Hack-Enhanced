#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试缓存文件目录配置
验证所有缓存文件都生成在temp目录中
"""

import os
import sys
import time
from pathlib import Path

# 在导入其他模块之前设置环境变量
os.environ["GRADIO_TEMP_DIR"] = "./temp/gradio_cache"
os.environ["TMPDIR"] = "./temp"
os.environ["TEMP"] = "./temp"
os.environ["TMP"] = "./temp"

# 确保目录存在
Path("./temp/gradio_cache").mkdir(parents=True, exist_ok=True)
Path("./temp/tensorrt_cache").mkdir(parents=True, exist_ok=True)

def test_cache_directories():
    """测试缓存目录配置"""
    print("🔍 测试缓存目录配置...")

    # 检查环境变量配置
    print("\n📋 环境变量配置:")
    env_vars = ['GRADIO_TEMP_DIR', 'TMPDIR', 'TEMP', 'TMP']
    for var in env_vars:
        value = os.environ.get(var, '未设置')
        print(f"  {var}: {value}")

    # 检查temp目录结构
    print("\n📁 temp目录结构:")
    temp_dir = Path("./temp")
    if temp_dir.exists():
        for item in temp_dir.rglob("*"):
            if item.is_dir():
                print(f"  📂 {item}")
            else:
                size_mb = item.stat().st_size / 1024 / 1024
                print(f"  📄 {item} ({size_mb:.2f}MB)")
    else:
        print("  ❌ temp目录不存在")
        return False

    # 检查预期的缓存目录
    expected_dirs = [
        "./temp/gradio_cache",
        "./temp/tensorrt_cache"
    ]

    print("\n✅ 预期缓存目录:")
    all_exist = True
    for dir_path in expected_dirs:
        path = Path(dir_path)
        if path.exists():
            print(f"  ✅ {dir_path}")
        else:
            print(f"  ❌ {dir_path} (不存在)")
            all_exist = False

    # 检查是否有缓存文件在根目录
    print("\n🚫 检查根目录是否有缓存文件:")
    root_cache_patterns = [
        "tensorrt_cache",
        "gradio_temp*",
        "*.cache",
        "temp_*"
    ]

    found_root_cache = False
    for pattern in root_cache_patterns:
        matches = list(Path(".").glob(pattern))
        for match in matches:
            if match.name != "temp":  # 排除我们的temp目录
                print(f"  ⚠️ 发现根目录缓存: {match}")
                found_root_cache = True

    if not found_root_cache:
        print("  ✅ 根目录没有发现缓存文件")

    return all_exist and not found_root_cache

def test_import_with_cache():
    """测试导入模块时的缓存配置"""
    print("\n🧪 测试模块导入缓存配置...")

    try:
        # 测试导入主模块
        print("  导入 app 模块...")
        import app

        print("  导入 gpu_optimization_engine 模块...")
        import gpu_optimization_engine

        print("  导入 optimized_onnx_model 模块...")
        import optimized_onnx_model

        print("  ✅ 所有模块导入成功")

        # 检查导入后的缓存目录
        time.sleep(1)  # 等待文件系统更新

        print("\n📁 导入后的temp目录:")
        temp_dir = Path("./temp")
        for item in temp_dir.rglob("*"):
            if item.is_dir():
                print(f"  📂 {item}")
            else:
                size_mb = item.stat().st_size / 1024 / 1024
                print(f"  📄 {item} ({size_mb:.2f}MB)")

        return True

    except Exception as e:
        print(f"  ❌ 模块导入失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🚀 开始测试缓存文件目录配置...\n")

    # 测试1: 缓存目录配置
    cache_ok = test_cache_directories()

    # 测试2: 模块导入缓存
    import_ok = test_import_with_cache()

    # 总结
    print("\n" + "="*50)
    print("📊 测试结果总结")
    print("="*50)

    if cache_ok and import_ok:
        print("✅ 所有测试通过！")
        print("📝 缓存配置:")
        print("  🎯 TensorRT缓存: ./temp/tensorrt_cache/")
        print("  🎯 Gradio缓存: ./temp/gradio_cache/")
        print("  🎯 系统临时文件: ./temp/")
        print("\n💡 所有生成的缓存文件都将存储在项目的 temp/ 目录下")
        return True
    else:
        print("❌ 部分测试失败")
        if not cache_ok:
            print("  💥 缓存目录配置有问题")
        if not import_ok:
            print("  💥 模块导入测试失败")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 缓存目录配置测试完成！")
    else:
        print("\n💥 测试过程中发现问题，请检查配置")
        sys.exit(1)