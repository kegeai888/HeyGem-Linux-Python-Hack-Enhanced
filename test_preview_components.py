#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试修改后的预览组件功能
"""

import os
import sys
import traceback
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

def test_gradio_components():
    """测试Gradio组件是否正常工作"""
    try:
        import gradio as gr
        print(f"✅ Gradio 版本: {gr.__version__}")

        # 创建简单的测试界面
        with gr.Blocks() as demo:
            gr.HTML("<h1>🎵 音频和视频预览组件测试</h1>")

            with gr.Row():
                # 测试音频组件
                audio_component = gr.Audio(
                    label="🎵 测试音频上传 (支持预览)",
                    type="filepath"
                )

                # 测试视频组件
                video_component = gr.Video(
                    label="🎬 测试视频上传 (支持预览)"
                )

            # 测试功能
            def test_files(audio, video):
                result = "文件上传测试结果:\n"

                if audio:
                    audio_path = audio if isinstance(audio, str) else audio.name
                    result += f"✅ 音频文件: {os.path.basename(audio_path)}\n"
                    result += f"   路径: {audio_path}\n"
                    result += f"   类型: {type(audio)}\n"
                else:
                    result += "❌ 未上传音频文件\n"

                if video:
                    video_path = video if isinstance(video, str) else video.name
                    result += f"✅ 视频文件: {os.path.basename(video_path)}\n"
                    result += f"   路径: {video_path}\n"
                    result += f"   类型: {type(video)}\n"
                else:
                    result += "❌ 未上传视频文件\n"

                return result

            test_btn = gr.Button("🧪 测试文件处理", variant="primary")
            output_text = gr.Textbox(label="测试结果", lines=10)

            test_btn.click(
                fn=test_files,
                inputs=[audio_component, video_component],
                outputs=[output_text]
            )

        print("✅ Gradio界面创建成功")
        print("📝 界面包含以下组件:")
        print("   🎵 gr.Audio 组件 (支持音频预览)")
        print("   🎬 gr.Video 组件 (支持视频预览)")
        print("   🧪 文件处理测试功能")

        return demo

    except Exception as e:
        print(f"❌ Gradio组件测试失败: {e}")
        traceback.print_exc()
        return None

def test_app_import():
    """测试主应用是否能正常导入"""
    try:
        # 测试导入主应用
        print("🔍 测试导入主应用...")

        # 检查关键模块是否可用
        try:
            import app
            print("✅ app.py 导入成功")
        except ImportError as e:
            print(f"❌ app.py 导入失败: {e}")
            return False

        # 检查VideoProcessor类
        try:
            processor = app.VideoProcessor()
            print("✅ VideoProcessor 初始化成功")
        except Exception as e:
            print(f"❌ VideoProcessor 初始化失败: {e}")
            return False

        # 检查界面创建函数
        try:
            interface = app.create_interface(processor)
            print("✅ create_interface 函数正常")
            return True
        except Exception as e:
            print(f"❌ create_interface 函数失败: {e}")
            return False

    except Exception as e:
        print(f"❌ 主应用测试失败: {e}")
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("🚀 开始测试预览组件功能...\n")

    # 测试1: Gradio组件
    print("=" * 50)
    print("📋 测试1: Gradio组件功能")
    print("=" * 50)
    demo = test_gradio_components()

    if demo:
        print("✅ Gradio组件测试通过")
    else:
        print("❌ Gradio组件测试失败")
        return False

    # 测试2: 主应用导入
    print("\n" + "=" * 50)
    print("📋 测试2: 主应用导入测试")
    print("=" * 50)
    app_ok = test_app_import()

    if app_ok:
        print("✅ 主应用导入测试通过")
    else:
        print("❌ 主应用导入测试失败")
        return False

    # 总结
    print("\n" + "=" * 50)
    print("🎉 测试总结")
    print("=" * 50)
    print("✅ 所有测试通过！")
    print("📝 修改内容:")
    print("   🔄 将 gr.File 组件替换为 gr.Audio 和 gr.Video")
    print("   🎵 音频组件支持直接试听和预览")
    print("   🎬 视频组件支持直接播放和预览")
    print("   🛠️ 更新了文件处理逻辑以适应新组件")
    print("   📱 优化了用户界面提示信息")

    print("\n🚀 可以启动应用程序测试预览功能:")
    print("   ./start_app.sh")
    print("   或")
    print("   python app.py")

    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎊 预览组件功能测试完成！")
    else:
        print("\n💥 测试过程中发现问题，请检查错误信息")
        sys.exit(1)