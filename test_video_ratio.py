#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试视频预览比例配置
验证输出视频按原始比例显示
"""

import os
import sys
import time
from pathlib import Path

# 设置环境变量
os.environ["GRADIO_TEMP_DIR"] = "./temp/gradio_cache"
os.environ["TMPDIR"] = "./temp"

def test_video_preview_ratio():
    """测试视频预览比例配置"""
    print("🎬 测试视频预览比例配置...\n")

    try:
        import gradio as gr
        print(f"✅ Gradio 版本: {gr.__version__}")

        # 创建测试界面来验证视频组件配置
        with gr.Blocks(css="""
        /* 输出视频样式（按原始比例显示） */
        .output-video {
            width: 100%;
            max-width: 100%;
            height: auto;
            object-fit: contain;
            border-radius: 8px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        }

        /* Gradio视频组件样式覆盖 */
        .gradio-container video {
            width: 100% !important;
            height: auto !important;
            max-height: none !important;
            object-fit: contain !important;
        }
        """) as demo:

            gr.HTML("<h1>🎬 视频预览比例测试</h1>")

            with gr.Row():
                with gr.Column():
                    gr.HTML("<h3>📤 上传测试视频</h3>")
                    input_video = gr.Video(
                        label="上传视频文件",
                        elem_classes=["input-video"]
                    )

                with gr.Column():
                    gr.HTML("<h3>📺 输出视频（按原始比例）</h3>")
                    output_video = gr.Video(
                        label="✨ 生成的数字人视频",
                        show_label=True,
                        autoplay=False,
                        show_download_button=True,
                        elem_classes=["output-video"]
                    )

            # 测试函数
            def test_video_display(video):
                if video:
                    print(f"📹 测试视频: {video}")
                    # 返回相同的视频来测试显示效果
                    return video
                return None

            input_video.change(
                fn=test_video_display,
                inputs=[input_video],
                outputs=[output_video]
            )

        print("✅ 测试界面创建成功")
        print("📝 配置检查:")
        print("  🎯 移除了固定height=400限制")
        print("  🎯 添加了自定义CSS类 'output-video'")
        print("  🎯 设置了object-fit: contain保持比例")
        print("  🎯 添加了height: auto自适应高度")
        print("  🎯 添加了!important样式覆盖")

        return demo

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return None

def check_app_video_config():
    """检查主应用的视频配置"""
    print("\n🔍 检查主应用视频配置...")

    try:
        # 读取app.py文件检查配置
        with open('app.py', 'r', encoding='utf-8') as f:
            content = f.read()

        checks = {
            "移除固定高度": "height=400" not in content,
            "添加CSS类": 'elem_classes=["output-video"]' in content,
            "添加原始比例CSS": ".output-video" in content,
            "添加Gradio覆盖CSS": ".gradio-container video" in content,
            "设置自动播放关闭": "autoplay=False" in content,
            "显示下载按钮": "show_download_button=True" in content
        }

        print("📋 配置检查结果:")
        all_good = True
        for check_name, passed in checks.items():
            status = "✅" if passed else "❌"
            print(f"  {status} {check_name}")
            if not passed:
                all_good = False

        return all_good

    except Exception as e:
        print(f"❌ 检查失败: {e}")
        return False

def test_css_styles():
    """测试CSS样式配置"""
    print("\n🎨 测试CSS样式配置...")

    expected_styles = [
        ".output-video {",
        "width: 100%;",
        "height: auto;",
        "object-fit: contain;",
        ".gradio-container video {",
        "width: 100% !important;",
        "height: auto !important;",
        "max-height: none !important;"
    ]

    try:
        with open('app.py', 'r', encoding='utf-8') as f:
            content = f.read()

        print("📋 CSS样式检查:")
        all_styles_present = True
        for style in expected_styles:
            if style in content:
                print(f"  ✅ {style}")
            else:
                print(f"  ❌ 缺少: {style}")
                all_styles_present = False

        return all_styles_present

    except Exception as e:
        print(f"❌ CSS检查失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🚀 开始测试视频预览比例配置...\n")

    # 测试1: Gradio视频组件
    print("=" * 50)
    print("📋 测试1: Gradio视频组件配置")
    print("=" * 50)
    demo = test_video_preview_ratio()

    # 测试2: 主应用配置检查
    print("=" * 50)
    print("📋 测试2: 主应用配置检查")
    print("=" * 50)
    app_config_ok = check_app_video_config()

    # 测试3: CSS样式检查
    print("=" * 50)
    print("📋 测试3: CSS样式检查")
    print("=" * 50)
    css_ok = test_css_styles()

    # 总结
    print("\n" + "=" * 50)
    print("🎉 测试结果总结")
    print("=" * 50)

    if demo and app_config_ok and css_ok:
        print("✅ 所有测试通过！")
        print("\n📝 视频预览配置:")
        print("  🎯 移除固定高度限制 (height=400)")
        print("  🎯 添加自适应高度 (height: auto)")
        print("  🎯 保持原始比例 (object-fit: contain)")
        print("  🎯 添加CSS样式覆盖 (!important)")
        print("  🎯 优化用户体验 (下载按钮、自动播放控制)")

        print("\n🎬 预览效果:")
        print("  📺 视频将按原始分辨率比例显示")
        print("  📱 自动适应容器宽度")
        print("  🔍 不会出现拉伸或压缩变形")
        print("  ✨ 保持视频清晰度和比例")

        print("\n🚀 可以启动应用测试效果:")
        print("  ./start_app.sh")
        print("  或")
        print("  python app.py")

        return True
    else:
        print("❌ 部分测试失败")
        if not demo:
            print("  💥 Gradio组件测试失败")
        if not app_config_ok:
            print("  💥 主应用配置检查失败")
        if not css_ok:
            print("  💥 CSS样式检查失败")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎊 视频预览比例配置测试完成！")
    else:
        print("\n💥 测试过程中发现问题，请检查配置")
        sys.exit(1)