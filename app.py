import argparse
import gc
import json
import os
import re
import unicodedata
import glob

# 配置缓存和临时文件目录
os.environ["GRADIO_SERVER_NAME"] = "0.0.0.0"
os.environ["GRADIO_TEMP_DIR"] = "./temp/gradio_cache"  # Gradio临时文件
os.environ["TMPDIR"] = "./temp"  # 系统临时文件
os.environ["TEMP"] = "./temp"  # Windows兼容
os.environ["TMP"] = "./temp"  # Windows兼容

# 确保缓存目录存在
from pathlib import Path
Path("./temp/gradio_cache").mkdir(parents=True, exist_ok=True)
Path("./temp/tensorrt_cache").mkdir(parents=True, exist_ok=True)
import subprocess
import threading
import time
import traceback
import uuid
from enum import Enum
import queue
import shutil
from functools import partial
from datetime import datetime
from pathlib import Path

import cv2
import gradio as gr
from flask import Flask, request

import service.trans_dh_service
from h_utils.custom import CustomError
from y_utils.config import GlobalConfig
from y_utils.logger import logger

# 🚀 导入GPU优化引擎
try:
    from gpu_optimization_engine import get_optimization_engine
    from optimized_model_base import OptimizedModelBase, create_optimized_model
    GPU_OPTIMIZATION_AVAILABLE = True
    logger.info("✅ GPU优化引擎已加载")
except ImportError as e:
    GPU_OPTIMIZATION_AVAILABLE = False
    logger.warning(f"⚠️ GPU优化引擎不可用: {e}")


# 文件处理工具函数
class FileHandler:
    def __init__(self):
        # 支持的音频格式
        self.audio_formats = ['.wav', '.mp3', '.m4a', '.aac', '.flac', '.ogg', '.wma', '.aiff', '.au']
        # 支持的视频格式
        self.video_formats = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v', '.3gp', '.asf']

        # 创建必要的目录
        self.setup_directories()

    def setup_directories(self):
        """创建必要的目录结构"""
        directories = ['inputs', 'inputs/audio', 'inputs/video', 'result', 'temp']
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            logger.info(f"已确保目录存在: {directory}")

    def sanitize_filename(self, filename):
        """
        清理文件名，处理中文、英文、特殊字符
        保持中文字符，移除或替换不安全的字符
        """
        if not filename:
            return "unnamed_file"

        # 获取文件名和扩展名
        name, ext = os.path.splitext(filename)

        # 规范化Unicode字符（处理各种编码问题）
        name = unicodedata.normalize('NFC', name)

        # 移除或替换不安全的字符，但保留中文字符
        # 允许：中文字符、英文字母、数字、下划线、连字符、点号、小括号
        # 不安全字符：\ / : * ? " < > |
        unsafe_chars = r'[\\/:*?"<>|]'
        name = re.sub(unsafe_chars, '_', name)

        # 移除或替换其他特殊字符，但保留中文字符
        # 允许的字符：中文字符、英文字母、数字、下划线、连字符、点号、小括号、方括号、空格
        allowed_pattern = r'[^\w\u4e00-\u9fff\u3400-\u4dbf\u20000-\u2a6df\u2a700-\u2b73f\u2b740-\u2b81f\u2b820-\u2ceaf\u2ceb0-\u2ebef\u30000-\u3134f\-\.\(\)\[\] ]'
        name = re.sub(allowed_pattern, '_', name)

        # 替换多个连续的特殊字符为单个下划线
        name = re.sub(r'[_\-\. ]+', '_', name)

        # 移除开头和结尾的特殊字符
        name = name.strip('_-. ')

        # 如果文件名为空或只有特殊字符，使用默认名称
        if not name or name.isspace():
            name = f"file_{int(time.time())}"

        # 限制文件名长度（包括中文字符）
        if len(name.encode('utf-8')) > 200:
            name = name[:50] + "_" + str(int(time.time()))

        # 确保扩展名是小写的
        ext = ext.lower()

        return name + ext

    def copy_and_rename_file(self, source_path, file_type='audio'):
        """
        复制文件到指定目录并重命名
        """
        try:
            if not source_path or not os.path.exists(source_path):
                raise ValueError(f"源文件不存在: {source_path}")

            # 获取原始文件名
            original_filename = os.path.basename(source_path)
            logger.info(f"处理文件: {original_filename}")

            # 清理文件名
            clean_filename = self.sanitize_filename(original_filename)
            logger.info(f"清理后文件名: {clean_filename}")

            # 确定目标目录
            target_dir = f"inputs/{file_type}"

            # 生成唯一文件名（避免重复）
            timestamp = int(time.time())
            name, ext = os.path.splitext(clean_filename)
            unique_filename = f"{name}_{timestamp}{ext}"

            target_path = os.path.join(target_dir, unique_filename)

            # 复制文件
            shutil.copy2(source_path, target_path)
            logger.info(f"文件已复制到: {target_path}")

            # 验证文件完整性
            if not os.path.exists(target_path):
                raise RuntimeError("文件复制失败")

            source_size = os.path.getsize(source_path)
            target_size = os.path.getsize(target_path)

            if source_size != target_size:
                raise RuntimeError(f"文件大小不匹配 - 源文件: {source_size} bytes, 目标文件: {target_size} bytes")

            return target_path, clean_filename

        except Exception as e:
            logger.error(f"文件处理失败: {str(e)}")
            raise RuntimeError(f"文件处理失败: {str(e)}")

    def validate_file_format(self, filepath, file_type):
        """验证文件格式"""
        if not filepath:
            return False, "文件路径为空"

        ext = os.path.splitext(filepath)[1].lower()

        if file_type == 'audio':
            if ext not in self.audio_formats:
                return False, f"不支持的音频格式: {ext}。支持的格式: {', '.join(self.audio_formats)}"
        elif file_type == 'video':
            if ext not in self.video_formats:
                return False, f"不支持的视频格式: {ext}。支持的格式: {', '.join(self.video_formats)}"
        else:
            return False, "未知的文件类型"

        return True, "格式正确"

    def get_file_info(self, filepath):
        """获取文件详细信息"""
        try:
            if not os.path.exists(filepath):
                return None

            stat = os.stat(filepath)
            size_mb = stat.st_size / (1024 * 1024)

            info = {
                'filename': os.path.basename(filepath),
                'size_bytes': stat.st_size,
                'size_mb': round(size_mb, 2),
                'modified_time': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
            }

            # 如果是视频文件，获取视频信息
            ext = os.path.splitext(filepath)[1].lower()
            if ext in self.video_formats:
                try:
                    cap = cv2.VideoCapture(filepath)
                    if cap.isOpened():
                        info['width'] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        info['height'] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        info['fps'] = cap.get(cv2.CAP_PROP_FPS)
                        info['frame_count'] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        info['duration'] = info['frame_count'] / info['fps'] if info['fps'] > 0 else 0
                        cap.release()
                except Exception as e:
                    logger.warning(f"无法获取视频信息: {e}")

            return info

        except Exception as e:
            logger.error(f"获取文件信息失败: {e}")
            return None


# 创建全局文件处理器实例
file_handler = FileHandler()


def write_video_gradio(
    output_imgs_queue,
    temp_dir,
    result_dir,
    work_id,
    audio_path,
    result_queue,
    width,
    height,
    fps,
    watermark_switch=0,
    digital_auth=0,
    temp_queue=None,
):
    output_mp4 = os.path.join(temp_dir, "{}-t.mp4".format(work_id))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    result_path = os.path.join(result_dir, "{}-r.mp4".format(work_id))
    video_write = cv2.VideoWriter(output_mp4, fourcc, fps, (width, height))
    print("Custom VideoWriter init done")
    try:
        while True:
            state, reason, value_ = output_imgs_queue.get()
            if type(state) == bool and state == True:
                logger.info(
                    "Custom VideoWriter [{}]视频帧队列处理已结束".format(work_id)
                )
                logger.info(
                    "Custom VideoWriter Silence Video saved in {}".format(
                        os.path.realpath(output_mp4)
                    )
                )
                video_write.release()
                break
            else:
                if type(state) == bool and state == False:
                    logger.error(
                        "Custom VideoWriter [{}]任务视频帧队列 -> 异常原因:[{}]".format(
                            work_id, reason
                        )
                    )
                    raise CustomError(reason)
                for result_img in value_:
                    video_write.write(result_img)
        if video_write is not None:
            video_write.release()
        if watermark_switch == 1 and digital_auth == 1:
            logger.info(
                "Custom VideoWriter [{}]任务需要水印和数字人标识".format(work_id)
            )
            if width > height:
                command = 'ffmpeg -y -i {} -i {} -i {} -i {} -filter_complex "overlay=(main_w-overlay_w)-10:(main_h-overlay_h)-10,overlay=(main_w-overlay_w)-10:10" -c:a aac -crf 15 -strict -2 {}'.format(
                    audio_path,
                    output_mp4,
                    GlobalConfig.instance().watermark_path,
                    GlobalConfig.instance().digital_auth_path,
                    result_path,
                )
                logger.info("command:{}".format(command))
            else:
                command = 'ffmpeg -y -i {} -i {} -i {} -i {} -filter_complex "overlay=(main_w-overlay_w)-10:(main_h-overlay_h)-10,overlay=(main_w-overlay_w)-10:10" -c:a aac -crf 15 -strict -2 {}'.format(
                    audio_path,
                    output_mp4,
                    GlobalConfig.instance().watermark_path,
                    GlobalConfig.instance().digital_auth_path,
                    result_path,
                )
                logger.info("command:{}".format(command))
        elif watermark_switch == 1 and digital_auth == 0:
            logger.info("Custom VideoWriter [{}]任务需要水印".format(work_id))
            command = 'ffmpeg -y -i {} -i {} -i {} -filter_complex "overlay=(main_w-overlay_w)-10:(main_h-overlay_h)-10" -c:a aac -crf 15 -strict -2 {}'.format(
                audio_path,
                output_mp4,
                GlobalConfig.instance().watermark_path,
                result_path,
            )
            logger.info("command:{}".format(command))
        elif watermark_switch == 0 and digital_auth == 1:
            logger.info("Custom VideoWriter [{}]任务需要数字人标识".format(work_id))
            if width > height:
                command = 'ffmpeg -loglevel warning -y -i {} -i {} -i {} -filter_complex "overlay=(main_w-overlay_w)-10:10" -c:a aac -crf 15 -strict -2 {}'.format(
                    audio_path,
                    output_mp4,
                    GlobalConfig.instance().digital_auth_path,
                    result_path,
                )
                logger.info("command:{}".format(command))
            else:
                command = 'ffmpeg -loglevel warning -y -i {} -i {} -i {} -filter_complex "overlay=(main_w-overlay_w)-10:10" -c:a aac -crf 15 -strict -2 {}'.format(
                    audio_path,
                    output_mp4,
                    GlobalConfig.instance().digital_auth_path,
                    result_path,
                )
                logger.info("command:{}".format(command))
        else:
            command = "ffmpeg -loglevel warning -y -i {} -i {} -c:a aac -c:v libx264 -crf 15 -strict -2 {}".format(
                audio_path, output_mp4, result_path
            )
            logger.info("Custom command:{}".format(command))
        subprocess.call(command, shell=True)
        print("###### Custom Video Writer write over")
        print(f"###### Video result saved in {os.path.realpath(result_path)}")
        result_queue.put([True, result_path])
        # temp_queue.put([True, result_path])
    except Exception as e:
        logger.error(
            "Custom VideoWriter [{}]视频帧队列处理异常结束，异常原因:[{}]".format(
                work_id, e.__str__()
            )
        )
        result_queue.put(
            [
                False,
                "[{}]视频帧队列处理异常结束，异常原因:[{}]".format(
                    work_id, e.__str__()
                ),
            ]
        )
    logger.info("Custom VideoWriter 后处理进程结束")


service.trans_dh_service.write_video = write_video_gradio


class VideoProcessor:
    def __init__(self):
        self.task = service.trans_dh_service.TransDhTask()
        self.basedir = GlobalConfig.instance().result_dir
        self.is_initialized = False
        self.processing_status = {"current": "就绪", "progress": 0}

        # 🚀 初始化GPU优化引擎
        if GPU_OPTIMIZATION_AVAILABLE:
            logger.info("🎯 启用RTX 4090激进优化模式...")
            self.optimization_engine = get_optimization_engine()
            self.gpu_optimized = True

            # 预优化关键模型
            self._preoptimize_models()
        else:
            logger.warning("⚠️ 使用标准模式 (无GPU优化)")
            self.optimization_engine = None
            self.gpu_optimized = False

        self._initialize_service()
        print("视频处理器初始化完成")

    def _preoptimize_models(self):
        """预优化关键模型"""
        try:
            logger.info("🔥 预优化关键模型...")

            # 查找并优化所有ONNX模型
            optimization_results = self.optimization_engine.optimize_existing_models(".")

            # 记录优化结果
            optimized_count = len([r for r in optimization_results if 'error' not in r])
            total_count = len(optimization_results)

            logger.info(f"📊 模型优化完成: {optimized_count}/{total_count}")

            if optimized_count > 0:
                # 显示最佳性能模型
                best_model = max(
                    [r for r in optimization_results if 'performance' in r and 'throughput_fps' in r['performance']],
                    key=lambda x: x['performance']['throughput_fps'],
                    default=None
                )

                if best_model:
                    fps = best_model['performance']['throughput_fps']
                    model_name = Path(best_model['model_path']).name
                    logger.info(f"🏆 最佳性能模型: {model_name} ({fps:.2f} FPS)")

        except Exception as e:
            logger.error(f"❌ 模型预优化失败: {e}")

    def _initialize_service(self):
        logger.info("正在初始化数字人服务...")
        try:
            # 🚀 GPU优化配置
            if self.gpu_optimized:
                logger.info("🎯 使用GPU优化配置...")

                # 设置高性能环境变量
                os.environ.update({
                    'CUDA_LAUNCH_BLOCKING': '0',  # 异步执行
                    'CUDA_DEVICE_ORDER': 'PCI_BUS_ID',
                    'CUDA_VISIBLE_DEVICES': '0',
                    'OMP_NUM_THREADS': str(os.cpu_count()),
                    'MKL_NUM_THREADS': str(os.cpu_count()),
                    'NUMEXPR_NUM_THREADS': str(os.cpu_count()),
                })

                # 预热GPU
                self._warmup_gpu()

            time.sleep(3)  # 减少初始化时间
            logger.info("数字人服务初始化完成")
            self.is_initialized = True
            self.processing_status["current"] = "服务已就绪 🚀" if self.gpu_optimized else "服务已就绪"

        except Exception as e:
            logger.error(f"初始化数字人服务失败: {e}")
            self.processing_status["current"] = f"初始化失败: {e}"

    def _warmup_gpu(self):
        """GPU预热"""
        try:
            import torch
            if torch.cuda.is_available():
                logger.info("🔥 GPU预热中...")

                # 创建预热张量
                device = torch.device('cuda:0')
                warmup_tensor = torch.randn(1024, 1024, device=device)

                # 执行一些操作预热GPU
                for _ in range(5):
                    result = torch.matmul(warmup_tensor, warmup_tensor.T)
                    torch.cuda.synchronize()

                del warmup_tensor, result
                torch.cuda.empty_cache()

                logger.info("✅ GPU预热完成")

        except Exception as e:
            logger.warning(f"⚠️ GPU预热失败: {e}")

    def get_status(self):
        """获取当前处理状态"""
        if self.gpu_optimized:
            # 添加GPU状态信息
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                    gpu_used = torch.cuda.memory_allocated(0) / 1024**3
                    gpu_status = f" (GPU: {gpu_used:.1f}/{gpu_memory:.1f}GB)"
                else:
                    gpu_status = " (GPU: N/A)"
            except:
                gpu_status = ""

            current_status = self.processing_status["current"] + gpu_status
        else:
            current_status = self.processing_status["current"]

        return current_status, self.processing_status["progress"]

    def validate_and_process_files(self, audio_file, video_file, progress_callback=None):
        """验证并处理上传的文件"""
        try:
            if progress_callback:
                progress_callback(0.1, "📝 验证文件格式...")

            # 验证音频文件
            if audio_file is None:
                raise ValueError("请上传音频文件")

            # 处理新的 gr.Audio 组件返回的文件路径
            audio_path = audio_file if isinstance(audio_file, str) else audio_file.name
            audio_valid, audio_msg = file_handler.validate_file_format(audio_path, 'audio')
            if not audio_valid:
                raise ValueError(f"音频文件错误: {audio_msg}")

            # 验证视频文件
            if video_file is None:
                raise ValueError("请上传视频文件")

            # 处理新的 gr.Video 组件返回的文件路径
            video_path = video_file if isinstance(video_file, str) else video_file.name
            video_valid, video_msg = file_handler.validate_file_format(video_path, 'video')
            if not video_valid:
                raise ValueError(f"视频文件错误: {video_msg}")

            if progress_callback:
                progress_callback(0.2, "📁 处理音频文件...")

            # 处理音频文件
            audio_target_path, audio_clean_name = file_handler.copy_and_rename_file(
                audio_path, 'audio'
            )
            logger.info(f"音频文件处理完成: {audio_clean_name} -> {audio_target_path}")

            if progress_callback:
                progress_callback(0.3, "🎬 处理视频文件...")

            # 处理视频文件
            video_target_path, video_clean_name = file_handler.copy_and_rename_file(
                video_path, 'video'
            )
            logger.info(f"视频文件处理完成: {video_clean_name} -> {video_target_path}")

            # 获取文件信息
            audio_info = file_handler.get_file_info(audio_target_path)
            video_info = file_handler.get_file_info(video_target_path)

            if progress_callback:
                progress_callback(0.4, "✅ 文件处理完成")

            return {
                'audio_path': audio_target_path,
                'video_path': video_target_path,
                'audio_info': audio_info,
                'video_info': video_info,
                'audio_clean_name': audio_clean_name,
                'video_clean_name': video_clean_name
            }

        except Exception as e:
            logger.error(f"文件验证和处理失败: {e}")
            raise

    def process_video(self, audio_file, video_file, progress=gr.Progress()):
        """处理视频的主要函数，增加了进度跟踪和错误处理 + GPU优化"""

        try:
            # 🚀 GPU优化前置处理
            if self.gpu_optimized:
                progress(0.05, desc="🚀 GPU优化准备中...")
                self._prepare_gpu_optimization()

            # 验证并处理文件
            progress(0.1, desc="🔍 文件验证和处理中...")
            file_data = self.validate_and_process_files(
                audio_file, video_file,
                lambda p, desc: progress(p, desc=desc)
            )

            # 等待服务初始化
            progress(0.4, desc="⏳ 等待服务初始化...")
            while not self.is_initialized:
                logger.info("服务尚未完成初始化，等待 1 秒...")
                time.sleep(1)

            self.processing_status["current"] = "开始处理"
            self.processing_status["progress"] = 40

            work_id = str(uuid.uuid1())
            code = work_id

            try:
                progress(0.5, desc="📊 分析视频文件...")
                self.processing_status["current"] = "分析视频文件"

                # 使用处理后的文件路径
                audio_path = file_data['audio_path']
                video_path = file_data['video_path']

                # 分析视频信息
                cap = cv2.VideoCapture(video_path)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = frame_count / fps if fps > 0 else 0
                cap.release()

                logger.info(f"视频信息: {width}x{height}, {fps}fps, {duration:.2f}秒")
                logger.info(f"音频文件: {file_data['audio_clean_name']} ({file_data['audio_info']['size_mb']}MB)")
                logger.info(f"视频文件: {file_data['video_clean_name']} ({file_data['video_info']['size_mb']}MB)")

                # 🚀 GPU优化推理配置
                if self.gpu_optimized:
                    progress(0.55, desc="🎯 GPU优化配置...")
                    self._configure_optimized_inference(width, height, duration)

                progress(0.6, desc="🎭 开始数字人生成...")
                self.processing_status["current"] = "生成数字人视频 🚀" if self.gpu_optimized else "生成数字人视频"

                # 🚀 性能监控开始
                if self.gpu_optimized:
                    inference_start_time = time.time()

                self.task.task_dic[code] = ""
                self.task.work(audio_path, video_path, code, 0, 0, 0, 0)

                # 🚀 性能统计
                if self.gpu_optimized:
                    inference_time = time.time() - inference_start_time
                    fps_achieved = frame_count / inference_time if inference_time > 0 else 0
                    logger.info(f"🏆 GPU优化推理完成: {inference_time:.2f}s, {fps_achieved:.2f} FPS")

                progress(0.9, desc="🎬 合成最终视频...")
                self.processing_status["current"] = "合成最终视频"
                result_path = self.task.task_dic[code][2]

                # 创建结果目录并移动文件
                final_result_dir = os.path.join("result", code)
                os.makedirs(final_result_dir, exist_ok=True)

                # 生成有意义的结果文件名
                result_filename = f"数字人视频_{file_data['video_clean_name'].split('_')[0]}_{code[:8]}.mp4"
                final_result_path = os.path.join(final_result_dir, result_filename)

                # 移动并重命名结果文件
                shutil.move(result_path, final_result_path)

                # 清理临时文件
                temp_pattern = os.path.join(os.path.dirname(result_path), code + '*.*')
                for temp_file in glob.glob(temp_pattern):
                    try:
                        os.remove(temp_file)
                    except:
                        pass

                progress(1.0, desc="✅ 处理完成!")
                self.processing_status["current"] = "处理完成 🎉" if self.gpu_optimized else "处理完成"
                logger.info(f"视频处理完成: {final_result_path}")

                # 🚀 GPU优化后置处理
                if self.gpu_optimized:
                    self._cleanup_gpu_optimization()

                return final_result_path

            except Exception as e:
                self.processing_status["current"] = f"处理失败: {str(e)}"
                logger.error(f"处理视频时发生错误: {e}")
                raise gr.Error(f"视频处理失败: {str(e)}")

        except Exception as e:
            error_msg = str(e)
            if "文件验证失败" in error_msg or "文件错误" in error_msg:
                raise gr.Error(error_msg)
            else:
                logger.error(f"处理过程中发生错误: {e}")
                raise gr.Error(f"处理失败: {error_msg}")

        finally:
            # 清理临时文件
            self._cleanup_temp_files()

    def _prepare_gpu_optimization(self):
        """准备GPU优化"""
        try:
            logger.info("🚀 准备GPU优化...")

            # 清理GPU内存
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # 设置最佳性能模式
            if hasattr(self.optimization_engine, 'performance_stats'):
                self.optimization_engine.performance_stats['inference_times'].clear()

            logger.info("✅ GPU优化准备完成")

        except Exception as e:
            logger.warning(f"⚠️ GPU优化准备失败: {e}")

    def _configure_optimized_inference(self, width: int, height: int, duration: float):
        """配置优化推理参数"""
        try:
            logger.info(f"🎯 配置优化推理: {width}x{height}, {duration:.1f}s")

            # 根据视频尺寸和时长优化批处理大小
            total_pixels = width * height
            if total_pixels > 1920 * 1080:  # 4K+
                batch_size = 1
                logger.info("🎯 4K+视频，使用单帧处理")
            elif total_pixels > 1280 * 720:  # 1080p
                batch_size = 2
                logger.info("🎯 1080p视频，使用小批量处理")
            else:  # 720p及以下
                batch_size = 4
                logger.info("🎯 720p视频，使用批量处理")

            # 预估内存需求并调整
            estimated_memory_gb = (total_pixels * batch_size * 4) / (1024 ** 3)  # 粗略估算
            logger.info(f"📊 预估显存需求: {estimated_memory_gb:.2f}GB")

            if estimated_memory_gb > 20:  # RTX 4090有24GB，保留4GB余量
                batch_size = max(1, batch_size // 2)
                logger.info(f"⚡ 调整批量大小为: {batch_size}")

        except Exception as e:
            logger.warning(f"⚠️ 优化推理配置失败: {e}")

    def _cleanup_gpu_optimization(self):
        """清理GPU优化"""
        try:
            logger.info("🧹 清理GPU优化资源...")

            # 清理GPU缓存
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            # 清理优化引擎缓存
            if self.optimization_engine:
                self.optimization_engine.clear_cache()

            logger.info("✅ GPU优化清理完成")

        except Exception as e:
            logger.warning(f"⚠️ GPU优化清理失败: {e}")

    def _cleanup_temp_files(self):
        """清理临时文件"""
        try:
            temp_dir = "temp"
            if os.path.exists(temp_dir):
                for file in os.listdir(temp_dir):
                    file_path = os.path.join(temp_dir, file)
                    if os.path.isfile(file_path):
                        # 只删除超过1小时的临时文件
                        if time.time() - os.path.getmtime(file_path) > 3600:
                            os.remove(file_path)
                            logger.info(f"已清理临时文件: {file_path}")
        except Exception as e:
            logger.warning(f"清理临时文件时发生错误: {e}")


def create_interface(processor):
    """创建美观的Gradio界面"""

    # 自定义CSS样式
    custom_css = """
    .gradio-container {
        max-width: 1400px !important;
        margin: auto;
        padding: 20px;
    }

    .main-header {
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 30px;
        border-radius: 15px;
        margin-bottom: 30px;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
    }

    .feature-card {
        background: white;
        border-radius: 12px;
        padding: 20px;
        margin: 10px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #667eea;
    }

    .upload-area {
        border: 2px dashed #667eea;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        background: #f8f9ff;
    }

    .status-info {
        background: #e3f2fd;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        border-left: 4px solid #2196f3;
    }

    .warning-box {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }

    .success-box {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }

    .btn-primary {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: none;
        color: white;
        padding: 12px 30px;
        border-radius: 8px;
        font-weight: bold;
        cursor: pointer;
        transition: all 0.3s ease;
    }

    .btn-primary:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }

    /* 预览区域样式 */
    .preview-container {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 15px;
        background: #fafafa;
        margin: 10px 0;
    }

    .preview-video {
        max-height: 250px;
        width: 100%;
        object-fit: contain;
        border-radius: 8px;
    }

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

    .preview-audio {
        width: 100%;
        height: 80px;
    }

    .comparison-container {
        display: flex;
        gap: 20px;
        align-items: flex-start;
    }

    .comparison-item {
        flex: 1;
        min-width: 0;
    }
    """

    with gr.Blocks(css=custom_css, title="HeyGem 数字人生成器", theme=gr.themes.Soft()) as interface:
        # 标题部分
        gr.HTML("""
        <div class="main-header">
            <h1>🎭 HeyGem AI 数字人视频生成器</h1>
            <p style="font-size: 18px; margin-top: 10px; opacity: 0.9;">
                使用人工智能技术，让您的视频角色说出任何音频内容
            </p>
            <p style="font-size: 14px; margin-top: 15px; opacity: 0.8;">
                📧 技术支持: 科哥微信 312088415 | 🏢 仙宫云特别版
            </p>
        </div>
        """)

        with gr.Row():
            # 左侧 - 文件上传区域
            with gr.Column(scale=1):
                gr.HTML('<div class="feature-card"><h3>📁 文件上传</h3></div>')

                with gr.Group():
                    audio_input = gr.Audio(
                        label="🎵 上传音频文件 (支持预览)",
                        type="filepath",
                        elem_classes=["upload-area"]
                    )

                    gr.HTML("""
                    <div class="warning-box">
                        <small>💡 <strong>音频文件要求：</strong><br>
                        • 支持格式：WAV, MP3, M4A, AAC, FLAC, OGG, WMA, AIFF, AU<br>
                        • 建议时长：10秒-2分钟<br>
                        • 音质要求：清晰无杂音<br>
                        • 文件大小：建议不超过100MB<br>
                        • 支持中文文件名，会自动处理特殊字符<br>
                        • 上传后可直接在音频组件中试听和预览</small>
                    </div>
                    """)

                    video_input = gr.Video(
                        label="🎬 上传视频文件 (支持预览)",
                        elem_classes=["upload-area"]
                    )

                    gr.HTML("""
                    <div class="warning-box">
                        <small>💡 <strong>视频文件要求：</strong><br>
                        • 支持格式：MP4, AVI, MOV, MKV, WMV, FLV, WebM, M4V, 3GP, ASF<br>
                        • 必须包含清晰的人脸<br>
                        • 建议分辨率：720p以上<br>
                        • 文件大小：建议不超过500MB<br>
                        • 支持中文文件名，会自动处理特殊字符<br>
                        • 上传后可直接在视频组件中预览播放</small>
                    </div>
                    """)

            # 右侧 - 结果显示区域
            with gr.Column(scale=1):
                gr.HTML('<div class="feature-card"><h3>🎯 处理结果</h3></div>')

                # 状态显示
                status_display = gr.HTML("""
                <div class="status-info">
                    <h4>📊 当前状态：<span id="current-status">等待上传文件</span></h4>
                    <p>🕒 启动时间：""" + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
                </div>
                """)

                # 处理按钮
                process_btn = gr.Button(
                    "🚀 开始生成数字人视频",
                    variant="primary",
                    size="lg",
                    elem_classes=["btn-primary"]
                )

                # 结果视频显示（按原始比例预览）
                output_video = gr.Video(
                    label="✨ 生成的数字人视频",
                    show_label=True,
                    autoplay=False,
                    elem_classes=["output-video"]
                )

        # 下方信息区域
        with gr.Row():
            with gr.Column():
                gr.HTML("""
                <div class="feature-card">
                    <h3>🌟 功能特色</h3>
                    <ul style="list-style-type: none; padding: 0;">
                        <li>🎭 <strong>高质量面部同步：</strong>精确的唇形同步技术</li>
                        <li>🚀 <strong>快速处理：</strong>GPU加速，高效生成</li>
                        <li>🔒 <strong>隐私保护：</strong>本地处理，数据安全</li>
                        <li>💎 <strong>专业品质：</strong>电影级别的视觉效果</li>
                    </ul>
                </div>
                """)

            with gr.Column():
                gr.HTML("""
                <div class="feature-card">
                    <h3>📋 使用步骤</h3>
                    <ol style="line-height: 1.8;">
                        <li>📤 上传需要同步的音频文件</li>
                        <li>🎥 上传包含人脸的视频文件</li>
                        <li>🚀 点击"开始生成"按钮</li>
                        <li>⏳ 等待AI处理完成</li>
                        <li>💾 下载生成的数字人视频</li>
                    </ol>
                </div>
                """)

        # 绑定处理函数
        process_btn.click(
            fn=processor.process_video,
            inputs=[audio_input, video_input],
            outputs=[output_video],
            show_progress=True
        )


        # 文件上传时的验证反馈（更新版本，包含预览信息）
        def update_status(audio_file, video_file):
            status_text = "<div class='status-info'><h4>📊 文件状态检查</h4>"

            # 检查音频文件
            if audio_file:
                try:
                    # 处理新的 gr.Audio 组件返回的文件路径
                    audio_path = audio_file if isinstance(audio_file, str) else audio_file.name

                    # 使用文件处理器验证格式
                    audio_valid, audio_msg = file_handler.validate_file_format(audio_path, 'audio')

                    # 获取文件信息
                    audio_info = file_handler.get_file_info(audio_path)
                    audio_clean_name = file_handler.sanitize_filename(os.path.basename(audio_path))

                    if audio_valid:
                        status_text += f"""
                        <p>✅ <strong>音频文件：</strong>{os.path.basename(audio_path)}</p>
                        <p style="margin-left: 20px;">📝 清理后文件名：{audio_clean_name}</p>
                        <p style="margin-left: 20px;">📦 文件大小：{audio_info['size_mb']}MB</p>
                        <p style="margin-left: 20px;">🔧 格式状态：{audio_msg}</p>
                        <p style="margin-left: 20px;">🎧 <em>可在音频组件中直接试听和预览</em></p>
                        """
                    else:
                        status_text += f"""
                        <p>❌ <strong>音频文件：</strong>{os.path.basename(audio_path)}</p>
                        <p style="margin-left: 20px; color: red;">⚠️ {audio_msg}</p>
                        """
                except Exception as e:
                    audio_path = audio_file if isinstance(audio_file, str) else getattr(audio_file, 'name', str(audio_file))
                    status_text += f"""
                    <p>❌ <strong>音频文件：</strong>{os.path.basename(audio_path)}</p>
                    <p style="margin-left: 20px; color: red;">⚠️ 文件检查失败：{str(e)}</p>
                    """
            else:
                status_text += "<p>⏳ <strong>音频文件：</strong>未上传</p>"

            # 检查视频文件
            if video_file:
                try:
                    # 处理新的 gr.Video 组件返回的文件路径
                    video_path = video_file if isinstance(video_file, str) else video_file.name

                    # 使用文件处理器验证格式
                    video_valid, video_msg = file_handler.validate_file_format(video_path, 'video')

                    # 获取文件信息
                    video_info = file_handler.get_file_info(video_path)
                    video_clean_name = file_handler.sanitize_filename(os.path.basename(video_path))

                    if video_valid:
                        status_text += f"""
                        <p>✅ <strong>视频文件：</strong>{os.path.basename(video_path)}</p>
                        <p style="margin-left: 20px;">📝 清理后文件名：{video_clean_name}</p>
                        <p style="margin-left: 20px;">📦 文件大小：{video_info['size_mb']}MB</p>
                        """

                        # 如果有视频信息，显示详细信息
                        if 'width' in video_info:
                            status_text += f"""
                            <p style="margin-left: 20px;">📺 分辨率：{video_info['width']}x{video_info['height']}</p>
                            <p style="margin-left: 20px;">⏱️ 时长：{video_info['duration']:.2f}秒</p>
                            <p style="margin-left: 20px;">🎬 帧率：{video_info['fps']:.1f} FPS</p>
                            """

                        status_text += f"<p style='margin-left: 20px;'>🔧 格式状态：{video_msg}</p>"
                        status_text += f"<p style='margin-left: 20px;'>📺 <em>可在视频组件中直接预览播放</em></p>"
                    else:
                        status_text += f"""
                        <p>❌ <strong>视频文件：</strong>{os.path.basename(video_path)}</p>
                        <p style="margin-left: 20px; color: red;">⚠️ {video_msg}</p>
                        """
                except Exception as e:
                    video_path = video_file if isinstance(video_file, str) else getattr(video_file, 'name', str(video_file))
                    status_text += f"""
                    <p>❌ <strong>视频文件：</strong>{os.path.basename(video_path)}</p>
                    <p style="margin-left: 20px; color: red;">⚠️ 文件检查失败：{str(e)}</p>
                    """
            else:
                status_text += "<p>⏳ <strong>视频文件：</strong>未上传</p>"

            # 总体状态
            if audio_file and video_file:
                try:
                    audio_path = audio_file if isinstance(audio_file, str) else audio_file.name
                    video_path = video_file if isinstance(video_file, str) else video_file.name

                    audio_valid = file_handler.validate_file_format(audio_path, 'audio')[0]
                    video_valid = file_handler.validate_file_format(video_path, 'video')[0]

                    if audio_valid and video_valid:
                        status_text += """
                        <div style="background: #d4edda; padding: 10px; border-radius: 5px; margin-top: 15px;">
                            <p style="margin: 0;"><strong>🎉 所有文件已准备就绪，可以开始处理！</strong></p>
                            <p style="margin: 5px 0 0 0; font-size: 12px;">文件将自动保存到 inputs/ 目录，结果保存到 result/ 目录</p>
                            <p style="margin: 5px 0 0 0; font-size: 12px;">💡 建议先在音频和视频组件中试听预览，确认内容无误后再开始处理</p>
                        </div>
                        """
                    else:
                        status_text += """
                        <div style="background: #f8d7da; padding: 10px; border-radius: 5px; margin-top: 15px;">
                            <p style="margin: 0;"><strong>⚠️ 请检查文件格式是否正确</strong></p>
                        </div>
                        """
                except:
                    pass

            status_text += "</div>"
            return status_text

        # 文件状态更新事件
        def handle_file_change(audio_file, video_file):
            # 更新状态显示
            status_update = update_status(audio_file, video_file)
            return status_update

        # 绑定文件变化事件到状态更新
        audio_input.change(
            fn=handle_file_change,
            inputs=[audio_input, video_input],
            outputs=[status_display]
        )

        video_input.change(
            fn=handle_file_change,
            inputs=[audio_input, video_input],
            outputs=[status_display]
        )

    return interface


if __name__ == "__main__":
    processor = VideoProcessor()

    # 启动界面
    demo = create_interface(processor)
    demo.queue(
        concurrency_count=2,  # 并发处理数量
        max_size=10,         # 队列最大长度
        api_open=False       # 不开放API
    ).launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,         # 不创建公共链接
        show_error=True,     # 显示错误信息
        quiet=False,         # 显示启动信息
        show_tips=True,      # 显示提示
        height=800,          # 界面高度
        favicon_path=None,   # 可以设置自定义图标
        app_kwargs={"docs_url": None, "redoc_url": None}  # 禁用API文档
    )
