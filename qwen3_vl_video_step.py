vllm serve Qwen3-VL-4B-Instruct \
  --tensor-parallel-size 1 \
  --limit-mm-per-prompt.video 0 \
  --async-scheduling \
  --gpu-memory-utilization 0.95 \
  --max-model-len 20480

import cv2
import base64
import time
import os
from pathlib import Path
from openai import OpenAI
import argparse
import json
from typing import List, Optional, Union
from PIL import Image
import numpy as np

class MediaDescriber:
    def __init__(self, api_key: str = "EMPTY", base_url: str = "http://localhost:8000/v1", timeout: int = 3600):
        """
        初始化媒体描述生成器
        
        Args:
            api_key: API密钥
            base_url: API基础URL
            timeout: 请求超时时间
        """
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout
        )
    
    def extract_video_frames(self, video_path: str, frame_interval: int = 25, max_frames: int = 20) -> List[str]:
        """
        从视频中提取帧并编码为base64 [1,2](@ref)
        
        Args:
            video_path: 视频文件路径
            frame_interval: 帧采样间隔
            max_frames: 最大帧数限制
            
        Returns:
            base64编码的帧列表
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"视频文件不存在: {video_path}")
        
        video = cv2.VideoCapture(video_path)
        base64_frames = []
        frame_count = 0
        
        print(f"正在从视频中提取帧: {video_path}")
        
        while video.isOpened():
            success, frame = video.read()
            if not success:
                break
            
            # 按间隔采样帧
            if frame_count % frame_interval == 0 and len(base64_frames) < max_frames:
                # 调整帧大小以优化处理
                frame = cv2.resize(frame, (768, 432))  # 保持16:9比例
                _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                base64_frames.append(base64.b64encode(buffer).decode("utf-8"))
            
            frame_count += 1
        
        video.release()
        
        print(f"成功提取 {len(base64_frames)} 帧 (总帧数: {frame_count})")
        return base64_frames
    
    def process_image_file(self, image_path: str, max_size: tuple = (768, 432)) -> List[str]:
        """
        处理单张图片并编码为base64 [2](@ref)
        
        Args:
            image_path: 图片文件路径
            max_size: 最大尺寸限制
            
        Returns:
            base64编码的图片列表
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图片文件不存在: {image_path}")
        
        print(f"正在处理图片: {image_path}")
        
        # 使用PIL打开图片并调整大小
        img = Image.open(image_path)
        img = img.convert('RGB')
        img.thumbnail(max_size, Image.Resampling.LANCZOS)
        
        # 转换为OpenCV格式
        img_cv = np.array(img)
        img_cv = img_cv[:, :, ::-1].copy()  # RGB转BGR
        
        # 编码为base64
        _, buffer = cv2.imencode(".jpg", img_cv, [cv2.IMWRITE_JPEG_QUALITY, 80])
        base64_image = base64.b64encode(buffer).decode("utf-8")
        
        return [base64_image]
    
    def extract_media_frames(self, media_path: str, **kwargs) -> List[str]:
        """
        统一媒体处理入口：自动检测类型并提取帧 [1,2](@ref)
        
        Args:
            media_path: 媒体文件路径
            **kwargs: 其他参数
            
        Returns:
            base64编码的帧列表
        """
        # 检查文件是否存在
        if not os.path.exists(media_path):
            raise FileNotFoundError(f"媒体文件不存在: {media_path}")
        
        # 获取文件扩展名并转换为小写
        file_ext = Path(media_path).suffix.lower()
        
        # 视频格式列表 [1,4](@ref)
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.m4v', '.webm']
        # 图片格式列表 [4](@ref)
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp']
        
        if file_ext in video_extensions:
            print("检测到视频文件，使用视频处理模式")
            return self.extract_video_frames(media_path, **kwargs)
        elif file_ext in image_extensions:
            print("检测到图片文件，使用图片处理模式")
            return self.process_image_file(media_path)
        else:
            raise ValueError(f"不支持的媒体格式: {file_ext}。支持格式: {video_extensions + image_extensions}")
    
    def generate_media_description(self, base64_frames: List[str], 
                                 media_type: str = "auto",
                                 model: str = "gpt-4-vision-preview",
                                 prompt: str = None,
                                 max_tokens: int = 500,
                                 temperature: float = 0.7) -> str:
        """
        使用视觉模型生成媒体描述 [6,7](@ref)
        
        Args:
            base64_frames: base64编码的媒体帧列表
            media_type: 媒体类型 (video/image/auto)
            model: 使用的模型名称
            prompt: 自定义提示词
            max_tokens: 最大token数
            temperature: 温度参数
            
        Returns:
            媒体描述文本
        """
        if not base64_frames:
            raise ValueError("帧列表不能为空")
        
        # 自动检测媒体类型
        if media_type == "auto":
            media_type = "视频" if len(base64_frames) > 1 else "图片"
        
        # 默认提示词模板 [8](@ref)
        default_prompts = {
            "video": """请仔细分析这些视频帧，生成一个全面、生动的视频描述。
描述应该包含以下要素：
1. 视频的主要内容和主题
2. 场景设置和环境描述
3. 出现的对象、人物或动物及其行为
4. 视觉风格、色彩和光线特点
5. 整体氛围和情感基调

请用中文提供详细、专业的描述。""",
            
            "image": """请仔细分析这张图片，生成一个全面、生动的图片描述。
描述应该包含以下要素：
1. 图片的主要内容和主题
2. 场景设置和环境描述
3. 出现的对象、人物或动物及其特征
4. 构图、视觉风格、色彩和光线特点
5. 整体氛围和情感基调

请用中文提供详细、专业的描述。"""
        }
        
        # 选择提示词
        if prompt:
            user_prompt = prompt
        else:
            prompt_key = "video" if len(base64_frames) > 1 else "image"
            user_prompt = default_prompts.get(prompt_key, default_prompts["image"])
        
        # 构建消息内容
        content = [{"type": "text", "text": user_prompt}]
        
        # 限制帧数量以避免token超限
        max_frames_to_send = 10 if len(base64_frames) > 1 else 1
        sampled_frames = base64_frames[:max_frames_to_send]
        
        for i, frame in enumerate(sampled_frames):
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{frame}",
                    "detail": "low"  # 使用低细节以节省token
                }
            })
        
        try:
            print(f"正在使用模型 {model} 生成{media_type}描述...")
            start_time = time.time()
            
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": content
                    }
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            processing_time = time.time() - start_time
            print(f"描述生成完成，耗时: {processing_time:.2f}秒")
            
            description = response.choices[0].message.content
            return description.strip()
            
        except Exception as e:
            print(f"生成{media_type}描述时出错: {e}")
            return ""
    
    def process_media(self, media_path: str, 
                     output_dir: str = "output",
                     model: str = "gpt-4-vision-preview",
                     media_type: str = "auto",
                     frame_interval: int = 25,
                     max_frames: int = 20,
                     prompt: str = None,
                     max_tokens: int = 500,
                     temperature: float = 0.7) -> dict:
        """
        完整处理媒体文件并生成描述 [6,7](@ref)
        
        Args:
            media_path: 输入媒体文件路径
            output_dir: 输出目录
            model: 使用的模型名称
            media_type: 媒体类型 (video/image/auto)
            frame_interval: 帧采样间隔（仅视频）
            max_frames: 最大帧数限制（仅视频）
            prompt: 自定义提示词
            max_tokens: 最大token数
            temperature: 温度参数
            
        Returns:
            处理结果字典
        """
        # 创建输出目录
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 提取媒体帧
        extract_args = {}
        if media_type in ["auto", "video"]:
            extract_args = {
                "frame_interval": frame_interval,
                "max_frames": max_frames
            }
        
        base64_frames = self.extract_media_frames(media_path, **extract_args)
        
        if not base64_frames:
            raise ValueError("无法从媒体文件中提取内容或文件为空")
        
        # 确定媒体类型
        detected_type = "video" if len(base64_frames) > 1 else "image"
        if media_type == "auto":
            media_type = detected_type
        
        results = {
            'media_path': media_path,
            'media_type': media_type,
            'frames_extracted': len(base64_frames),
            'model_used': model
        }
        
        # 生成媒体描述
        print(f"正在生成{media_type}描述...")
        description = self.generate_media_description(
            base64_frames=base64_frames,
            media_type=media_type,
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        results['description'] = description
        results['description_length'] = len(description)
        
        print(f"{media_type.capitalize()}描述生成完成!")
        print(f"描述长度: {len(description)} 字符")
        print(f"描述内容: {description}")
        
        # 保存文本结果
        timestamp = int(time.time())
        media_name = Path(media_path).stem
        text_output_path = output_path / f"{media_type}_description_{media_name}_{timestamp}.txt"
        json_output_path = output_path / f"{media_type}_results_{media_name}_{timestamp}.json"
        
        # 保存纯文本描述
        with open(text_output_path, 'w', encoding='utf-8') as f:
            f.write(f"=== {media_type.capitalize()}分析结果 ===\n\n")
            f.write(f"媒体文件: {media_path}\n")
            f.write(f"媒体类型: {media_type}\n")
            f.write(f"分析模型: {model}\n")
            f.write(f"分析时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"提取帧数: {len(base64_frames)}\n")
            f.write(f"\n=== {media_type.capitalize()}描述 ===\n\n")
            f.write(description + "\n")
        
        # 保存JSON结果
        with open(json_output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\n=== 处理完成 ===")
        print(f"文本结果已保存: {text_output_path}")
        print(f"完整结果已保存: {json_output_path}")
        
        return results

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="媒体描述生成脚本（支持图片和视频）")
    parser.add_argument("media_path", help="输入媒体文件路径（支持图片和视频）")
    parser.add_argument("--media-type", choices=["auto", "video", "image"], default="auto",
                       help="媒体类型：auto(自动检测)/video/image")
    parser.add_argument("--output-dir", default="output", help="输出目录")
    parser.add_argument("--model", default="gpt-4-vision-preview", help="使用的模型名称")
    parser.add_argument("--frame-interval", type=int, default=25, 
                       help="帧采样间隔（仅视频，默认: 25）")
    parser.add_argument("--max-frames", type=int, default=20, 
                       help="最大帧数限制（仅视频，默认: 20）")
    parser.add_argument("--max-tokens", type=int, default=500, 
                       help="最大token数（默认: 500）")
    parser.add_argument("--temperature", type=float, default=0.7, 
                       help="温度参数（默认: 0.7）")
    parser.add_argument("--prompt", help="自定义提示词")
    parser.add_argument("--api-key", default="EMPTY", help="API密钥")
    parser.add_argument("--base-url", default="http://localhost:8000/v1", 
                       help="API基础URL")
    parser.add_argument("--timeout", type=int, default=3600, help="请求超时时间")
    
    args = parser.parse_args()
    
    try:
        # 初始化描述器
        describer = MediaDescriber(
            api_key=args.api_key,
            base_url=args.base_url,
            timeout=args.timeout
        )
        
        # 处理媒体文件
        results = describer.process_media(
            media_path=args.media_path,
            output_dir=args.output_dir,
            model=args.model,
            media_type=args.media_type,
            frame_interval=args.frame_interval,
            max_frames=args.max_frames,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature
        )
        
        print("\n=== 处理摘要 ===")
        print(f"媒体文件: {results['media_path']}")
        print(f"媒体类型: {results['media_type']}")
        print(f"使用模型: {results['model_used']}")
        print(f"提取帧数: {results['frames_extracted']}")
        print(f"描述长度: {results['description_length']} 字符")
        
    except Exception as e:
        print(f"处理媒体文件时出错: {e}")
        return 1
    
    return 0

'''
if __name__ == "__main__":
    # 使用示例
    if False:  # 设置为True可直接运行示例
        describer = MediaDescriber(
            api_key="EMPTY",
            base_url="http://localhost:8000/v1",
            timeout=3600
        )
        
        # 处理示例视频
        video_results = describer.process_media(
            media_path="your_video.mp4",
            model="Qwen3-VL-4B-Instruct",
            media_type="auto"
        )
        
        # 处理示例图片
        image_results = describer.process_media(
            media_path="your_image.jpg",
            model="Qwen3-VL-4B-Instruct",
            media_type="auto"
        )
    else:
        exit(main())
'''

describer = MediaDescriber(
            api_key="EMPTY",
            base_url="http://localhost:8000/v1",
            timeout=3600
        )

results = describer.process_media(
            media_path="Little_Xiang_Lookalike_Videos/0AqKqJYU6lBAKyfI.mp4",
            model="Qwen3-VL-4B-Instruct",
            max_tokens = 1024,
            frame_interval = 10,
            max_frames = 10,
            media_type="auto",
            prompt = "使用中文描述视频内容，视频的主体为一个男孩，使用'男孩'进行称呼，描述男孩的外貌、动作和背景信息，只描述结果，不进行任何其他讨论。",
        )
print(results["description"])


results = describer.process_media(
            media_path="Little_Xiang_Lookalike_Images/G3n28NwWkAA7n9B.jpg",
            model="Qwen3-VL-4B-Instruct",
            max_tokens = 1024,
            frame_interval = 10,
            max_frames = 10,
            media_type="auto",
            prompt = "使用中文描述图片内容，图片的主体为一个男孩，使用'男孩'进行称呼，描述男孩的外貌、动作和背景信息，只描述结果，不进行任何其他讨论。",
        )
print(results["description"])




