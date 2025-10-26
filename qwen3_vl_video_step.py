#!/usr/bin/env python3
"""
视频描述生成脚本
使用GPT视觉能力分析视频内容并生成描述
支持自定义模型和API端点
"""



import cv2
import base64
import time
import os
from pathlib import Path
from openai import OpenAI
import argparse
import json
from typing import List, Optional

class VideoDescriber:
    def __init__(self, api_key: str = "EMPTY", base_url: str = "http://localhost:8000/v1", timeout: int = 3600):
        """
        初始化视频描述生成器
        
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
    
    def extract_frames(self, video_path: str, frame_interval: int = 25, max_frames: int = 20) -> List[str]:
        """
        从视频中提取帧并编码为base64
        
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
    
    def generate_video_description(self, base64_frames: List[str], 
                                 model: str = "gpt-4-vision-preview",
                                 prompt: str = None,
                                 max_tokens: int = 500,
                                 temperature: float = 0.7) -> str:
        """
        使用视觉模型生成视频描述
        
        Args:
            base64_frames: base64编码的视频帧列表
            model: 使用的模型名称
            prompt: 自定义提示词
            max_tokens: 最大token数
            temperature: 温度参数
            
        Returns:
            视频描述文本
        """
        if not base64_frames:
            raise ValueError("帧列表不能为空")
        
        default_prompt = """请仔细分析这些视频帧，生成一个全面、生动的视频描述。
描述应该包含以下要素：
1. 视频的主要内容和主题
2. 场景设置和环境描述
3. 出现的对象、人物或动物及其行为
4. 视觉风格、色彩和光线特点
5. 整体氛围和情感基调

请用中文提供详细、专业的描述。"""
        
        user_prompt = prompt or default_prompt
        
        # 构建消息内容
        content = [{"type": "text", "text": user_prompt}]
        
        # 添加图像帧（限制数量以避免token超限）
        sampled_frames = base64_frames[:10]  # 最多使用10帧
        
        for i, frame in enumerate(sampled_frames):
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{frame}",
                    "detail": "low"  # 使用低细节以节省token
                }
            })
        
        try:
            print(f"正在使用模型 {model} 生成视频描述...")
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
            print(f"生成视频描述时出错: {e}")
            return ""
    
    def process_video(self, video_path: str, 
                     output_dir: str = "output",
                     model: str = "gpt-4-vision-preview",
                     frame_interval: int = 25,
                     max_frames: int = 20,
                     prompt: str = None,
                     max_tokens: int = 500,
                     temperature: float = 0.7) -> dict:
        """
        完整处理视频并生成描述
        
        Args:
            video_path: 输入视频路径
            output_dir: 输出目录
            model: 使用的模型名称
            frame_interval: 帧采样间隔
            max_frames: 最大帧数限制
            prompt: 自定义提示词
            max_tokens: 最大token数
            temperature: 温度参数
            
        Returns:
            处理结果字典
        """
        # 创建输出目录
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 提取视频帧
        base64_frames = self.extract_frames(video_path, frame_interval, max_frames)
        
        if not base64_frames:
            raise ValueError("无法从视频中提取帧或视频为空")
        
        results = {
            'video_path': video_path,
            'frames_extracted': len(base64_frames),
            'model_used': model
        }
        
        # 生成视频描述
        print("正在生成视频描述...")
        description = self.generate_video_description(
            base64_frames=base64_frames,
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        results['description'] = description
        results['description_length'] = len(description)
        
        print("视频描述生成完成!")
        print(f"描述长度: {len(description)} 字符")
        print(f"描述内容: {description}")
        
        # 保存文本结果
        timestamp = int(time.time())
        text_output_path = output_path / f"video_description_{timestamp}.txt"
        json_output_path = output_path / f"results_{timestamp}.json"
        
        # 保存纯文本描述
        with open(text_output_path, 'w', encoding='utf-8') as f:
            f.write("=== 视频分析结果 ===\n\n")
            f.write(f"视频文件: {video_path}\n")
            f.write(f"分析模型: {model}\n")
            f.write(f"分析时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"提取帧数: {len(base64_frames)}\n")
            f.write("\n=== 视频描述 ===\n\n")
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
    parser = argparse.ArgumentParser(description="视频描述生成脚本")
    parser.add_argument("video_path", help="输入视频文件路径")
    parser.add_argument("--output-dir", default="output", help="输出目录")
    parser.add_argument("--model", default="gpt-4-vision-preview", help="使用的模型名称")
    parser.add_argument("--frame-interval", type=int, default=25, 
                       help="帧采样间隔（默认: 25）")
    parser.add_argument("--max-frames", type=int, default=20, 
                       help="最大帧数限制（默认: 20）")
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
        describer = VideoDescriber(
            api_key=args.api_key,
            base_url=args.base_url,
            timeout=args.timeout
        )
        
        # 处理视频
        results = describer.process_video(
            video_path=args.video_path,
            output_dir=args.output_dir,
            model=args.model,
            frame_interval=args.frame_interval,
            max_frames=args.max_frames,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature
        )
        
        print("\n=== 处理摘要 ===")
        print(f"视频文件: {results['video_path']}")
        print(f"使用模型: {results['model_used']}")
        print(f"提取帧数: {results['frames_extracted']}")
        print(f"描述长度: {results['description_length']} 字符")
        
    except Exception as e:
        print(f"处理视频时出错: {e}")
        return 1
    
    return 0

'''
if __name__ == "__main__":
    # 使用默认配置的示例
    if False:  # 设置为True可直接运行示例
        describer = VideoDescriber(
            api_key="EMPTY",
            base_url="http://localhost:8000/v1",
            timeout=3600
        )
        
        # 处理示例视频
        results = describer.process_video(
            video_path="your_video.mp4",
            model="gpt-4-vision-preview"
        )
    else:
        exit(main())
'''

describer = VideoDescriber(
            api_key="EMPTY",
            base_url="http://localhost:8000/v1",
            timeout=3600
        )


results = describer.process_video(
            video_path="Little_Xiang_Lookalike_Videos/0AqKqJYU6lBAKyfI.mp4",
            model="Qwen3-VL-4B-Instruct",
            max_tokens = 1024,
            frame_interval = 10,
            max_frames = 10,
            prompt = "使用中文描述视频内容，视频的主体为一个男孩，使用'男孩'进行称呼，描述男孩的外貌、动作和背景信息，只描述结果，不进行任何其他讨论。",
        )


