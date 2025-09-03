'''
pip install moviepy==1.0.3

git clone https://huggingface.co/datasets/svjack/Prince_Ciel_Phantomhive_Videos_Captioned

python run_qwen_vl_video_caption.py Prince_Ciel_Phantomhive_Videos_Captioned Prince_Ciel_Phantomhive_Videos_Qwen_VL_Captioned \
--system_content "You are QwenVL, you are a helpful assistant expert in turning images into words. 给你的视频中可能出现的主要人物为两个（可能出现一个或两个），当人物为一个戴眼罩的男孩时，男孩的名字是'夏尔',当人物是一个穿燕尾西服的成年男子时，男子的名字是'塞巴斯蒂安',在你的视频描述中要使用人物的名字并且简单描述人物的外貌及衣着。" --text "使用戴人物名字的中文描述视频内容"

python txt_fix.py --source Prince_Ciel_Phantomhive_Videos_Qwen_VL_Captioned \
--output Prince_Ciel_Phantomhive_Videos_qwen_vl_captioned \
--search "塞巴斯蒂安" --replace "夏尔"

git clone https://huggingface.co/datasets/svjack/Sebastian_Michaelis_Videos_Captioned

python run_qwen_vl_video_caption.py Sebastian_Michaelis_Videos_Captioned \ 
Sebastian_Michaelis_Videos_Qwen_VL_Captioned \
--system_content "You are QwenVL, you are a helpful assistant expert in turning images into words. 给你的视频中可能出现的主要人物为两个（可能出现一个或两个），当人物为一个戴眼罩的男孩时，男孩的名字是'夏尔',当人物是一个穿燕尾西服的成年男子时，男子的名字是'塞巴斯蒂安',在你的视频描述中要使用人物的名字并且简单描述人物的外貌及衣着。" --text "使用戴人物名字的中文描述视频内容"

python txt_fix.py --source Sebastian_Michaelis_Videos_Qwen_VL_Captioned \
--output Sebastian_Michaelis_Videos_qwen_vl_captioned \
--search "夏尔" --replace "塞巴斯蒂安"
'''

import os
import torch
import argparse
from pathlib import Path
import shutil
from torchvision.transforms import ToPILImage
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)
from qwen_vl_utils import process_vision_info

from moviepy.editor import VideoFileClip

def get_video_duration(video_path):
    """Get duration of video in seconds"""
    try:
        with VideoFileClip(video_path) as video:
            return video.duration
    except:
        return float('inf')  # Return infinity if there's an error reading the file

class Qwen2_VQA:
    def __init__(self, args):
        self.args = args
        self.model_checkpoint = None
        self.processor = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bf16_support = (
            torch.cuda.is_available()
            and torch.cuda.get_device_capability(self.device)[0] >= 8
        )

    def setup_model(self):
        """设置和加载模型"""
        model_id = f"qwen/{self.args.model}"
        self.model_checkpoint = os.path.join(
            "models", "prompt_generator", os.path.basename(model_id)
        )

        if not os.path.exists(self.model_checkpoint):
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id=model_id,
                local_dir=self.model_checkpoint,
                local_dir_use_symlinks=False,
            )

        if self.processor is None:
            self.processor = AutoProcessor.from_pretrained(
                self.model_checkpoint, 
                min_pixels=self.args.min_pixels, 
                max_pixels=self.args.max_pixels
            )

        if self.model is None:
            if self.args.quantization == "4bit":
                quantization_config = BitsAndBytesConfig(load_in_4bit=True)
            elif self.args.quantization == "8bit":
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            else:
                quantization_config = None

            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_checkpoint,
                torch_dtype=torch.bfloat16 if self.bf16_support else torch.float16,
                device_map="auto",
                attn_implementation=self.args.attention,
                quantization_config=quantization_config,
            )

    def process_video(self, video_path, output_dir):
        """处理单个视频文件"""
        # 复制视频文件到输出目录
        video_name = os.path.basename(video_path)

        duration = get_video_duration(video_path)
        print("video :", video_path, "duration :", duration)
        if duration >= 10:
            print("skip")
            return
        
        output_video_path = os.path.join(output_dir, video_name)
        shutil.copy2(video_path, output_video_path)
        
        # 准备系统提示词 - 可从命令行参数获取
        system_content = self.args.system_content if self.args.system_content else '''You are QwenVL, you are a helpful assistant expert in turning images into words.\n 给你的视频中可能出现的主要人物为两个（可能出现一个或两个），当人物为一个戴眼罩的男孩时，男孩的名字是"夏尔",当人物是一个穿燕尾西服的成年男子时，男子的名字是"塞巴斯蒂安",在你的视频描述中要使用人物的名字并且简单描述人物的外貌及衣着。'''
        
        video_path = [{'type': 'video', 'video': video_path, 'fps': 1.0}]
        messages = [
            {
                "role": "system",
                "content": system_content
            },
            {
                "role": "user",
                "content": video_path + [
                    {"type": "text", "text": self.args.text},
                ],
            },
        ]

        with torch.no_grad():
            # 准备推理输入
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.device)
            
            # 推理：生成输出
            generated_ids = self.model.generate(
                **inputs, 
                max_new_tokens=self.args.max_new_tokens, 
                temperature=self.args.temperature
            )
            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            result = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

        # 保存结果到文本文件
        txt_filename = os.path.splitext(video_name)[0] + ".txt"
        txt_path = os.path.join(output_dir, txt_filename)
        result = result.split("addCriterion")[0].strip()
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(result)
        
        # 在命令行打印caption
        print(f"Video: {video_name}")
        print(f"Caption: {result}")
        print("-" * 50)
        
        return result

    def process_all_videos(self):
        """处理所有视频文件"""
        # 创建输出目录
        os.makedirs(self.args.output_dir, exist_ok=True)
        
        # 设置模型
        self.setup_model()
        
        # 处理单个视频文件
        if os.path.isfile(self.args.source_path):
            self.process_video(self.args.source_path, self.args.output_dir)
        
        # 处理目录中的所有视频文件
        elif os.path.isdir(self.args.source_path):
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
            for file in os.listdir(self.args.source_path):
                if any(file.lower().endswith(ext) for ext in video_extensions):
                    video_path = os.path.join(self.args.source_path, file)
                    self.process_video(video_path, self.args.output_dir)
        
        if not self.args.keep_model_loaded:
            self.cleanup()

    def cleanup(self):
        """清理模型和释放内存"""
        del self.processor
        del self.model
        self.processor = None
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

def main():
    """命令行主函数"""
    parser = argparse.ArgumentParser(description="Qwen2.5-VL视频描述生成工具")
    
    # 必需参数
    parser.add_argument("source_path", help="输入视频文件路径或包含视频的文件夹路径")
    parser.add_argument("output_dir", help="输出目录路径")
    
    # 模型参数
    parser.add_argument("--model", default="Qwen2.5-VL-7B-Instruct",
                       choices=["Qwen2.5-VL-3B-Instruct", "Qwen2.5-VL-7B-Instruct", 
                               "Qwen2.5-VL-32B-Instruct", "Qwen2.5-VL-72B-Instruct"],
                       help="选择使用的模型")
    parser.add_argument("--quantization", default="none", 
                       choices=["none", "4bit", "8bit"],
                       help="量化类型")
    parser.add_argument("--attention", default="eager",
                       choices=["eager", "sdpa", "flash_attention_2"],
                       help="注意力机制实现")
    
    # 生成参数
    parser.add_argument("--text", default="请描述这个视频的内容",
                       help="用于视频描述的提示文本")
    parser.add_argument("--system_content", default="",
                       help="系统角色内容，可覆盖默认的系统提示")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="生成温度（0-1）")
    parser.add_argument("--max_new_tokens", type=int, default=2048,
                       help="最大新生成token数量")
    
    # 图像处理参数
    parser.add_argument("--min_pixels", type=int, default=256 * 28 * 28,
                       help="最小像素数")
    parser.add_argument("--max_pixels", type=int, default=1280 * 28 * 28,
                       help="最大像素数")
    
    # 其他参数
    parser.add_argument("--keep_model_loaded", action="store_true",
                       help="处理完成后保持模型加载状态")
    parser.add_argument("--seed", type=int, default=-1,
                       help="随机种子（-1表示随机）")
    
    args = parser.parse_args()
    
    # 设置随机种子
    if args.seed != -1:
        torch.manual_seed(args.seed)
    
    # 创建处理器并处理视频
    processor = Qwen2_VQA(args)
    processor.process_all_videos()

if __name__ == "__main__":
    main()
