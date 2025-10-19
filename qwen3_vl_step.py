from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

# default: Load the model on the available device(s)
model = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen3-VL-4B-Instruct", dtype="auto", device_map="auto"
)

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# model = Qwen3VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen3-VL-4B-Instruct",
#     dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

processor = AutoProcessor.from_pretrained("Qwen3-VL-4B-Instruct")

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "F1.webp",
            },
            {"type": "text", "text": "使用中文描述这个图片，不要指出人物的名字或出处或其它猜测，只严格描述画面人物的特征。"},
        ],
    }
]

# Preparation for inference
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
)
inputs = inputs.to(model.device)

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=1024)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)

messages += [
        {
        "role": "assistant",
        "content": [
            {"type": "text", "text": output_text[0]},
        ],
    },
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "F2.jpg",
            },
            {"type": "text", "text": "使用中文描述这个图片是如何根据上一个图片进行编辑得到的,不要指出人物的名字或出处或其它猜测，只严格描述进行了哪些编辑和改变。"},
        ],
    }
]

# Preparation for inference
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
)
inputs = inputs.to(model.device)

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=1024)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)

#############

import json
import os
import tempfile
from tqdm import tqdm
from datasets import load_dataset
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from PIL import Image

def process_teyvat_goods_dataset(output_dir="output"):
    """
    处理Teyvat_Goods_Source_Images_Pair_RMBG数据集
    自动处理PIL Image对象的保存和清理
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建临时目录用于存储临时图像文件
    temp_dir = tempfile.mkdtemp()
    print(f"创建临时目录: {temp_dir}")
    
    # 加载模型和处理器
    print("正在加载Qwen3-VL模型...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        "Qwen3-VL-4B-Instruct", 
        dtype="auto", 
        device_map="auto"
    )
    
    processor = AutoProcessor.from_pretrained("Qwen3-VL-4B-Instruct")
    
    # 加载数据集
    print("正在加载数据集...")
    dataset = load_dataset("svjack/Teyvat_Goods_Source_Images_Pair_RMBG")
    
    # 确定要处理的数据集分割
    split = "train" if "train" in dataset else list(dataset.keys())[0]
    data_split = dataset[split]
    
    print(f"开始处理数据集，共{len(data_split)}个样本...")
    
    results = []
    
    # 使用tqdm创建进度条
    for idx, item in enumerate(tqdm(data_split, desc="处理图像对")):
        try:
            # 获取PIL Image对象
            source_img = item["source_img_rmbg"]
            edited_img = item["image_rmbg"]
            
            # 验证图像对象类型
            if not isinstance(source_img, Image.Image) or not isinstance(edited_img, Image.Image):
                print(f"警告: 第{idx}个样本的图像不是PIL Image对象，跳过处理")
                continue
            
            # 保存PIL Image到临时文件[1,3](@ref)
            source_temp_path = os.path.join(temp_dir, f"source_temp_{idx}.png")
            edited_temp_path = os.path.join(temp_dir, f"edited_temp_{idx}.png")
            
            # 使用PNG格式保存以保持质量[1,4](@ref)
            source_img.save(source_temp_path, format="PNG")
            edited_img.save(edited_temp_path, format="PNG")
            
            # 第一轮对话：描述源图像
            messages_round1 = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": source_temp_path},
                        {"type": "text", "text": "使用中文描述这个图片，不要指出人物的名字或出处或其它猜测，只严格描述画面人物的特征。"}
                    ]
                }
            ]
            
            # 准备第一轮推理输入
            inputs_round1 = processor.apply_chat_template(
                messages_round1,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            )
            inputs_round1 = inputs_round1.to(model.device)
            
            # 第一轮推理生成
            generated_ids_round1 = model.generate(**inputs_round1, max_new_tokens=1024)
            generated_ids_trimmed_round1 = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs_round1.input_ids, generated_ids_round1)
            ]
            output_text_round1 = processor.batch_decode(
                generated_ids_trimmed_round1, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            
            description_round1 = output_text_round1[0]
            
            # 构建第二轮对话历史
            messages_round2 = messages_round1 + [
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": description_round1}]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": edited_temp_path},
                        {"type": "text", "text": "使用中文描述这个图片是如何根据上一个图片进行编辑得到的,不要指出人物的名字或出处或其它猜测，只严格描述进行了哪些编辑和改变。以'进行下面的修改：'开头"}
                    ]
                }
            ]
            
            # 准备第二轮推理输入
            inputs_round2 = processor.apply_chat_template(
                messages_round2,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            )
            inputs_round2 = inputs_round2.to(model.device)
            
            # 第二轮推理生成
            generated_ids_round2 = model.generate(**inputs_round2, max_new_tokens=1024)
            generated_ids_trimmed_round2 = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs_round2.input_ids, generated_ids_round2)
            ]
            output_text_round2 = processor.batch_decode(
                generated_ids_trimmed_round2, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            
            description_round2 = output_text_round2[0]
            
            # 删除临时文件以释放空间
            os.remove(source_temp_path)
            os.remove(edited_temp_path)
            
            # 构建结果记录
            result = {
                "source_image": f"source_temp_{idx}.png",  # 只保存文件名，不保存路径
                "edited_image": f"edited_temp_{idx}.png",
                "first_description": description_round1,
                "edit_description": description_round2,
                "sample_id": f"{idx:06d}"
            }
            
            results.append(result)
            
            # 每处理10个样本保存一次中间结果
            if (idx + 1) % 10 == 0:
                save_intermediate_results(results, output_dir, idx)
                
        except Exception as e:
            print(f"处理第{idx}个样本时出错: {str(e)}")
            # 尝试清理可能创建的临时文件
            try:
                if 'source_temp_path' in locals() and os.path.exists(source_temp_path):
                    os.remove(source_temp_path)
                if 'edited_temp_path' in locals() and os.path.exists(edited_temp_path):
                    os.remove(edited_temp_path)
            except:
                pass
            continue
    
    # 保存最终结果
    save_final_results(results, output_dir)
    
    # 清理临时目录（如果为空）
    try:
        os.rmdir(temp_dir)
        print(f"已清理临时目录: {temp_dir}")
    except:
        print(f"临时目录 {temp_dir} 不为空，保留目录")
    
    return results

def save_intermediate_results(results, output_dir, current_idx):
    """保存中间结果"""
    intermediate_file = os.path.join(output_dir, f"intermediate_results_{current_idx:06d}.json")
    with open(intermediate_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

def save_final_results(results, output_dir):
    """保存最终结果，每个样本一个JSON文件"""
    print("正在保存最终结果...")
    
    for result in tqdm(results, desc="保存JSON文件"):
        filename = f"{result['sample_id']}.json"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    
    # 同时保存一个包含所有结果的汇总文件
    summary_file = os.path.join(output_dir, "all_results.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"处理完成！共处理{len(results)}个样本。")
    print(f"结果保存在: {output_dir}")


if __name__ == "__main__":
    # 基本用法
    process_teyvat_goods_dataset("teyvat_results")
    
    # 或者使用优化版本处理前100个样本
    # process_teyvat_goods_optimized("teyvat_results_optimized", max_samples=100)


# 使用中文描述这个图片，不要指出人物的名字或出处或其它猜测，只严格描述画面中不同部分的物体与整体风格特征。

# 使用中文描述这个图片是如何根据上一个图片用相同的画风进行编辑得到的,不要指出人物的名字或出处或其它猜测，只严格描述进行了哪些在同一画风下的编辑和改变，注意要将这个照片中与上一张照片中画风相对应的部分进行提及，对物体取代、色彩变化、整体画风遵循进行对应描述。

vllm serve Qwen3-VL-4B-Instruct \
  --tensor-parallel-size 1 \
  --limit-mm-per-prompt.video 0 \
  --async-scheduling \
  --gpu-memory-utilization 0.95 \
  --max-model-len 10240

import time
from openai import OpenAI

client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1",
    timeout=3600
)

import base64

def image_to_data_url(image_path):
    with open(image_path, "rb") as image_file:
        base64_data = base64.b64encode(image_file.read()).decode('utf-8')
        return f"data:image/jpeg;base64,{base64_data}"

# 使用示例
image_data_url = image_to_data_url("IMG_0.jpeg")

messages = [
    {
        "role": "user", 
        "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": image_data_url
                }
            },
            {
                "type": "text",
                "text": "使用中文描述这个图片，不要指出人物的名字或出处或其它猜测，只严格描述画面中不同部分的物体与整体风格特征。"
            }
        ]
    }
]

start = time.time()
response = client.chat.completions.create(
    model="Qwen3-VL-4B-Instruct",
    messages=messages,
    max_tokens=1024
)
print(f"Response costs: {time.time() - start:.2f}s")
print(f"Generated text: {response.choices[0].message.content}")

# 使用示例
image_data_url = image_to_data_url("IMG_1.jpeg")

messages += [
        {
        "role": "assistant",
        "content": [
            {"type": "text", "text": response.choices[0].message.content},
        ],
    },
    {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {"url" : image_data_url},
            },
            {"type": "text", "text": "使用中文描述这个图片是如何根据上一个图片用相同的画风进行编辑得到的,不要指出人物的名字或出处或其它猜测，只严格描述进行了哪些在同一画风下的编辑和改变，注意要将这个照片中与上一张照片中画风相对应的部分进行提及，对物体取代、色彩变化、整体画风遵循进行对应描述。以'进行下面的修改：'开头"},
        ],
    }
]

start = time.time()
response = client.chat.completions.create(
    model="Qwen3-VL-4B-Instruct",
    messages=messages,
    max_tokens=1024
)
print(f"Response costs: {time.time() - start:.2f}s")
print(f"Generated text: {response.choices[0].message.content}")


