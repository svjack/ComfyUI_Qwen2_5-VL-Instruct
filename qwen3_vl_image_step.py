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
  --max-model-len 20480

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

vllm serve Qwen3-VL-8B-Instruct \
  --tensor-parallel-size 1 \
  --limit-mm-per-prompt.video 0 \
  --async-scheduling \
  --gpu-memory-utilization 0.95 \
  --max-model-len 10240

import os
import base64
import time
from openai import OpenAI
from PIL import Image
import ast
from datasets import load_dataset
from tqdm import tqdm
import json

# 初始化OpenAI客户端
client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1",
    timeout=3600
)

def image_to_data_url(image_path):
    """将图像文件转换为data URL格式"""
    with open(image_path, "rb") as image_file:
        base64_data = base64.b64encode(image_file.read()).decode('utf-8')
        return f"data:image/jpeg;base64,{base64_data}"

def save_pil_image(image, output_dir, index):
    """保存PIL图像到文件[6,7](@ref)"""
    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成文件名（带前导零的6位数字）
    filename = f"{index:06d}.jpg"
    filepath = os.path.join(output_dir, filename)
    
    # 保存图像[6](@ref)
    image.save(filepath, "JPEG", quality=95)
    return filepath

def process_dataset_sample(sample, index, output_dir):
    """处理单个数据集样本"""
    try:
        # 获取图像和names列
        pil_image = sample['image']
        names_str = sample['names']
        
        # 将names字符串转换为列表[1](@ref)
        if isinstance(names_str, str):
            names_list = ast.literal_eval(names_str)
        else:
            names_list = names_str
        
        # 保存图像到临时文件
        image_path = save_pil_image(pil_image, os.path.join(output_dir, "images"), index)
        
        # 构建消息
        messages = []
        for name in names_list:
            # 注意：这里需要根据你实际的人物图片路径调整
            # 假设人物图片在 Genshin_Impact_Portrait_chr 目录下
            character_image_path = f"Genshin_Impact_Portrait_chr/{name}.png"
            if os.path.exists(character_image_path):
                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": image_to_data_url(character_image_path)},
                        },
                        {"type": "text", "text": f"这个是人物'{name}'的图片，请记住它的特征。"},
                    ],
                })
        
        # 添加主要查询消息
        messages += [
            {
                "role": "user", 
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": image_to_data_url(image_path)}
                    },
                    {
                        "type": "text",
                        "text": f"现在根据你知道的各个人物的名字，给出漫画格的位置叙述，将上面对漫画的描写中人物的名字加到叙述中，未提到的人物则保持原来的叙述，得到漫画描述，以'下面是对漫画每格的描述:\\n...'为格式",
                    }
                ]
            }
        ]
        
        # 调用API
        start_time = time.time()
        response = client.chat.completions.create(
            model="Qwen3-VL-8B-Instruct",
            messages=messages,
            max_tokens=1024
        )
        
        # 保存结果到文本文件
        result_text = response.choices[0].message.content
        txt_filename = f"{index:06d}.txt"
        txt_filepath = os.path.join(output_dir, txt_filename)
        
        with open(txt_filepath, "w", encoding="utf-8") as f:
            f.write(result_text)
        
        # 清理临时图像文件
        os.remove(image_path)
        
        return {
            "index": index,
            "names": names_list,
            "response_time": time.time() - start_time,
            "text_length": len(result_text)
        }
        
    except Exception as e:
        print(f"处理样本 {index} 时出错: {e}")
        return None

def main():
    """主函数"""
    # 配置参数
    dataset_name = "svjack/Genshin-Impact-Manga-Rm-Text-3-Named"
    output_dir = "output_results"
    max_samples = None  # 设置为None处理所有样本，或设置具体数字进行测试
    
    print("正在加载数据集...")
    
    try:
        # 加载数据集[3](@ref)
        dataset = load_dataset(dataset_name)
        
        # 假设我们使用训练集
        if 'train' in dataset:
            data = dataset['train']
        else:
            data = dataset
            
        print(f"数据集加载成功，包含 {len(data)} 个样本")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
        
        # 处理样本
        successful_count = 0
        failed_count = 0
        results = []
        
        # 确定要处理的样本范围
        if max_samples is not None:
            indices = range(min(max_samples, len(data)))
        else:
            indices = range(len(data))
        
        print("开始处理样本...")
        for i in tqdm(indices, desc="处理进度"):
            sample = data[i]
            result = process_dataset_sample(sample, i, output_dir)
            
            if result is not None:
                successful_count += 1
                results.append(result)
            else:
                failed_count += 1
            
            # 添加延迟避免API过载
            time.sleep(1)
        
        # 保存处理统计信息
        stats = {
            "total_processed": len(indices),
            "successful": successful_count,
            "failed": failed_count,
            "success_rate": successful_count / len(indices) if len(indices) > 0 else 0,
            "results": results
        }
        
        with open(os.path.join(output_dir, "processing_stats.json"), "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        print(f"\n处理完成！")
        print(f"总处理样本: {len(indices)}")
        print(f"成功: {successful_count}")
        print(f"失败: {failed_count}")
        print(f"成功率: {stats['success_rate']:.2%}")
        
        if results:
            avg_time = sum(r['response_time'] for r in results) / len(results)
            print(f"平均响应时间: {avg_time:.2f}秒")
        
    except Exception as e:
        print(f"处理数据集时发生错误: {e}")

if __name__ == "__main__":
    main()

