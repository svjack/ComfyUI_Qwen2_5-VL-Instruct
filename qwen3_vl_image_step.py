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

vllm serve Qwen3-VL-2B-Instruct \
  --tensor-parallel-size 1 \
  --limit-mm-per-prompt.video 0 \
  --async-scheduling \
  --gpu-memory-utilization 0.95 \
  --max-model-len 20480

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

prompt = '''
        使用中文描述动漫图片中的内容，
        一、 角色特征方面
        外貌与身份锚点：
        角色应具有怎样的标志性外貌以增强辨识度？（例如：独特的发型/发色、瞳色、面部特征如泪痣或虎牙）
        角色的体型与头身比是多少？这如何体现其年龄或种族特征？（例如：4-6头身体现Q版可爱，7-8头身更接近少年少女）
        角色的服装与配饰有哪些关键细节？这些设计如何反映其身份、职业或世界观？（例如：汉元素服饰、科幻装甲、学院制服、具有象征意义的配饰如项链或武器）
        神态与动态表达：
        角色正处于怎样的特定动作或姿态中？这个动作是动态的还是静态的？（例如：坐在树梢、奔跑回眸、持剑凝视）
        角色面部需要展现怎样的精确表情？（例如：微笑的幅度、眼神的焦点、眼睑的开合程度）
        角色如何与环境或道具互动？（例如：轻抚花瓣、手持发光书卷、衣物随风飘动）
        二、 场景设定方面
        环境与世界观构建：
        主场景发生在何处？这是一个怎样的非现实性或艺术化的环境？（例如：浮空岛水墨庭院、赛博朋克都市、奇幻森林）
        背景中包含哪些关键的环境元素与细节以丰富画面层次？（例如：远处的山水、飘落的花瓣、流动的符文、未来主义的全息广告）
        透视与构图有何要求？（例如：视角是平视、仰视还是俯视？有无需要强调的视觉引导焦点？）
        氛围与视觉美学：
        画面的光影效果应如何设计？（例如：光源的类型、方向、角度、强度，以及产生的效果如丁达尔效应、高光与阴影的对比）
        整体的色彩基调与配色方案是什么？主色、辅色和点缀色分别是什么？（例如：青蓝与月白的柔和渐变，搭配少量暖色作为点缀）
        需要营造怎样的整体氛围与情绪？（例如：宁静神秘、热血激昂、温馨治愈）
        三、 风格与质量方面
        图像应遵循何种统一的艺术风格？（例如：日系赛璐璐风格、吉卜力风格、新海诚式写实光影、赛博朋克风格）
        对画面质量有何具体要求？（例如：8K分辨率、精致的细节渲染、无视觉瑕疵）
        有哪些需要明确避免的元素？（例如：避免写实感、比例失调、模糊、水印、文字或特定的不想要的内容）
        
        将你分上面不同维度的分析结果合成一段关于这个画面的中文描述，不进行任何讨论，直接给出中文结果。
        '''


### three bind position labeled

import os
import base64
import time
from openai import OpenAI
from PIL import Image
import ast
from tqdm import tqdm
import json
import uuid

# 初始化OpenAI客户端
client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1",
    timeout=3600
)

def image_to_data_url(image_path):
    """将图像文件转换为data URL格式[1,6](@ref)"""
    with open(image_path, "rb") as image_file:
        base64_data = base64.b64encode(image_file.read()).decode('utf-8')
        return f"data:image/jpeg;base64,{base64_data}"

def save_pil_image(image, output_path):
    """保存PIL图像到文件"""
    # 创建输出目录（如果不存在）
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 保存图像
    image.save(output_path, "JPEG", quality=95)
    return output_path

def initialize_stats_file(output_dir):
    """初始化统计文件"""
    stats_file = os.path.join(output_dir, "processing_stats.json")
    
    initial_stats = {
        "total_processed": 0,
        "successful": 0,
        "failed": 0,
        "success_rate": 0,
        "results": [],
        "start_time": time.time(),
        "last_update": time.time()
    }
    
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(initial_stats, f, ensure_ascii=False, indent=2)
    
    return stats_file

def update_stats_incrementally(stats_file, result, status):
    """增量更新统计信息"""
    try:
        # 读取当前的统计信息
        if os.path.exists(stats_file):
            with open(stats_file, "r", encoding="utf-8") as f:
                stats = json.load(f)
        else:
            stats = {
                "total_processed": 0,
                "successful": 0,
                "failed": 0,
                "success_rate": 0,
                "results": [],
                "start_time": time.time()
            }
        
        # 更新统计信息
        stats["total_processed"] += 1
        stats["last_update"] = time.time()
        
        if status == "success":
            stats["successful"] += 1
            if result is not None:
                stats["results"].append({
                    "original_image": result["original_image"],
                    "prompt_file": result["prompt_file"],
                    "response_time": result["response_time"],
                    "base_filename": result["base_filename"],
                    "processing_time": time.strftime("%Y-%m-%d %H:%M:%S")
                })
        else:
            stats["failed"] += 1
        
        # 计算成功率
        if stats["total_processed"] > 0:
            stats["success_rate"] = stats["successful"] / stats["total_processed"]
        
        # 计算平均处理时间
        if stats["successful"] > 0:
            total_time = sum(r["response_time"] for r in stats["results"])
            stats["average_response_time"] = total_time / stats["successful"]
        
        # 增量写入更新后的统计信息
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        return True
        
    except Exception as e:
        print(f"更新统计信息时出错: {e}")
        return False

def process_single_image(image_path, index, output_dir, stats_file):
    """处理单张图片：先问位置，再描述场景"""
    try:
        # 生成唯一的基础文件名
        unique_id = uuid.uuid4().hex[:8]
        base_filename = f"{index:06d}_{unique_id}"
        
        # 保存原始图片到输出目录
        final_output_dir = os.path.join(output_dir, "final_pairs")
        original_image_path = os.path.join(final_output_dir, f"{base_filename}.jpg")
        
        # 打开并保存原始图片
        img = Image.open(image_path)
        save_pil_image(img, original_image_path)
        
        print(f"正在处理图片: {os.path.basename(image_path)}")
        
        # 第一步：询问两个人物在第三张图片中的位置
        position_messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": image_to_data_url(image_path)},
                    },
                    {
                        "type": "text", 
                        "text": "这张图片纵向分为3部分，最左边的是人物1，中间的是人物2，最右边的是人物1与人物2的互动场景。请分析最右边的互动场景，告诉我人物1和人物2分别位于场景的左侧还是右侧？请用简洁的语言回答。"
                    },
                ],
            },
        ]
        
        # 调用API获取位置信息
        start_time = time.time()
        position_response = client.chat.completions.create(
            model="Qwen3-VL-8B-Instruct",
            messages=position_messages,
            max_tokens=200
        )
        
        position_text = position_response.choices[0].message.content
        position_time = time.time() - start_time
        
        print(f"位置分析结果: {position_text}")
        
        # 第二步：基于位置信息描述第三张图片
        description_messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": image_to_data_url(image_path)},
                    },
                    {
                        "type": "text", 
                        "text": "这张图片纵向分为3部分，最左边的是人物1，中间的是人物2，最右边的是人物1与人物2的互动场景。请分析最右边的互动场景，告诉我人物1和人物2分别位于场景的左侧还是右侧？请用简洁的语言回答。"
                    },
                ],
            },
            {
                "role": "assistant",
                "content": position_text
            },
            {
                "role": "user", 
                "content": [
                    {
                        "type": "text",
                        "text": """基于上面两个角色的位置特征和角色特征，请为最右边的第三张图片生成一个详细的提示词描述。
                        
要求：
1. 在你给出的描述中要以“图片1角色”和“图片2角色”称呼第一个角色和第二个角色，而不要提到任何人物名字。
2. 结合第一个角色和第二个角色的特征，要在叙述的开头描述“图片1角色”和“图片2角色”位于最右边的第三张图片的左边还是右边。
3. 描述这两个角色在第三张图片中可能的互动场景。
4. 包括人物表情、衣着、环境背景、角色姿态、情感表达等细节
5. 保持角色特征的一致性
6. 生成富有创意且合理的场景描述
7. 注意对人物的称呼一概为“图片1角色”和“图片2角色”称呼人物

请以以下格式回复：
角色互动场景描述：
[你的详细描述]"""
                    }
                ]
            }
        ]
        
        # 调用API获取场景描述
        description_start_time = time.time()
        description_response = client.chat.completions.create(
            model="Qwen3-VL-8B-Instruct",
            messages=description_messages,
            max_tokens=500
        )
        
        description_text = description_response.choices[0].message.content
        description_time = time.time() - description_start_time
        
        # 合并两个API调用的结果
        final_prompt_text = f"""位置分析：
{position_text}

场景描述：
{description_text}"""
        
        # 保存提示词文本文件
        txt_filename = f"{base_filename}.txt"
        txt_filepath = os.path.join(final_output_dir, txt_filename)
        
        with open(txt_filepath, "w", encoding="utf-8") as f:
            f.write(final_prompt_text)
        
        # 准备结果数据
        result = {
            "index": index,
            "base_filename": base_filename,
            "original_image": original_image_path,
            "prompt_file": txt_filepath,
            "response_time": position_time + description_time,
            "position_response": position_text,
            "description_response": description_text
        }
        
        # 增量更新统计信息
        update_stats_incrementally(stats_file, result, "success")
        
        return result
        
    except Exception as e:
        print(f"处理图片 {image_path} 时出错: {e}")
        # 更新失败的统计信息
        update_stats_incrementally(stats_file, None, "failed")
        return None

def main():
    """主函数"""
    # 配置参数
    input_dir = "output_all_combinations/"  # 输入图片目录
    output_dir = "output_image_description_pairs"  # 输出目录
    
    print("正在准备处理图片...")
    
    try:
        # 获取输入目录中的所有图片文件
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        image_files = []
        
        if os.path.exists(input_dir):
            for file in os.listdir(input_dir):
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    image_files.append(os.path.join(input_dir, file))
        
        if not image_files:
            print(f"在目录 {input_dir} 中未找到图片文件")
            return
        
        print(f"找到 {len(image_files)} 个图片文件")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "final_pairs"), exist_ok=True)
        
        # 初始化统计文件
        stats_file = initialize_stats_file(output_dir)
        print(f"统计文件已初始化: {stats_file}")
        
        # 处理图片
        successful_count = 0
        failed_count = 0
        results = []
        
        print("开始处理图片...")
        for i, image_path in enumerate(tqdm(image_files, desc="处理进度")):
            result = process_single_image(image_path, i, output_dir, stats_file)
            
            if result is not None:
                successful_count += 1
                results.append(result)
                print(f"✓ 成功处理: {os.path.basename(image_path)}")
            else:
                failed_count += 1
                print(f"✗ 处理失败: {os.path.basename(image_path)}")
            
            # 添加延迟避免API过载
            time.sleep(2)  # 增加延迟时间，因为现在有两个API调用
        
        # 最终统计信息
        final_stats = {
            "total_processed": len(image_files),
            "successful": successful_count,
            "failed": failed_count,
            "success_rate": successful_count / len(image_files) if len(image_files) > 0 else 0,
            "completion_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_duration": time.time() - time.mktime(time.strptime(
                open(stats_file, "r", encoding="utf-8").read().split('"start_time": ')[1].split(',')[0], 
                "%Y-%m-%d %H:%M:%S"
            )) if os.path.exists(stats_file) else 0
        }
        
        # 保存最终统计信息
        with open(os.path.join(output_dir, "final_stats.json"), "w", encoding="utf-8") as f:
            json.dump(final_stats, f, ensure_ascii=False, indent=2)
        
        print(f"\n处理完成！")
        print(f"总处理图片: {len(image_files)}")
        print(f"成功: {successful_count}")
        print(f"失败: {failed_count}")
        print(f"成功率: {final_stats['success_rate']:.2%}")
        
        if results:
            avg_time = sum(r['response_time'] for r in results) / len(results)
            print(f"平均响应时间: {avg_time:.2f}秒")
            print(f"输出目录: {os.path.join(output_dir, 'final_pairs')}")
            print(f"实时统计文件: {stats_file}")
            print(f"最终统计文件: {os.path.join(output_dir, 'final_stats.json')}")
        
    except Exception as e:
        print(f"处理图片时发生错误: {e}")

if __name__ == "__main__":
    main()
