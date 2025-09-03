'''
python txt_fix.py --source Prince_Ciel_Phantomhive_Videos_Qwen_VL_Captioned \
--output Prince_Ciel_Phantomhive_Videos_qwen_vl_captioned \
--search "塞巴斯蒂安" --replace "夏尔"

python txt_fix.py --source Sebastian_Michaelis_Videos_Qwen_VL_Captioned \
--output Sebastian_Michaelis_Videos_qwen_vl_captioned \
--search "夏尔" --replace "塞巴斯蒂安"
'''

import os
import shutil
import argparse

def process_files(source_dir, output_dir, search_name, replace_name):
    """
    处理文件夹中的文件：拷贝所有文件，并根据条件替换txt文件内容
    
    Args:
        source_dir: 源目录路径
        output_dir: 输出目录路径
        search_name: 要搜索的名称
        replace_name: 要替换的名称
    """
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 遍历源目录中的所有文件
    for filename in os.listdir(source_dir):
        source_path = os.path.join(source_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        # 处理mp4文件：直接拷贝
        if filename.endswith('.mp4'):
            shutil.copy2(source_path, output_path)
            print(f"已拷贝: {filename}")
        
        # 处理txt文件：根据条件替换内容
        elif filename.endswith('.txt'):
            try:
                # 读取txt文件内容
                with open(source_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                
                # 检查条件：包含search_name但不包含replace_name
                if search_name in content and replace_name not in content:
                    # 替换内容中的search_name为replace_name
                    new_content = content.replace(search_name, replace_name)
                    
                    # 写入到输出目录
                    with open(output_path, 'w', encoding='utf-8') as file:
                        file.write(new_content)
                    
                    print(f"已替换内容: {filename} (将'{search_name}'替换为'{replace_name}')")
                else:
                    # 不满足条件，直接拷贝
                    shutil.copy2(source_path, output_path)
                    print(f"已拷贝: {filename} (无需替换)")
                        
            except Exception as e:
                print(f"处理文件 {filename} 时出错: {e}")
                # 出错时也尝试拷贝原文件
                try:
                    shutil.copy2(source_path, output_path)
                    print(f"出错后已拷贝原文件: {filename}")
                except:
                    print(f"无法拷贝文件: {filename}")

def main():
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='处理视频文件：根据条件替换txt文件内容')
    parser.add_argument('--source', '-s', default='Prince_Ciel_Phantomhive_Videos_Qwen_VL_Captioned',
                       help='源目录路径，默认为 Prince_Ciel_Phantomhive_Videos_Qwen_VL_Captioned')
    parser.add_argument('--output', '-o', default='Prince_Ciel_Phantomhive_Videos_qwen_vl_captioned',
                       help='输出目录路径，默认为 Prince_Ciel_Phantomhive_Videos_qwen_vl_captioned')
    parser.add_argument('--search', '-sn', default='塞巴斯蒂安',
                       help='要在txt中搜索的名称，默认为 塞巴斯蒂安')
    parser.add_argument('--replace', '-rn', default='夏尔',
                       help='要替换的名称，默认为 夏尔')
    
    args = parser.parse_args()
    
    print(f"开始处理文件...")
    print(f"源目录: {args.source}")
    print(f"输出目录: {args.output}")
    print(f"搜索名称: {args.search}")
    print(f"替换名称: {args.replace}")
    print("-" * 50)
    
    # 处理文件
    process_files(args.source, args.output, args.search, args.replace)
    print("-" * 50)
    print("处理完成！")

if __name__ == "__main__":
    main()
