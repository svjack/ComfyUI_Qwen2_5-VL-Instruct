import os
import shutil

def analyze_and_copy_files(*folder_paths, target_folders):
    """
    分析多个文件夹中txt文件的关键词出现情况，并复制满足条件的文件对到指定文件夹
    
    Parameters:
    *folder_paths: 任意多个源文件夹路径
    target_folders: 目标文件夹路径的字典，包含三个键：
                    - 'only_ciel': 只出现"夏尔"的目标文件夹
                    - 'only_sebastian': 只出现"塞巴斯蒂安"的目标文件夹  
                    - 'both': 同时出现两个关键词的目标文件夹
    """
    
    # 确保目标文件夹存在
    for folder in target_folders.values():
        os.makedirs(folder, exist_ok=True)
    
    # 遍历所有源文件夹
    for folder_path in folder_paths:
        if not os.path.exists(folder_path):
            print(f"警告: 文件夹 '{folder_path}' 不存在，跳过")
            continue
            
        if not os.path.isdir(folder_path):
            print(f"警告: '{folder_path}' 不是文件夹，跳过")
            continue
            
        # 获取所有txt文件
        txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
        
        for txt_filename in txt_files:
            # 获取对应的mp4文件名（假设文件名相同，只是扩展名不同）
            base_name = os.path.splitext(txt_filename)[0]
            mp4_filename = base_name + '.mp4'
            
            txt_path = os.path.join(folder_path, txt_filename)
            mp4_path = os.path.join(folder_path, mp4_filename)
            
            # 检查mp4文件是否存在
            if not os.path.exists(mp4_path):
                print(f"警告: 找不到对应的mp4文件 '{mp4_filename}'，跳过")
                continue
            
            try:
                # 读取txt文件内容
                with open(txt_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    has_ciel = '夏尔' in content
                    has_sebastian = '塞巴斯蒂安' in content
                    
                    # 确定目标文件夹
                    if has_ciel and not has_sebastian:
                        target_folder = target_folders['only_ciel']
                    elif has_sebastian and not has_ciel:
                        target_folder = target_folders['only_sebastian']
                    elif has_ciel and has_sebastian:
                        target_folder = target_folders['both']
                    else:
                        # 两个关键词都没有，跳过
                        continue
                    
                    # 复制文件对到目标文件夹
                    shutil.copy2(txt_path, os.path.join(target_folder, txt_filename))
                    shutil.copy2(mp4_path, os.path.join(target_folder, mp4_filename))
                    print(f"已复制: {txt_filename} 和 {mp4_filename} 到 {target_folder}")
                    
            except UnicodeDecodeError:
                # 如果UTF-8编码失败，尝试其他编码
                try:
                    with open(txt_path, 'r', encoding='gbk') as file:
                        content = file.read()
                        has_ciel = '夏尔' in content
                        has_sebastian = '塞巴斯蒂安' in content
                        
                        if has_ciel and not has_sebastian:
                            target_folder = target_folders['only_ciel']
                        elif has_sebastian and not has_ciel:
                            target_folder = target_folders['only_sebastian']
                        elif has_ciel and has_sebastian:
                            target_folder = target_folders['both']
                        else:
                            continue
                        
                        shutil.copy2(txt_path, os.path.join(target_folder, txt_filename))
                        shutil.copy2(mp4_path, os.path.join(target_folder, mp4_filename))
                        print(f"已复制: {txt_filename} 和 {mp4_filename} 到 {target_folder}")
                        
                except Exception as e:
                    print(f"无法处理文件 {txt_path}: {e}")
            except Exception as e:
                print(f"处理文件 {txt_path} 时出错: {e}")

def main():
    # 源文件夹路径
    source_folders = [
        "Prince_Ciel_Phantomhive_Videos_qwen_vl_captioned",
        "Sebastian_Michaelis_Videos_qwen_vl_captioned",
    ]
    
    # 目标文件夹路径
    target_folders = {
        'only_ciel': "only_ciel_files",
        'only_sebastian': "only_sebastian_files", 
        'both': "both_files"
    }
    
    # 执行分析和复制
    analyze_and_copy_files(*source_folders, target_folders=target_folders)
    
    # 统计结果
    print("\n复制完成！各文件夹文件数量统计:")
    for category, folder in target_folders.items():
        txt_count = len([f for f in os.listdir(folder) if f.endswith('.txt')])
        mp4_count = len([f for f in os.listdir(folder) if f.endswith('.mp4')])
        print(f"{folder}: {txt_count}个txt文件, {mp4_count}个mp4文件")

if __name__ == "__main__":
    main()
