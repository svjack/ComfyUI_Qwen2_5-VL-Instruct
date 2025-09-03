```bash
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

python split_to_3.py

mv only_ciel_files Prince_Ciel_Phantomhive_only_Videos_qwen_vl_captioned
mv only_sebastian_files Sebastian_Michaelis_only_Videos_qwen_vl_captioned
mv both_files Prince_Ciel_Phantomhive_Sebastian_Michaelis_both_Videos_qwen_vl_captioned
```

<br/>

```bash
git clone https://huggingface.co/datasets/svjack/Prince_Ciel_Phantomhive_Sebastian_Michaelis_both_Videos_qwen_vl_captioned

git clone https://huggingface.co/datasets/svjack/Sebastian_Michaelis_only_Videos_qwen_vl_captioned

git clone https://huggingface.co/datasets/svjack/Prince_Ciel_Phantomhive_only_Videos_qwen_vl_captioned



mkdir -p Ciel_Phantomhive_Sebastian_Michaelis

# 进入源目录
cd Prince_Ciel_Phantomhive_Sebastian_Michaelis_both_Videos_qwen_vl_captioned

# 找出所有.mp4文件，提取其前缀，并限制150个
find . -maxdepth 1 -name "*.mp4" | head -n 150 | while read mp4_file; do
    base_name=$(basename "$mp4_file" .mp4)
    # 查找对应的.txt文件
    txt_file="${base_name}.txt"
    if [ -f "$txt_file" ]; then
        cp "$mp4_file" "$txt_file" ../Ciel_Phantomhive_Sebastian_Michaelis/
    fi
done

cd ..


mkdir -p Ciel_Phantomhive

# 进入源目录
cd Prince_Ciel_Phantomhive_only_Videos_qwen_vl_captioned

# 找出所有.mp4文件，提取其前缀，并限制150个
find . -maxdepth 1 -name "*.mp4" | head -n 150 | while read mp4_file; do
    base_name=$(basename "$mp4_file" .mp4)
    # 查找对应的.txt文件
    txt_file="${base_name}.txt"
    if [ -f "$txt_file" ]; then
        cp "$mp4_file" "$txt_file" ../Ciel_Phantomhive/
    fi
done

cd ..


mkdir -p Sebastian_Michaelis

# 进入源目录
cd Sebastian_Michaelis_only_Videos_qwen_vl_captioned

# 找出所有.mp4文件，提取其前缀，并限制150个
find . -maxdepth 1 -name "*.mp4" | head -n 150 | while read mp4_file; do
    base_name=$(basename "$mp4_file" .mp4)
    # 查找对应的.txt文件
    txt_file="${base_name}.txt"
    if [ -f "$txt_file" ]; then
        cp "$mp4_file" "$txt_file" ../Sebastian_Michaelis/
    fi
done

cd ..

```

# ComfyUI_Qwen2_5-VL-Instruct

This is an implementation of [Qwen2.5-VL-Instruct](https://github.com/QwenLM/Qwen2.5-VL) by [ComfyUI](https://github.com/comfyanonymous/ComfyUI), which includes, but is not limited to, support for text-based queries, video queries, single-image queries, and multi-image queries to generate captions or responses.

---

## Basic Workflow

- **Text-based Query**: Users can submit textual queries to request information or generate descriptions. For instance, a user might input a description like "What is the meaning of life?"

![Chat_with_text_workflow preview](examples/Chat_with_text_workflow.png)

- **Video Query**: When a user uploads a video, the system can analyze the content and generate a detailed caption for each frame or a summary of the entire video. For example, "Generate a caption for the given video."

![Chat_with_video_workflow preview](examples/Chat_with_video_workflow.png)

- **Single-Image Query**: This workflow supports generating a caption for an individual image. A user could upload a photo and ask, "What does this image show?" resulting in a caption such as "A majestic lion pride relaxing on the savannah."

![Chat_with_single_image_workflow preview](examples/Chat_with_single_image_workflow.png)

- **Multi-Image Query**: For multiple images, the system can provide a collective description or a narrative that ties the images together. For example, "Create a story from the following series of images: one of a couple at a beach, another at a wedding ceremony, and the last one at a baby's christening."

![Chat_with_multiple_images_workflow preview](examples/Chat_with_multiple_images_workflow.png)

> [!IMPORTANT]
> Important Notes for Using the Workflow
> - Please ensure that you have the "Display Text node" available in your ComfyUI setup. If you encounter any issues with this node being missing, you can find it in the [ComfyUI_MiniCPM-V-4 repository](https://github.com/IuvenisSapiens/ComfyUI_MiniCPM-V-4). Installing this additional addon will make the "Display Text node" available for use.

## Installation

- Install from [ComfyUI Manager](https://github.com/ltdrdata/ComfyUI-Manager) (search for `Qwen2`)

- Download or git clone this repository into the `ComfyUI\custom_nodes\` directory and run:

```python
pip install -r requirements.txt
```

## Download Models

All the models will be downloaded automatically when running the workflow if they are not found in the `ComfyUI\models\prompt_generator\` directory.
