# ChatHouseDiffusion

Large language models and diffusion models are used to generate and edit the room plan with text prompts.

Code based on [Imagen-pytorch](https://github.com/lucidrains/imagen-pytorch) and [Graphormer](https://github.com/microsoft/Graphormer).

```bib
@misc{qin2024chathousediffusionpromptguidedgenerationediting,
      title={ChatHouseDiffusion: Prompt-Guided Generation and Editing of Floor Plans},
      author={Sizhong Qin and Chengyu He and Qiaoyun Chen and Sen Yang and Wenjie Liao and Yi Gu and Xinzheng Lu},
      year={2024},
      eprint={2410.11908},
      archivePrefix={arXiv},
      primaryClass={cs.HC},
      url={https://arxiv.org/abs/2410.11908},
}
```

---

## 数据转换

输入 JSONL 格式为 `final_graph_dataset_v*.jsonl`，输出目录结构：

```text
data/chathousediffusion/chat/
  images/        ← 训练集语义分割图
  masks/         ← 训练集外轮廓 mask
  texts/         ← 训练集 JSON + texts.csv
  images_test/   ← 验证/测试集
  masks_test/
  texts_test/
```

### 训练集

```bash
python chathousediffusion-comp/convert_to_chathousediffusion.py \
    --data data/jsonl/final_graph_dataset_v3.jsonl \
    --out  data/chathousediffusion/chat \
```

### 验证集（训练时随机抽 224 条）

```bash
python chathousediffusion-comp/convert_to_chathousediffusion.py \
    --data data/jsonl/val_graph_dataset_18k5.jsonl \
    --out  data/chathousediffusion/chat \
    --suffix _test
```

### 测试集

```bash
python chathousediffusion-comp/convert_to_chathousediffusion.py \
    --data data/jsonl/test_graph_dataset_18k5.jsonl \
    --out  data/chathousediffusion/chat \
    --suffix _test
```

---

## 训练

```bash
cd chathousediffusion-comp
CUDA_VISIBLE_DEVICES=0 python train.py
# 从断点恢复
CUDA_VISIBLE_DEVICES=0 python train.py --resume latest
```

模型权重保存至 `results/ours_v1/`，每 5000 步保存一次。

---

## 测试

模型从 HuggingFace 下载（`wzmmmm/chathousediffusion`）：

```bash
HF_ENDPOINT=https://hf-mirror.com hf download wzmmmm/chathousediffusion \
    model-latest.pt params.pkl \
    --local-dir chathousediffusion-comp/results/ours_v1
```

运行测试（全量测试集）：

```bash
CUDA_VISIBLE_DEVICES=0 python chathousediffusion-comp/test.py \
    --results chathousediffusion-comp/results/ours_v1 \
    --data_root data/chathousediffusion/chat \
    --milestone latest
```

结果（生成图、对比图、IoU）保存至 `results/ours_v1/cond_scale-1-latest/`。
