# 文本条件约束力弱的原因分析

当前 `NodeDiffusionTransformer` 使用"文本 token 拼接 + 全局 self-attention"的条件化方式，
评估结果表明提示词对生成坐标的约束能力极弱。以下四个原因层层叠加，共同导致了这一问题。

---

## 原因一：图结构已把文本信息覆盖

文本描述的内容（"走廊连接卧室和卫生间"）在邻接矩阵里已经以边的形式编码。
模型读图结构就足以完成噪声预测任务，文本成为冗余信息。
冗余信息不会产生有效的训练梯度，模型没有动力去学习利用文本。

---

## 原因二：self-attention 中文本 token 与节点竞争注意力权重

`EncoderLayer.forward` 中全局注意力的实现：

```python
# model.py EncoderLayer.forward
global_out = self.global_attn(x2, x2, x2, pad_mask)
```

序列 `x2 = [text_tokens | node_tokens]`，节点在分配注意力时，
文本 token 和其他节点同时是候选项（Q、K、V 均来自同一序列）。
图结构信号直接、强烈；文本语义到坐标的映射抽象、模糊。
训练早期图结构路径就能快速降低 loss，权重自然集中到节点侧，文本 token 权重被挤压至接近零。

对比 Stable Diffusion 的 cross-attention：
```
Q = 图像/节点特征
K, V = 文本编码器输出   ← 文本是唯一候选，无法被忽视
```

---

## 原因三：文本权重趋零后梯度死亡，不可逆

attention 权重接近零
→ 通过文本路径的梯度也接近零
→ 模型没有信号来学习"如何从文本中获取坐标约束"
→ 文本路径永久固化在无效状态

这是一个自我强化的收敛过程：一旦模型选择依赖图结构，文本路径的梯度就消失，再也无法恢复。

---

## 原因四：文本编码器缺乏预训练语义

当前使用从零训练的 BPE `nn.Embedding`（`bpe_vocab_size=10000`）：

```python
# model.py NodeDiffusionTransformer.__init__
self.text_embed = nn.Embedding(bpe_vocab_size, model_channels, padding_idx=0)
```

"卧室"与"走廊"的词向量之间没有任何预训练的语义距离，
模型即使想利用文本，也无法从这些向量中提取出有意义的空间约束。

对比 DALL-E 2 / Stable Diffusion 使用 CLIP / T5 文本编码器：
词向量本身就携带了在数十亿图文对上对齐的丰富语义，文本条件天然有效。

---

## Cross-Attention 的优势

### 架构层面：文本成为强制依赖

Cross-attention 中 Q、K、V 来自不同序列：

```python
# 节点特征作为 Q，文本特征作为 K/V
Q = W_q(node_feat)   # [B, N_nodes, D]
K = W_k(text_feat)   # [B, N_text,  D]
V = W_v(text_feat)   # [B, N_text,  D]

attn = softmax(Q @ K.T / sqrt(D))  # [B, N_nodes, N_text]
out  = attn @ V                    # [B, N_nodes, D]
```

attention 权重矩阵只有 `N_nodes × N_text` 维，**节点只能在文本 token 里分配权重**，
没有"转而看其他节点"的选项。模型想降低 loss，唯一的路径是学会读懂文本。

### 训练动力学层面：梯度强制流过文本路径

- Self-attention 拼接：文本权重可趋零 → 文本路径梯度死亡（不可逆）
- Cross-attention：文本是 K/V 的唯一来源 → 梯度必须流过文本编码 → 模型被迫学习文本语义

### 推理层面：每步噪声预测都由文本引导

cross-attention 位于网络内部，是 forward pass 的组成部分，
而不是对噪声预测结果的后处理。

```
x_t ──→ [ adj_attn → cross_attn(K/V=text) → FFN ] ──→ ε_pred ──→ x_{t-1}
                          ↑
                  文本在预测噪声的过程中注入
                  （不是在减去噪声之后）
```

推理时 1000 步每步都通过 cross-attention 查询文本，
累积效果是把去噪轨迹从随机噪声引导到符合文本描述的坐标分布。

### 与 CFG 配合放大文本信号

加入 Classifier-Free Guidance 后，每步同时跑有条件和无条件两次 forward：

```
ε_cond   = model(x_t, t, text)
ε_uncond = model(x_t, t, null)
ε_final  = ε_uncond + w × (ε_cond − ε_uncond)
```

`(ε_cond − ε_uncond)` 是文本对噪声预测方向的净影响，
guidance scale `w`（通常 3～7.5）将这个方向放大，
使最终坐标更强烈地偏向文本描述的布局。

---

## 改进方向

| 问题 | 改进方案 |
|------|---------|
| 文本可被忽视 | 将 `global_attn` 改为专用 cross-attention（节点为 Q，文本为 K/V） |
| 推理时信号弱 | 加入 Classifier-Free Guidance（训练时随机 null 文本，推理时放大条件方向） |
| 文本编码器无语义 | 替换为预训练小模型（sentence-transformers / bert-base），冻结文本编码器 |
