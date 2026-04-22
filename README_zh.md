

<!-- ## **ControlFoley** -->

[English](./README.md)

<div align="center">

# ControlFoley: Unified and Controllable Video-to-Audio Generation with Cross-Modal Conflict Handling

<p align="center">
  <a href="https://arxiv.org/abs/2604.15086" style="text-decoration:none"><img src="https://img.shields.io/badge/arXiv-2506.21448-b31b1b.svg" alt="arXiv"/></a>
  &nbsp;
  <a href="https://github.com/xiaomi-research/controlfoley" style="text-decoration:none"><img src="https://img.shields.io/badge/GitHub.io-Code-blue?logo=Github&style=flat-square" alt="GitHub"/></a>
  &nbsp;
  <a href="https://yjx-research.github.io/ControlFoley_web_page/" style="text-decoration:none"><img src="https://img.shields.io/badge/Project Page-Project-blue" alt="Project Page"/></a>
  &nbsp;
  <a href="https://yjx-research.github.io/ControlFoley/" style="text-decoration:none"><img src="https://img.shields.io/badge/Demo Page-Demo-blue" alt="Demo Page"/></a>
  &nbsp;
  <a href="https://huggingface.co/YJX-Xiaomi/ControlFoley" style="text-decoration:none"><img src="https://img.shields.io/badge/HuggingFace-Models-orange?logo=huggingface" alt="Hugging Face"/></a>
  &nbsp;
  <a href="https://clawhub.ai/yjx-research/controlfoley-audio-generator" style="text-decoration:none"><img src="https://img.shields.io/badge/ClawHub-ClawHub-red" alt="ClawHub"/></a>
</p>

</div>

<p align="center">
如果您觉得这个项目有用，不妨考虑点个星标⭐️~
</p>


<div align="center">

<hr style="border: none; border-top: 3px solid #333; margin: 16px 0;">

### 👥 **Authors**

<div>
    <!-- Row 1: 6 authors -->
    <div style="margin-bottom: 2px;">
        杨剑轩<sup>1*†</sup>,&nbsp;
        郭新月<sup>1*</sup>,&nbsp;
        程智<sup>1,2</sup>,&nbsp;
        王凯<sup>1,2</sup>,&nbsp;
        张李攀<sup>1</sup>,&nbsp;
        胡锦杰<sup>1</sup>
    </div>
    <!-- Row 2: 7 authors -->
    <div>
        纪强<sup>1</sup>,&nbsp;
        曹益华<sup>1</sup>,&nbsp;
        孟逸浩<sup>1,2</sup>,&nbsp;
        崔昭悦<sup>1,2</sup>,&nbsp;
        刘孟美<sup>1</sup>,&nbsp;
        孟猛<sup>1</sup>,&nbsp;
        栾剑<sup>1</sup>
    </div>
</div>
<!-- Affiliations -->
<div>
    <sup>1</sup>小米大模型Plus. &nbsp;&nbsp; <sup>2</sup>武汉大学
    <br>
    *同等贡献 &nbsp;&nbsp; †通讯作者
</div>
</div>

<hr style="border: none; border-top: 3px solid #333; margin: 16px 0;">

## 📰 **新闻**

- [2026-04] 技术报告发布于 [arXiv](https://arxiv.org/abs/2604.15086)。
- [2026-04] [项目页面](https://yjx-research.github.io/ControlFoley_web_page/) 已上线。
- [2026-04] [推理代码](https://github.com/xiaomi-research/controlfoley) 和 [预训练模型](https://huggingface.co/YJX-Xiaomi/ControlFoley) 已发布。
- [2026-04] 在线推理已上线于 [项目推理接口](https://yjx-research.github.io/ControlFoley_web_page/#try-gen)，即刻体验！
- [2026-04] 发布 skill [ControlFoley Audio Generator](https://clawhub.ai/yjx-research/controlfoley-audio-generator)。

<hr style="border: none; border-top: 3px solid #333; margin: 16px 0;">

## 🔄 **更新**

- [x] 在 arXiv 上发布技术报告。
- [x] 发布项目页面。
- [x] 发布推理代码与预训练模型。
- [x] 发布在线推理代码（可在项目页面获取）。
- [x] 发布 skill。

<hr style="border: none; border-top: 3px solid #333; margin: 16px 0;">

## 📺 **介绍视频**

https://github.com/user-attachments/assets/3c8949ee-9341-426a-94ff-2d24a4bc5175

如需获取本模型的更多结果, 请访问[项目界面](https://yjx-research.github.io/ControlFoley_web_page/)。 如需查看本方法与其他方法的对比结果, 请访问[Demo界面](https://yjx-research.github.io/ControlFoley/)。

<hr style="border: none; border-top: 3px solid #333; margin: 16px 0;">

## 🎧 **概述**

ControlFoley 是一款统一且可控的多模态视频生成音频（V2A）框架，可通过视频、文本及参考音频对音频的生成实现精准控制。

与现有依赖单一模态或在输入冲突时表现不佳的方法不同，ControlFoley 旨在处理复杂的多模态交互，即便在输入模态信息不一致的情况下也能保持出色的可控性。

<hr style="border: none; border-top: 3px solid #333; margin: 16px 0;">

## 🎨 **Tease Figure**

<div align="center">
    <img src="assets/tease.png" width="100%">
    <p style="margin-top: 8px; text-align: center; font-style: italic;">
        左侧：ControlFoley 框架概述，该框架具备三种多模态调节模式，可实现可控的视频同步音频生成。右侧：视频生成音频模型的性能雷达图。
    </p>
</div>

<hr style="border: none; border-top: 3px solid #333; margin: 16px 0;">

## 🚀 **功能**

ControlFoley 支持广泛的应用场景：

- 🎬 <strong>文本引导的视频生成音频（TV2A）</strong><br>
  在文本引导下进行视频内容自适应配音与同步音效生成。

- 📝 <strong>文本控制的视频生成音频 (TC-V2A)</strong><br>
  视频与文本冲突场景下的音频生成，生成音频的语义与文本提示保持一致，且与视频内容在时序上同步。

- 🎧 <strong>参考音频控制的视频生成音频 (AC-V2A)</strong><br>
  基于参考音频进行音频生成，生成音频的音色与参考音频保持一致，并与视频内容在时间上同步。

- 📝 <strong>文本生成音频 (T2A)</strong><br>
  从文本提示直接生成音频，作为我们统一框架的一项附加功能。

<hr style="border: none; border-top: 3px solid #333; margin: 16px 0;">

## 🧠 **核心创新点**

<div align="center">
    <img src="assets/controlfoley.png" width="100%">
</div>

- <strong>面向鲁棒多模态控制的联合视觉编码：</strong>
  结合 CLIP 与 CAV-MAE-ST 表征，同时捕捉视觉-文本以及视觉-听觉的关联，提升模态冲突场景下的模型鲁棒性。

- <strong>以音色为核心的参考音频控制：</strong>
  在抑制时序信息的同时提取全局音色表征，实现精准的声学风格控制，且不影响同步效果。

- <strong>基于统一对齐的多模态鲁棒训练：</strong>
  提出全模态随机丢弃机制与统一的多模态表征对齐优化目标，用以提升模型在各类模态组合下的鲁棒性。

- <strong>VGGSound-TVC 基准数据集：</strong>
  一种用于评估视觉-文本语义冲突下文本可控性的全新基准数据集。

<hr style="border: none; border-top: 3px solid #333; margin: 16px 0;">

## 🧪 **VGGSound-TVC 基准数据集**

我们提出 VGGSound-TVC，用于评估在不同程度的视觉-文本冲突下视频生成音频的文本可控性。在该数据集中，视频的文本描述将按照下述规则进行重构。

- L0 → 没有冲突，文字描述与视频内容保持一致。
- L1_subject →  在主语层面引入轻微语义冲突，即动作描述保持不变，而发声主语被替换。
- L1_action → 在动作层面引入轻微语义冲突，即主体保持不变，而动作描述被修改。
- L2 → 中等程度的语义冲突，文本描述属于不同的语义类别，同时保持与视觉内容相似的时间结构或声学节奏。
- L3 → 强烈冲突，文本描述被随机替换。

上述基准使得我们能够在视觉-文本的不一致性不断加剧的情况下，对模态主导性与可控性开展系统性分析。以下为来自 VGGSound-TVC 数据集的示例样本。
<div align="center">
    <img src="assets/benchmark.png" width="100%">
</div>

<hr style="border: none; border-top: 3px solid #333; margin: 16px 0;">

## 📊 **性能**

ControlFoley 在多项视频生成音频任务中均表现出色，展现出优异的生成质量与强大的可控性。

🎬 <strong>TV2A</strong>

ControlFoley 在多个基准测试中均取得了当前最优性能，其中包括 VGGSound-Test、Kling-Audio-Eval 以及 MovieGen-Audio-Bench。

- 最高 CLAP 得分（语义对齐效果最佳）
- 最低 DeSync 得分（时间同步性能最佳）
- 最高 IS 得分（最好的音频质量）——比其他模型提升将近 27% (22.08 vs. 17.36 on VGGSound测试集)。

<div align="center">
    <img src="assets/result1.png" width="80%">
</div>

📝 <strong>TC-V2A</strong>

ControlFoley 在视觉与文本冲突不断加剧的情况下，仍展现出强大的文本可控性。

- 在各类冲突等级下均保持最佳的 CLAP（文本对齐）性能
- 有效减少冲突情境下的IB得分（降低对视觉的依赖）
- 在可控性与生成质量之间实现更优的平衡

<div align="center">
    <img src="assets/result2.png" width="60%">
</div>

🎧 <strong>AC-V2A</strong>

ControlFoley 在 Greatest Hits 数据集上的所有评估指标中均取得了最佳性能：

- 音色相似度最佳 （Resemblyzer）  
- 时间同步最佳 (DeSync)  
- 音频质量最佳 (IS)  
  
值得注意的是，该模型的性能优于领域内专用基准模型 CondFoleyGen，展现出了强大的泛化能力。

<div align="center">
    <img src="assets/result3.png" width="50%">
</div>

##
ControlFoley 在与 Kling-Foley 等强大的专用闭源系统相比时，同样展现出具有竞争力甚至更优的性能，这凸显了其作为开源、可控的解决方案的有效性。

<hr style="border: none; border-top: 3px solid #333; margin: 16px 0;">

## 🛠 **快速开始**

### 🔑 **准备**

- Python 3.10+
- PyTorch 2.5.1+
- CUDA 11.8+
- FFmpeg (conda install -c conda-forge ffmpeg)

### 🧱 **安装**

```bash
# 克隆该代码仓库
git clone https://github.com/xiaomi-research/controlfoley
cd controlfoley

# 创建 conda 环境
conda create -n controlfoley python=3.10.16
conda activate controlfoley

# 安装依赖
pip install -r requirements.txt

# 下载预训练权重
pip install huggingface-hub==0.26.2
huggingface-cli download YJX-Xiaomi/ControlFoley --resume-download --local-dir model_weights --local-dir-use-symlinks False
```

或者你也可以从[此处](https://huggingface.co/YJX-Xiaomi/ControlFoley/tree/main/)下载权重文件并把它们放在“model_weights”文件夹.

### 🎨 **推理**

```
python demo.py [选项]

选项:
  --video            TEXT       输入视频文件的路径。 (default: None)
  --audio            TEXT       输入参考音频文件的路径。 (default: None)
  --prompt           TEXT       用于音频生成的文本提示词。 (default: None)
  --negative_prompt  TEXT       用于音频生成的负向文本提示词。 (default: None)
  --duration         FLOAT      生成音频的时长，单位为秒。 (default: 8.0)
  --output           TEXT       生成的音频文件的输出目录。 (default: ./output)
```

### 📌 **支持的任务**

| Task   | video      | audio      | prompt   |
|--------|------------|------------|----------|
| TV2A   | required   | None       | required |
| TC-V2A | required   | None       | required |
| AC-V2A | required   | required   | optional |
| V2A    | required   | None       | None     |
| T2A    | None       | None       | required |

### 📋 **使用示例**

- TV2A

```bash
python demo.py --video "assets/001.mp4" --prompt "the skateboard wheels scraping and grinding on the ground." --duration 8.0 --output "./output"
```

- TC-V2A

```bash
python demo.py --video "assets/002.mp4" --prompt "man whistling." --duration 8.0 --output "./output"
```

- AC-V2A

```bash
python demo.py --video "assets/003.mp4" --audio "assets/003.wav" --duration 8.0 --output "./output"
```

- V2A

```bash
python demo.py --video "assets/004.mp4" --duration 8.0 --output "./output"
```

- T2A

```bash
python demo.py --prompt "A bird sings melodically in a forest." --duration 8.0 --output "./output"
```

<hr style="border: none; border-top: 3px solid #333; margin: 16px 0;">

## 📝 **引用**

如果您觉得本代码库对您有所帮助，欢迎引用我们的论文：

```bibtex
@misc{yang2026controlfoleyunifiedcontrollablevideotoaudio,
  title={ControlFoley: Unified and Controllable Video-to-Audio Generation with Cross-Modal Conflict Handling}, 
  author={Jianxuan Yang and Xinyue Guo and Zhi Cheng and Kai Wang and Lipan Zhang and Jinjie Hu and Qiang Ji and Yihua Cao and Yihao Meng and Zhaoyue Cui and Mengmei Liu and Meng Meng and Jian Luan},
  year={2026},
  eprint={2604.15086},
  archivePrefix={arXiv},
  primaryClass={cs.MM},
  url={https://arxiv.org/abs/2604.15086}, 
}
```

<hr style="border: none; border-top: 3px solid #333; margin: 16px 0;">

## 🔒 **许可证**

本代码库采用 [Apache License 2.0](./LICENSE) 许可协议， [模型权重](https://huggingface.co/YJX-Xiaomi/ControlFoley/tree/main/) 采用 [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) 许可协议。

<hr style="border: none; border-top: 3px solid #333; margin: 16px 0;">

## 🙏 **致谢**

本项目使用以下数据集：<br>
VGGSound，Kling-Audio-Eval，The Greatest Hits (<a href="https://creativecommons.org/licenses/by/4.0/" target="_blank" style="color:#007bff; text-decoration:none;">CC BY 4.0</a>)
以及 MovieGen-Audio-Bench (<a href="https://creativecommons.org/licenses/by-nc/4.0/" target="_blank" style="color:#dc3545; text-decoration:none;">CC BY-NC 4.0</a>)。<br>
所有数据仅用于<strong>学术及非商业展示用途</strong>。

感谢以下项目的贡献：<br>
[stable-audio-tools](https://github.com/Stability-AI/stable-audio-tools)，[MMAudio](https://github.com/hkchengrex/MMAudio)，[Make-An-Audio 2](https://github.com/bytedance/Make-An-Audio-2)，[Synchformer](https://github.com/v-iashin/Synchformer)以及 [audiocraft](https://github.com/facebookresearch/audiocraft)。<br>

<hr style="border: none; border-top: 3px solid #333; margin: 16px 0;">

## 📞 **联系**

如果您有任何疑问或建议，欢迎随时通过邮箱 yangjianxuan@xiaomi.com 与我们联系。

<hr style="border: none; border-top: 3px solid #333; margin: 16px 0;">

<div align="center">

2026 ControlFoley Project. All Rights Reserved.

</div>
