# VTA Text Conflict Dataset

## 1. Background

Current Video-to-Audio (VTA) generation tasks typically adopt a dual-modal conditioning framework, where the model takes both video and text descriptions as generation conditions. In this framework, text serves as a key control signal for fine-grained guidance over the semantic direction and event category of the generated audio.

However, virtually all training and evaluation data for existing VTA models comes from distributions where video and text are highly consistent. In such settings, the visual and textual modalities tend to provide redundant information, and the model can still generate plausible audio from video content alone, even when the text condition is weakened or ignored. As a result, this data distribution cannot genuinely measure a model's text controllability, nor can it assess the dominance relationship between modalities under conflicting conditions.

To address this issue, we constructed a text conflict evaluation dataset based on VGG-SS (VGG-SS is a video-level audio-visual sound source localization dataset, characterized by clearly visible sounding objects in the videos). In this dataset, text descriptions are deliberately designed to create semantic conflicts with the video content, breaking the original consistent distribution and forcing the model to balance between visual and textual signals during generation.

Furthermore, we designed four conflict intensity levels, ranging from no conflict (L0) to strong conflict (L3), to systematically analyze the model's modal dependency behavior at different conflict intensities and quantify how text controllability changes as conflict intensity increases.

---

## 2. Data Description: Scale and Types

This dataset contains **25,005 video-text sample pairs** in total.

The data is built upon **5,001 videos**, with each video corresponding to **5 different text description versions (4 conflict levels, where L1 includes two variants)** (L0, L1_subject, L1_action, L2, L3), resulting in a total sample count of:

5,001 × 5 = 25,005

Each sample consists of a video clip and its corresponding text label.

### Text Field Descriptions

Each video sample contains the following text fields:

**label_L0**

- Normalized label
- Unified into a "subject + action" structure
- Simplified from the original multi-labels

**label_L1_subject**

- Introduces a minor conflict at the subject level only

**label_L1_action**

- Introduces a minor conflict at the action level only

**label_L2**

- Moderate semantic conflict text
- Crosses semantic categories, but maintains consistent temporal structure and acoustic rhythm

**label_L3**

- Strong conflict text constructed by random category replacement
- Only guarantees that labels conform to the real label distribution

The **L0, L1, and L2 labels** were generated using the **Gemini 2.5 Pro multimodal large language model**.

---

## 3. Known Issues and Limitations

L3-level text is constructed via random category replacement and is intended solely to measure the model's modal dependency behavior under extreme semantic conflict. Since its semantics are not guaranteed to match the structural content of the video, it is **not recommended for use as training data**.

---

## 4. Safety Statement

This dataset is built upon the publicly available dataset **VGG-SS**. The original videos are sourced from publicly accessible online video platforms, and the data itself does not contain user identity information, contact details, or any other personally identifiable private fields.

The text conflict labels in the dataset involve only semantic rewrites of sound categories, with no relation to speech content in videos, personal privacy, political content, religious content, or other sensitive topics.

The video content in the dataset primarily consists of environmental sounds and common event categories, such as:

- Speech
- Musical instrument playing
- Animal sounds
- Natural ambient sounds

The dataset does not contain any illegal, violent, inciting, or otherwise inappropriate content.

This dataset is intended solely for **research and evaluation purposes of multimodal generative models**.
