# ⚡ ComfyUI Z-Image-i2L (Image-to-LoRA)

A ComfyUI custom node for **Z-Image-i2L** (Image-to-LoRA). 
This node allows you to "extract" style, composition, or details from any image and save them as a lightweight **LoRA** (`.safetensors`) file using the powerful Z-Image architecture.

---

## 🚀 Key Features

* **🧠 Smart VRAM Detection:** Automatically detects your GPU VRAM.
    * **< 16GB VRAM:** Automatically runs the heavy DINOv3 encoder on **CPU (RAM)** to prevent crashes.
    * **> 16GB VRAM:** Automatically uses **GPU** for maximum speed.
* **📉 FP8 Model Support:** Compatible with FP8 models to save VRAM.
* **✨ One-Click LoRA:** Takes an image input and outputs a ready-to-use LoRA file.

### 📂 Folder Structure & Model Placement

You need to create a folder named `I2L` inside your `ComfyUI/models/` directory.

**Path:** `ComfyUI/models/I2L/`

Inside that folder, create two sub-folders: `CLIPS` and `Z-Image`.

#### **1. The Vision Encoders (Required)**
Place these inside: `ComfyUI/models/I2L/CLIPS/`

| File Name | Description | Source |
| :--- | :--- | :--- |
| **`SigLIP2-G384.safetensors`** | Vision Encoder (Small) | [Download](https://huggingface.co/DiffSynth-Studio/General-Image-Encoders/blob/main/SigLIP2-G384/model.safetensors) |
| **`DINOv3-7B.safetensors`** | Vision Encoder (Large) | [Download](https://huggingface.co/DiffSynth-Studio/General-Image-Encoders/blob/main/DINOv3-7B/model.safetensors) |

#### **2. The Z-Image Adapter**
Place this inside: `ComfyUI/models/I2L/Z-Image/`

| File Name | Description | Source |
| :--- | :--- | :--- |
| **`Z-Image-i2L.safetensors`** | Z-Image i2L Model | [Download](https://modelscope.cn/models/DiffSynth-Studio/Z-Image-i2L) (Rename to `Z-Image-i2L.safetensors`) |

---

## 📦 Installation

### **Method 1: ComfyUI Manager (Coming Soon)**

### **Method 2: Manual Installation**
1.  Navigate to your custom nodes folder:
    ```bash
    cd ComfyUI/custom_nodes
    ```
2.  Clone this repository:
    ```bash
    git clone https://github.com/your-repo/ComfyUI-Z-Image-i2L.git
    ```
3.  Install dependencies:
    ```bash
    cd ComfyUI-Z-Image-i2L
    pip install -r requirements.txt
    ```
4.  Restart ComfyUI.

---

## 🎛️ Usage Guide

### **Step 1: The "Factory" Workflow (Creating the LoRA)**
*Use this workflow only to CREATE the file.*

1.  **Add Node:** `Z-Image i2L Loader`
    * Select `Z-Image-i2L.safetensors` (Ensure it is in `models/I2L/Z-Image`).
2.  **Add Node:** `Z-Image i2L Apply`
    * Connect `z_image_file_path` from the Loader.
    * Connect your **Style Image** to `images`.
    * **Use CPU for Vision:** Enabled by default (Safe for 8GB VRAM). Disable if you have 24GB+ VRAM.
3.  **Add Node:** `Save LoRA`
    * Connect `lora_data` from the Apply node.
    * Set your filename (e.g., `my_cool_style`).
4.  **Queue Prompt:** Wait for the analysis to finish. It will save the file to `ComfyUI/output/loras/`.

### **Step 2: The "Generation" Workflow (Using the LoRA)**
*Once you have the file, you don't need the Z-Image nodes anymore.*

1.  Load your standard Text-to-Image workflow (compatible with the base model, likely Flux or Z-Image depending on what the LoRA is for).
2.  Add a standard **Load LoRA** node.
3.  Select the `my_cool_style.safetensors` you just created.
4.  Generate images!

---

## 🤝 Credits
* **Original Code:** [ComfyUI-Qwen-Image-i2L](https://github.com/gajjar4/ComfyUI-Qwen-Image-i2L)
* **Pipeline:** [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio)
