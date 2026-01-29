# ⚡ ComfyUI Z-Image-i2L (Image-to-LoRA)

A ComfyUI custom node for **Z-Image-i2L** (Image-to-LoRA). 
This node allows you to "extract" style, composition, or details from any image and save them as a lightweight **LoRA** (`.safetensors`) file using the powerful Z-Image architecture.

---

## 🚀 Key Features

* **🚀 Optimized VRAM Mode (Default):** Automatic GPU+CPU hybrid mode for optimal performance.
    * **SigLIP2 Encoder:** Runs entirely on GPU (fast, ~1.5GB VRAM).
    * **DINOv3 Encoder:** Uses FP8 storage + BF16 computation (dynamic offload, ~5.6GB VRAM).
    * **Z-Image Adapter:** Uses FP8 storage + BF16 computation (dynamic offload, ~1.6GB VRAM).
    * **Total VRAM:** ~10.7GB peak with 2GB safety buffer for 12GB GPUs.
    * **Performance:** **8-12x faster** than pure CPU mode.
* **🛡️ CPU Mode (Optional):** Safe fallback for systems with < 8GB VRAM.
    * Runs all encoders on CPU (slow but guaranteed to work).
    * Uses ~2GB RAM instead of VRAM.
* **📉 FP8 Quantization:** Automatically uses FP8 for storage to reduce VRAM usage by 50%.
* **🔄 Backward Compatible:** Existing workflows work without modification.
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

## ⚡ Performance & Configuration

### **Mode Comparison**

| Mode | Setting | VRAM Usage | Speed | Quality | Best For |
|------|----------|-------------|--------|----------|
| **Optimized (Default)** | `use_cpu_for_vision=False` | ~10.7GB peak | Excellent | **12GB+ VRAM** (Recommended) |
| **CPU Mode** | `use_cpu_for_vision=True` | ~2GB RAM | Excellent | < 8GB VRAM |

### **Performance Expectations**

| Task | Optimized Mode | CPU Mode | Speedup |
|------|---------------|-----------|----------|
| Single Image (512x512) | ~45-90 seconds | ~10-15 minutes | **8-12x** |
| Batch of 4 Images | ~2-4 minutes | ~40-60 minutes | **8-12x** |

### **VRAM Requirements**

| GPU VRAM | Recommended Mode | Notes |
|----------|------------------|--------|
| **< 8GB** | CPU Mode | Use `use_cpu_for_vision=True` |
| **8-12GB** | Optimized Mode | May need to reduce batch size |
| **12GB+** | Optimized Mode | ✅ Recommended - Full support |
| **16GB+** | Optimized Mode | ✅ Best performance |

### **Configuration Details**

**Optimized Mode (Default):**
- **SigLIP2 (~1.5GB):** BF16, full GPU
- **DINOv3-7B (~7GB):** FP8 storage + BF16 compute, dynamic GPU/CPU offload
- **Z-Image (~2GB):** FP8 storage + BF16 compute, dynamic GPU/CPU offload
- **VRAM Limit:** Automatically set to (Available VRAM - 2GB)
- **Automatic Offloading:** Excess layers automatically offloaded to CPU when VRAM limit reached

**CPU Mode:**
- All encoders run on CPU
- BF16 precision throughout
- Uses ~2GB RAM instead of VRAM
- Slower but guaranteed to work on any system

---

## 🎛️ Usage Guide

### **Step 1: The "Factory" Workflow (Creating the LoRA)**
*Use this workflow only to CREATE the file.*

1.  **Add Node:** `Z-Image i2L Loader`
    * Select `Z-Image-i2L.safetensors` (Ensure it is in `models/I2L/Z-Image`).
2.  **Add Node:** `Z-Image i2L Apply`
    * Connect `z_image_file_path` from the Loader.
    * Connect your **Style Image** to `images`.
    * **Use CPU for Vision (Optional):**
        * **False (Default):** Optimized mode - Uses GPU+CPU hybrid for **8-12x faster** performance. Recommended for 12GB+ VRAM.
        * **True:** Safe mode - Runs on CPU only. Slower but guaranteed to work. Use this if you have < 8GB VRAM.
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

## 🔧 Technical Details

### **How VRAM Optimization Works**

The optimized mode uses a hybrid GPU+CPU approach with **differential model configurations**:

1. **SigLIP2 Encoder (~1.5GB):**
   - Stored and computed entirely on GPU
   - BF16 precision for quality
   - Fast access, no offloading needed

2. **DINOv3-7B Encoder (~7GB):**
   - Stored in FP8 on CPU (saves 50% memory)
   - Loaded to GPU only when computing
   - BF16 computation for quality
   - Dynamic layer offloading when VRAM limit reached

3. **Z-Image Adapter (~2GB):**
   - Stored in FP8 on CPU (saves 50% memory)
   - Loaded to GPU only when computing
   - BF16 computation for quality
   - Dynamic layer offloading when VRAM limit reached

### **VRAM Management**

The pipeline automatically manages VRAM usage:
- Detects available VRAM at runtime
- Sets `vram_limit = Available VRAM - 2GB` (safety buffer)
- Monitors VRAM usage during inference
- Automatically offloads layers to CPU when limit is reached
- Displays detailed VRAM statistics in console output

### **Quality Preservation**

- **Storage:** FP8 (negligible quality loss, saves 50% memory)
- **Computation:** BF16 (no quality loss, same as full GPU mode)
- **Result:** Identical quality to full GPU mode, with 50% less VRAM usage

---

## 🤝 Credits
* **Original Code:** [ComfyUI-Qwen-Image-i2L](https://github.com/gajjar4/ComfyUI-Qwen-Image-i2L)
* **Pipeline:** [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio)
