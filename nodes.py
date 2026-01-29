import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import folder_paths
from safetensors.torch import save_file
import comfy.model_management

try:
    from diffsynth.pipelines.z_image import (
        ZImagePipeline, 
        ModelConfig, 
        ZImageUnit_Image2LoRAEncode, 
        ZImageUnit_Image2LoRADecode
    )
    DIFFSYNTH_AVAILABLE = True
except ImportError:
    DIFFSYNTH_AVAILABLE = False
    print("\n[CRITICAL WARNING] DiffSynth-Studio not found! Please install it.\n")

# --- MONKEY PATCH FOR TORCH.CPU ---
# Fixes AttributeError: module 'torch.cpu' has no attribute 'empty_cache'
# This is a bug in DiffSynth-Studio when running in CPU mode.
if hasattr(torch, "cpu") and not hasattr(torch.cpu, "empty_cache"):
    torch.cpu.empty_cache = lambda: None
# ----------------------------------

def create_optimized_configs(vram_available_gb):
    """
    生成 VRAM 优化的差异化配置
    
    Args:
        vram_available_gb: 可用 VRAM (GB)
    
    Returns:
        siglip_config, dino_config, i2l_config, vram_limit_bytes
    """
    
    # SigLIP2: 小模型 (~1.5GB)，全程 GPU
    siglip_config = {
        "offload_dtype": torch.bfloat16,
        "offload_device": "cuda",
        "onload_dtype": torch.bfloat16,
        "onload_device": "cuda",
        "preparing_dtype": torch.bfloat16,
        "preparing_device": "cuda",
        "computation_dtype": torch.bfloat16,
        "computation_device": "cuda",
    }
    
    # DINOv3: 大模型 (7B)，FP8 存储 + GPU 计算
    dino_config = {
        "offload_dtype": torch.float8_e4m3fn,
        "offload_device": "cpu",
        "onload_dtype": torch.float8_e4m3fn,
        "onload_device": "cpu",
        "preparing_dtype": torch.float8_e4m3fn,
        "preparing_device": "cuda",
        "computation_dtype": torch.bfloat16,
        "computation_device": "cuda",
    }
    
    # Z-Image: 中型模型 (~2GB)，FP8 存储 + GPU 计算
    i2l_config = {
        "offload_dtype": torch.float8_e4m3fn,
        "offload_device": "cpu",
        "onload_dtype": torch.float8_e4m3fn,
        "onload_device": "cpu",
        "preparing_dtype": torch.float8_e4m3fn,
        "preparing_device": "cuda",
        "computation_dtype": torch.bfloat16,
        "computation_device": "cuda",
    }
    
    # VRAM 限制: 可用 VRAM - 2GB 安全缓冲
    vram_limit_bytes = (vram_available_gb - 2.0) * 1024**3
    
    return siglip_config, dino_config, i2l_config, vram_limit_bytes

class ZImageI2L_Loader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_filename": (["Z-Image-i2L.safetensors"],), # Default simple list, user can expand if needed
            }
        }

    RETURN_TYPES = ("ZI2L_FILE",)
    RETURN_NAMES = ("z_image_file_path",)
    FUNCTION = "load_local"
    CATEGORY = "Z-Image-i2L"

    def load_local(self, model_filename):
        base_path = folder_paths.models_dir
        base_dir = os.path.join(base_path, "I2L")
        
        lora_dir = os.path.join(base_dir, "Z-Image")
        clips_dir = os.path.join(base_dir, "CLIPS")

        print(f"\n{'='*60}")
        print(f"🔧 Z-Image i2L Pipeline Loader")
        print(f"📁 Base Directory: {base_dir}")
        print(f"{'='*60}\n")
        
        i2l_path = os.path.join(lora_dir, model_filename)
        
        # Check required files
        siglip_path = os.path.join(clips_dir, "SigLIP2-G384.safetensors")
        dino_path = os.path.join(clips_dir, "DINOv3-7B.safetensors")

        print("📋 Checking Required Files:\n")
        
        files_to_check = [
            ("i2L Model", i2l_path),
            ("SigLIP2-G384", siglip_path),
            ("DINOv3-7B", dino_path),
        ]
        
        missing_files = []
        for name, path in files_to_check:
            if os.path.exists(path):
                size_mb = os.path.getsize(path) / (1024 * 1024)
                print(f"  ✅ {name}: {size_mb:.1f} MB")
            else:
                print(f"  ❌ {name}: NOT FOUND")
                missing_files.append((name, path))
        
        if missing_files:
            print(f"\n{'='*60}")
            print(f"❌ ERROR: Missing {len(missing_files)} file(s)")
            print(f"{'='*60}\n")
            print("Please place the following files in the correct locations:\n")
            for name, path in missing_files:
                print(f"  • {name}")
                print(f"    Expected: {path}\n")
            print(f"{'='*60}\n")
            raise FileNotFoundError("Required model files are missing. See console output above.")

        print(f"\n{'='*60}")
        print(f"✅ All Required Files Found!")
        print(f"{'='*60}\n")

        return (i2l_path,)

class ZImageI2L_Apply:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "z_image_file_path": ("ZI2L_FILE",),
                "images": ("IMAGE",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "use_cpu_for_vision": ("BOOLEAN", {"default": False, "label": "Force CPU for Vision Encoders (Safe for 8GB VRAM)"}),
            },
            "optional": {
                "model": ("MODEL",),
            }
        }

    RETURN_TYPES = ("MODEL", "lora_weights") 
    RETURN_NAMES = ("patched_model", "lora_data")
    FUNCTION = "apply_style"
    CATEGORY = "Z-Image-i2L"

    def apply_style(self, z_image_file_path, images, strength, use_cpu_for_vision, model=None):
        if not DIFFSYNTH_AVAILABLE: 
            raise Exception("❌ DiffSynth-Studio not installed! Run: pip install git+https://github.com/modelscope/DiffSynth-Studio.git")

        # Clear memory safely
        try:
            comfy.model_management.unload_all_models()
            comfy.model_management.soft_empty_cache()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"⚠️ Warning during cache clearing: {e}")

        # 检测可用 VRAM
        vram_available_gb = 0
        if torch.cuda.is_available():
            vram_total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            vram_free_gb = torch.cuda.mem_get_info("cuda")[0] / (1024**3)
            vram_available_gb = vram_free_gb
            print(f"💾 VRAM Info: Total={vram_total_gb:.1f}GB, Free={vram_free_gb:.1f}GB")

        # 配置选择逻辑
        if use_cpu_for_vision:
            # 用户明确要求 CPU 模式 → 纯 CPU（慢速但安全）
            mode_name = "SAFE (CPU)"
            device_str = "cpu"
            
            # 统一 CPU 配置
            unified_config = {
                "offload_dtype": torch.bfloat16,
                "offload_device": "cpu",
                "onload_dtype": torch.bfloat16,
                "onload_device": "cpu",
                "preparing_dtype": torch.bfloat16,
                "preparing_device": "cpu",
                "computation_dtype": torch.bfloat16,
                "computation_device": "cpu",
            }
            siglip_config = unified_config.copy()
            dino_config = unified_config.copy()
            i2l_config = unified_config.copy()
            vram_limit_bytes = None
            
        else:
            # 默认使用优化模式 (GPU+CPU 混合，FP8 存储)
            mode_name = "OPTIMIZED (Dynamic GPU+CPU)"
            device_str = "cuda"
            
            siglip_config, dino_config, i2l_config, vram_limit_bytes = create_optimized_configs(vram_available_gb)
            
            print(f"⚙️ VRAM Optimization Mode Enabled")
            print(f"   VRAM Limit: {vram_limit_bytes / (1024**3):.1f}GB")
            print(f"   SigLIP2: BF16 (GPU, ~1.5GB)")
            print(f"   DINOv3: FP8 storage + BF16 compute (Dynamic, ~5.6GB)")
            print(f"   Z-Image: FP8 storage + BF16 compute (Dynamic, ~1.6GB)")
        
        print(f"\n{'='*60}")
        print(f"🎨 Z-Image i2L Apply - {mode_name}")
        print(f"📊 Strength: {strength}")
        print(f"🖼️ Images: {len(images)}")
        print(f"{'='*60}\n")
        
        base_path = folder_paths.models_dir
        clips_dir = os.path.join(base_path, "I2L", "CLIPS")
        
        siglip_path = os.path.join(clips_dir, "SigLIP2-G384.safetensors")
        dino_path = os.path.join(clips_dir, "DINOv3-7B.safetensors")

        # Convert ComfyUI images to PIL
        pil_images = []
        for img in images:
            img_np = np.clip(255. * img.cpu().numpy(), 0, 255).astype(np.uint8)
            pil_images.append(Image.fromarray(img_np))

        try:
            print("\n📄 Loading Pipeline...")
            
            # Build model configs list with differentiated configs
            model_configs = [
                ModelConfig(
                    model_id="DiffSynth-Studio/General-Image-Encoders", 
                    origin_file_pattern=siglip_path, 
                    **siglip_config
                ),
                ModelConfig(
                    model_id="DiffSynth-Studio/General-Image-Encoders", 
                    origin_file_pattern=dino_path, 
                    **dino_config
                ),
                ModelConfig(
                    model_id="DiffSynth-Studio/Z-Image-i2L", 
                    origin_file_pattern=z_image_file_path, 
                    **i2l_config
                )
            ]
            
            # Pipeline 初始化
            pipeline_kwargs = {
                "torch_dtype": torch.bfloat16,
                "device": device_str,
                "model_configs": model_configs,
                "tokenizer_config": None
            }

            # 添加 vram_limit（仅在优化模式）
            if vram_limit_bytes is not None:
                pipeline_kwargs["vram_limit"] = vram_limit_bytes
                print(f"🔧 Pipeline vram_limit set to {vram_limit_bytes / (1024**3):.1f}GB")

            pipe = ZImagePipeline.from_pretrained(**pipeline_kwargs)
            
            print("✅ Pipeline Loaded Successfully\n")

            print("🔍 Analyzing images...")
            with torch.no_grad():
                # VRAM 监控（仅优化模式）
                if not use_cpu_for_vision and torch.cuda.is_available():
                    vram_start = torch.cuda.memory_allocated() / (1024**3)
                    print(f"   VRAM start: {vram_start:.2f}GB")
                
                # 编码图像
                embs = ZImageUnit_Image2LoRAEncode().process(
                    pipe, 
                    image2lora_images=pil_images
                )
                print("✅ Images encoded\n")
                
                if not use_cpu_for_vision and torch.cuda.is_available():
                    vram_after_encode = torch.cuda.memory_allocated() / (1024**3)
                    print(f"   VRAM after encode: {vram_after_encode:.2f}GB")
                
                # 确保张量在正确设备
                forced_embs = {}
                for k, v in embs.items():
                    if isinstance(v, torch.Tensor):
                        forced_embs[k] = v.to(device_str)
                    else:
                        forced_embs[k] = v

                # 生成 LoRA 权重
                print("⚙️ Generating LoRA weights...")
                result = ZImageUnit_Image2LoRADecode().process(pipe, **forced_embs)
                lora_weights = result["lora"]
                print("✅ LoRA weights generated\n")
                
                if not use_cpu_for_vision and torch.cuda.is_available():
                    vram_peak = torch.cuda.max_memory_allocated() / (1024**3)
                    print(f"   VRAM peak: {vram_peak:.2f}GB")

            # Apply strength multiplier
            if strength != 1.0:
                print(f"📊 Applying strength multiplier: {strength}")
                for k in lora_weights: 
                    lora_weights[k] *= strength
            
            print(f"\n{'='*60}")
            print(f"✅ SUCCESS: LoRA Generated!")
            print(f"📦 Contains {len(lora_weights)} weight tensors")
            print(f"{'='*60}\n")

        except RuntimeError as e:
            print(f"\n{'='*60}")
            print(f"❌ ERROR: Inference Failed (RuntimeError)")
            print(f"{'='*60}")
            print(f"Error: {e}\n")
            
            if "device" in str(e).lower() or "type" in str(e).lower():
                print("💡 Device mismatch detected.")
                if not use_cpu_for_vision:
                    print("👉 Try enabling 'Force CPU for Vision Encoders' in the node settings.")
            
            if "out of memory" in str(e).lower() or "cuda out of memory" in str(e).lower():
                print(f"💡 VRAM OOM Detected:")
                if torch.cuda.is_available():
                    vram_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    vram_peak = torch.cuda.max_memory_allocated() / (1024**3)
                    print(f"   Total VRAM: {vram_total:.1f}GB")
                    print(f"   Peak Usage: {vram_peak:.1f}GB")
                    print(f"   Shortage: {vram_peak - vram_total:.1f}GB")
                print("👉 Solutions:")
                print("   1. Enable 'Force CPU for Vision Encoders'")
                print("   2. Process images one at a time")
            
            import traceback
            traceback.print_exc()
            lora_weights = {}

        except Exception as e:
            print(f"\n{'='*60}")
            print(f"❌ ERROR: Inference Failed")
            print(f"{'='*60}")
            print(f"Error: {e}\n")
            import traceback
            traceback.print_exc()
            lora_weights = {}

        # Cleanup
        if 'pipe' in locals():
            del pipe
        
        # Final cleanup safely
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass
        
        return (model.clone() if model else None, lora_weights)

class ZImageI2L_Save:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "lora_data": ("lora_weights",),
                "filename": ("STRING", {"default": "z_image_style_lora"}),
            }
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "save_lora"
    CATEGORY = "Z-Image-i2L"

    def save_lora(self, lora_data, filename):
        if not lora_data:
            print("⚠️ No LoRA data to save (inference failed)")
            return ()
        
        # Remove .safetensors if user added it
        if filename.endswith('.safetensors'):
            filename = filename[:-12]
            
        out_dir = os.path.join(folder_paths.get_output_directory(), "loras")
        os.makedirs(out_dir, exist_ok=True)
        
        save_path = os.path.join(out_dir, f"{filename}.safetensors")
        
        try:
            save_file(lora_data, save_path)
            file_size = os.path.getsize(save_path) / (1024 * 1024)
            
            print(f"\n{'='*60}")
            print(f"💾 LoRA Saved Successfully!")
            print(f"📁 Path: {save_path}")
            print(f"📊 Size: {file_size:.2f} MB")
            print(f"📦 Tensors: {len(lora_data)}")
            print(f"{'='*60}\n")
        except Exception as e:
            print(f"❌ Failed to save LoRA: {e}")
        
        return ()

# Node mappings
NODE_CLASS_MAPPINGS = {
    "ZImageI2L_PipelineLoader": ZImageI2L_Loader,
    "ZImageI2L_Apply": ZImageI2L_Apply,
    "ZImageI2L_Save": ZImageI2L_Save
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZImageI2L_PipelineLoader": "🔧 Z-Image i2L Loader",
    "ZImageI2L_Apply": "🎨 Z-Image i2L Apply",
    "ZImageI2L_Save": "💾 Save LoRA"
}
