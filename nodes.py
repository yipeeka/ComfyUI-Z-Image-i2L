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

class ForceCPUWrapper(nn.Module):
    def __init__(self, original_module):
        super().__init__()
        self.original_module = original_module
    
    def forward(self, x, *args, **kwargs):
        # Force input to CPU to avoid "Meta device" errors
        if hasattr(x, "to") and hasattr(x, "device") and x.device.type != "cpu":
            x = x.to("cpu")
            
        # Helper to move tensors to CPU
        def to_cpu(obj):
            if hasattr(obj, "to") and hasattr(obj, "device") and obj.device.type != "cpu":
                return obj.to("cpu")
            return obj

        # Process args and kwargs to ensure CPU compatibility
        args = tuple(to_cpu(arg) for arg in args)
        kwargs = {k: to_cpu(v) for k, v in kwargs.items()}
        
        # CRITICAL FIX: Override device argument
        # DiffSynth encoders default to "cuda" or get_device_type() in forward()
        # We must explicitly force this to "cpu"
        kwargs["device"] = "cpu"
            
        try:
            return self.original_module(x, *args, **kwargs)
        except TypeError:
            # Fallback for modules that might not accept a 'device' kwarg
            kwargs.pop("device", None)
            return self.original_module(x, *args, **kwargs)

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
                "use_cpu_for_vision": ("BOOLEAN", {"default": True, "label": "Force CPU for Vision Encoders (Safe for 8GB VRAM)"}),
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

        device_str = "cpu" if use_cpu_for_vision else "cuda"
        mode_name = "SAFE (CPU)" if use_cpu_for_vision else "FAST (GPU)"
        
        print(f"\n{'='*60}")
        print(f"🎨 Z-Image i2L Apply - {mode_name}")
        print(f"📊 Strength: {strength}")
        print(f"🖼️ Images: {len(images)}")
        print(f"{'='*60}\n")
        
        base_path = folder_paths.models_dir
        clips_dir = os.path.join(base_path, "I2L", "CLIPS")
        
        siglip_path = os.path.join(clips_dir, "SigLIP2-G384.safetensors")
        dino_path = os.path.join(clips_dir, "DINOv3-7B.safetensors")

        # Config for minimal VRAM usage
        active_config = {
            "offload_dtype": torch.bfloat16, 
            "offload_device": "cpu",
            "onload_dtype": torch.bfloat16, 
            "onload_device": "cpu",
            "preparing_dtype": torch.bfloat16, 
            "preparing_device": device_str,
            "computation_dtype": torch.bfloat16, 
            "computation_device": device_str,
        }

        # Convert ComfyUI images to PIL
        pil_images = []
        for img in images:
            img_np = np.clip(255. * img.cpu().numpy(), 0, 255).astype(np.uint8)
            pil_images.append(Image.fromarray(img_np))

        try:
            print("\n📄 Loading Pipeline...")
            
            # Build model configs list
            model_configs = [
                ModelConfig(
                    model_id="DiffSynth-Studio/General-Image-Encoders", 
                    origin_file_pattern=siglip_path, 
                    **active_config
                ),
                ModelConfig(
                    model_id="DiffSynth-Studio/General-Image-Encoders", 
                    origin_file_pattern=dino_path, 
                    **active_config
                ),
                # Add i2L model
                ModelConfig(
                    model_id="DiffSynth-Studio/Z-Image-i2L", 
                    origin_file_pattern=z_image_file_path, 
                    **active_config
                )
            ]
            
            # Initialize Pipeline without loading unnecessary components (DiT, etc.)
            pipe = ZImagePipeline.from_pretrained(
                torch_dtype=torch.bfloat16,
                device=device_str, 
                model_configs=model_configs,
                # We don't need text encoder or tokenizer for i2L extraction
                tokenizer_config=None 
            )
            
            print("✅ Pipeline Loaded Successfully\n")
            
            # Apply CPU wrappers if needed
            if use_cpu_for_vision:
                print("🛡️ Applying CPU wrappers to vision encoders...")
                if hasattr(pipe, 'siglip2_image_encoder') and pipe.siglip2_image_encoder is not None:
                    pipe.siglip2_image_encoder = ForceCPUWrapper(pipe.siglip2_image_encoder)
                if hasattr(pipe, 'dinov3_image_encoder') and pipe.dinov3_image_encoder is not None:
                    pipe.dinov3_image_encoder = ForceCPUWrapper(pipe.dinov3_image_encoder)
                print("✅ CPU wrappers applied\n")

            print("🔍 Analyzing images...")
            with torch.no_grad():
                # Encode images
                # Z-Image uses specific units for this
                embs = ZImageUnit_Image2LoRAEncode().process(
                    pipe, 
                    image2lora_images=pil_images
                )
                print("✅ Images encoded\n")
                
                # Ensure tensors are on correct device
                forced_embs = {}
                for k, v in embs.items():
                    if isinstance(v, torch.Tensor):
                        forced_embs[k] = v.to(device_str)
                    else:
                        forced_embs[k] = v

                # Decode to LoRA weights
                print("⚙️ Generating LoRA weights...")
                result = ZImageUnit_Image2LoRADecode().process(pipe, **forced_embs)
                lora_weights = result["lora"]
                print("✅ LoRA weights generated\n")

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
                print("💡 SUGGESTION: You might be running out of VRAM or have a device mismatch.")
                print("👉 Try enabling 'Force CPU for Vision Encoders' in the node settings.")
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
