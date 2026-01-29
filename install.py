import subprocess
import sys

def install_package(package_name):
    print(f"Installing {package_name}...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        print(f"Successfully installed {package_name}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to install {package_name}: {e}")

if __name__ == "__main__":
    print("Installing dependencies for Z-Image-i2L...")
    
    # Install core dependencies
    requirements = [
        "huggingface_hub",
        "safetensors",
        "modelscope",
        "git+https://github.com/modelscope/DiffSynth-Studio.git"
    ]
    
    for req in requirements:
        install_package(req)
    
    print("\nInstallation complete. Please restart ComfyUI.")
