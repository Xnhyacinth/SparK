import subprocess
import re
import torch

def supports_flash_attention():
    try:
        if not torch.cuda.is_available():
            return False, "CUDA not available"
        
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,compute_cap', '--format=csv'], 
                               capture_output=True, text=True, check=True)
        output = result.stdout
        
        supported_gpus = []
        unsupported_gpus = []
        
        lines = output.strip().split('\n')[1:]  # 跳过标题行
        for line in lines:
            if line.strip():
                parts = [part.strip() for part in line.split(',')]
                if len(parts) >= 2:
                    gpu_name = parts[0]
                    compute_cap = parts[1]
                    
                    try:
                        major, minor = map(int, compute_cap.split('.'))
                        compute_capability = major * 10 + minor
                        
                        if compute_capability >= 80:
                            supported_gpus.append({
                                'name': gpu_name,
                                'compute_cap': compute_cap,
                                'architecture': get_gpu_architecture(compute_capability)
                            })
                        else:
                            unsupported_gpus.append({
                                'name': gpu_name,
                                'compute_cap': compute_cap,
                                'architecture': get_gpu_architecture(compute_capability)
                            })
                    except ValueError:
                        unsupported_gpus.append({
                            'name': gpu_name,
                            'compute_cap': compute_cap,
                            'architecture': 'Unknown'
                        })
        
        if supported_gpus:
            return True, supported_gpus
        else:
            return False, {
                'supported': supported_gpus,
                'unsupported': unsupported_gpus,
                'message': "No GPUs with compute capability >= 8.0 found"
            }

    except subprocess.CalledProcessError as e:
        return False, f"nvidia-smi command failed: {e}"
    except FileNotFoundError:
        return False, "nvidia-smi not found. Ensure NVIDIA drivers are installed."
    except Exception as e:
        return False, f"Error checking GPU capabilities: {str(e)}"


def get_gpu_architecture(compute_capability):
    arch_map = {
        30: "Kepler",
        35: "Kepler",
        37: "Kepler",
        50: "Maxwell",
        52: "Maxwell",
        53: "Maxwell",
        60: "Pascal",
        61: "Pascal",
        62: "Pascal",
        70: "Volta",
        72: "Volta", 
        75: "Turing",
        80: "Ampere",
        86: "Ampere",
        87: "Ampere",
        89: "Ada Lovelace",
        90: "Hopper"
    }
    
    return arch_map.get(compute_capability, f"Unknown (CC {compute_capability/10:.1f})")
# is_ampere, info = is_ampere_gpu()
# if is_ampere:
#     print(f"Ampere GPU detected: {info}")
# else:
#     print(f"No Ampere GPU detected: {info}")