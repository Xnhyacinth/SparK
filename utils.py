import subprocess
import re

def is_ampere_gpu():
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv'], 
                               capture_output=True, text=True)
        output = result.stdout

        gpu_names = [line.strip() for line in output.split('\n')[1:] if line.strip()]
        
        ampere_keywords = [
            'RTX 3060', 'RTX 3070', 'RTX 3080', 'RTX 3090', 
            'A100', 'A10', 'A30', 'A40', 'A6000'
        ]

        for gpu_name in gpu_names:
            if any(keyword in gpu_name for keyword in ampere_keywords):
                return True, gpu_name
        return False, gpu_names

    except FileNotFoundError:
        return False, "nvidia-smi not found. Ensure NVIDIA drivers are installed."
    except Exception as e:
        return False, f"Error: {str(e)}"

# is_ampere, info = is_ampere_gpu()
# if is_ampere:
#     print(f"Ampere GPU detected: {info}")
# else:
#     print(f"No Ampere GPU detected: {info}")