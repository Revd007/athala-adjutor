import torch
import sys # Tambahkan ini untuk melihat versi Python

print(f"Python version: {sys.version}") # Tambahkan output versi Python
print(f"PyTorch version: {torch.__version__}")
print("--- Checking CUDA Availability ---") # Tambahkan pemisah
cuda_available = torch.cuda.is_available()
print(f"CUDA available: {cuda_available}")

if cuda_available:
    print(f"CUDA version used by PyTorch: {torch.version.cuda}")
    print(f"Number of GPUs available: {torch.cuda.device_count()}")
    try:
         print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    except Exception as e:
         print(f"Could not get GPU name: {e}")
else:
    print("-----")
    print("CUDA IS NOT AVAILABLE FOR PYTORCH!")
    print("Possible reasons listed in previous message.")
    print("-----")

print("--- Script Finished ---") # Tambahkan penanda akhir