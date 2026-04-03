import os
import subprocess

# The Hardware Rip List
BENCHMARKS = [
    "benchmarks/bench_relu.py",
    "benchmarks/bench_layernorm.py",
    "benchmarks/bench_layernorm_fp8.py", # Added the FP8 Flex
    "benchmarks/bench_flash_attn.py"
]

def run_rip():
    print("🚀 INITIALIZING H100 HARDWARE RIP...")
    
    # Check if we are actually on a GPU
    import torch
    if not torch.cuda.is_available():
        print("❌ CRITICAL ERROR: No GPU detected. Benchmarking hardware on CPU is invalid.")
        return

    print(f"🔥 Found GPU: {torch.cuda.get_device_name(0)}")
    print("🛠️ Setting Persistence Mode...")
    # This keeps the clock speeds high so the data is consistent
    subprocess.run(["sudo", "nvidia-smi", "-pm", "1"], capture_output=True)

    for script in BENCHMARKS:
        print(f"\n📊 STARTING: {script}")
        try:
            subprocess.run(["python3", script], check=True)
            print(f"✅ Data captured for {script}")
        except Exception as e:
            print(f"⚠️ Failed to rip {script}: {e}")

    print("\n🏁 MISSION COMPLETE.")
    print("Check root for PNG/CSV files. Use 'git add *.png' to save your proof!")

if __name__ == "__main__":
    run_rip()