
import onnxruntime as ort
import numpy as np
import time
import pynvml

pynvml.nvmlInit()
gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)

onnx_path = "/home/al3xmyku/thesis-backend/D-FINE/weights/dfine_l.onnx"
providers_fp32 = ['CUDAExecutionProvider']
providers_fp16 = [('TensorrtExecutionProvider', {'trt_fp16_enable': True, 'trt_engine_cache_enable': True, 'trt_engine_cache_path': './trt_cache'}), 'CUDAExecutionProvider']

def measure_latency(providers, name):
    sess = ort.InferenceSession(onnx_path, providers=providers)
    feed = {}
    for inp in sess.get_inputs():
        feed[inp.name] = np.array([[640, 640]], dtype=np.int64) if "orig" in inp.name else np.random.randn(1, 3, 640, 640).astype(np.float32)
    
    print(f"    [-->] {name}: 100 Warmup iterations...")
    for _ in range(100): sess.run(None, feed)
    
    print(f"    [-->] {name}: 2000 Benchmark iterations...")
    start = time.time()
    for _ in range(2000): sess.run(None, feed)
    total_time = time.time() - start
    
    print(f"RESULT_LAT_{name}:{(total_time/2000)*1000:.2f}")

def measure_throughput():
    sess = ort.InferenceSession(onnx_path, providers=providers_fp32)
    batch_sizes = [1, 4, 8, 16, 32, 64, 128]
    
    for b in batch_sizes:
        try:
            feed = {}
            for inp in sess.get_inputs():
                feed[inp.name] = np.tile(np.array([[640, 640]], dtype=np.int64), (b, 1)) if "orig" in inp.name else np.random.randn(b, 3, 640, 640).astype(np.float32)
            
            print(f"    [-->] Batch {b}: 50 Warmup iterations...")
            for _ in range(50): sess.run(None, feed)
            
            print(f"    [-->] Batch {b}: 500 Benchmark iterations (Processing {b * 500} images)...")
            start = time.time()
            for _ in range(500): sess.run(None, feed)
            total_time = time.time() - start
            
            fps = (b * 500) / total_time
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
            vram_gb = mem_info.used / (1024**3)
            
            print(f"RESULT_FPS_B{b}:{fps:.1f}")
            print(f"RESULT_VRAM_B{b}:{vram_gb:.2f}")
        except Exception as e:
            print(f"    [!] Batch {b} caused OOM or failed. Ceiling reached! Stopping throughput tests.")
            break

print("[INFO] Measuring FP32 Latency (Huge Benchmark)...")
measure_latency(providers_fp32, "FP32")

print("[INFO] Compiling/Loading TensorRT FP16 (May take 3-5 mins)...")
measure_latency(providers_fp16, "FP16")

print("[INFO] Stress-Testing Batched Throughput and VRAM to ceiling...")
measure_throughput()
