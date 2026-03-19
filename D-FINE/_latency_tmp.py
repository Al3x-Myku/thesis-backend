
import onnxruntime as ort
import numpy as np
import time

onnx_path = "/home/al3xmyku/thesis-backend/D-FINE/weights/dfine_l.onnx"
providers = ['CUDAExecutionProvider']

try:
    session = ort.InferenceSession(onnx_path, providers=providers)
except Exception as e:
    print(f"ONNX Load Failed: {e}")
    import sys; sys.exit(1)

# D-FINE requires two specific inputs
input_feed = {}
for inp in session.get_inputs():
    if inp.name == "orig_target_sizes":
        input_feed[inp.name] = np.array([[640, 640]], dtype=np.int64)
    else:
        input_feed[inp.name] = np.random.randn(1, 3, 640, 640).astype(np.float32)

print("[INFO] Warming up GPU...")
for _ in range(20): session.run(None, input_feed)

iters = 200
print(f"[INFO] Benchmarking {iters} iterations...")
start_time = time.time()
for _ in range(iters): session.run(None, input_feed)
avg_ms = ((time.time() - start_time) / iters) * 1000

print(f"RESULTS_LATENCY:{avg_ms:.2f}")
