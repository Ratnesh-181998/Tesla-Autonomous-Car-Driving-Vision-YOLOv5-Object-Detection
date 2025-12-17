import time
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("yolo_int8.onnx", providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name

dummy = np.random.rand(1,3,640,640).astype(np.float32)

start = time.time()
for _ in range(100):
    session.run(None, {input_name: dummy})
end = time.time()

print("Avg latency (ms):", (end-start)/100*1000)
