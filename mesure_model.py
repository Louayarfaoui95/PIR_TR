import time
import psutil
from pycoral.adapters import classify, detect
from pycoral.utils import edgetpu
from PIL import Image
import numpy as np

# Define the path for the model and the image
model_path = "face-detector-quantized_edgetpu.tflite"
image_path = "face_image.jpg"

def load_model(model_path):
    """Loads a TFLite model into the Edge TPU and measures time and memory usage."""
    start_time = time.time()
    interpreter = edgetpu.make_interpreter(model_path)
    interpreter.allocate_tensors()
    load_time = time.time() - start_time
    current_process = psutil.Process(os.getpid())
    memory_info = current_process.memory_info()
    return interpreter, load_time, memory_info.rss

def run_inference(interpreter, image_path):
    """Runs inference using the loaded model and measures inference time and memory usage."""
    image = Image.open(image_path)
    _, height, width, _ = interpreter.get_input_details()[0]['shape']
    image = image.resize((width, height))
    start_time = time.time()
    interpreter.set_tensor(interpreter.get_input_details()[0]['index'], [np.array(image).flatten()])
    interpreter.invoke()
    inference_time = time.time() - start_time
    current_process = psutil.Process(os.getpid())
    memory_info = current_process.memory_info()
    return inference_time, memory_info.rss

# Load model and run inference
interpreter, load_time, load_memory = load_model(model_path)
inference_time, inference_memory = run_inference(interpreter, image_path)

# Display performance metrics
print(f"Load Time: {load_time} seconds")
print(f"Memory Usage on Load: {load_memory} bytes")
print(f"Inference Time: {inference_time} seconds")
print(f"Memory Usage on Inference: {inference_memory} bytes")
