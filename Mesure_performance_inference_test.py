import numpy as np
import tflite_runtime.interpreter as tflite
import time
import psutil
from PIL import Image

def load_interpreter(model_path):
    interpreter = tflite.Interpreter(model_path=model_path,
                                     experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')])
    interpreter.allocate_tensors()
    return interpreter

def prepare_image(image_path, input_size):
    image = Image.open(image_path)
    image = image.resize(input_size, Image.ANTIALIAS)
    image = np.array(image, dtype=np.float32)
    # Normalize if required by the model
    image /= 255.0
    return image

def run_inference(interpreter, image):
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    interpreter.set_tensor(input_details['index'], image)
    
    memory_before = psutil.Process().memory_info().rss / (1024 * 1024)  # Convert to MB
    start_time = time.perf_counter()
    interpreter.invoke()
    end_time = time.perf_counter()
    memory_after = psutil.Process().memory_info().rss / (1024 * 1024)  # Convert to MB

    output_data = interpreter.get_tensor(output_details['index'])
    
    return output_data, (end_time - start_time), memory_before, memory_after

# Define model paths and corresponding input sizes
models = {
    'Face Detector': ('path_to_face_detector_model.tflite', (320, 320)),
    'Inception': ('path_to_inception_model.tflite', (224, 224)),
    'Bird Recognition': ('path_to_bird_recognition_model.tflite', (224, 224)),
    'Plant Recognition': ('path_to_plant_recognition_model.tflite', (224, 224)),
    'Object Detection': ('path_to_object_detection_model.tflite', (300, 300))
}

# Path to the test image
image_path = 'path_to_your_test_image.jpg'

for model_name, (model_path, input_size) in models.items():
    interpreter = load_interpreter(model_path)
    image = prepare_image(image_path, input_size)
    output, inference_time, memory_before, memory_after = run_inference(interpreter, image[np.newaxis, ...])

    print(f"Model: {model_name}")
    print(f"Inference time: {inference_time:.6f} seconds")
    print(f"Memory usage before inference: {memory_before:.2f} MB")
    print(f"Memory usage after inference: {memory_after:.2f} MB")
    print("-" * 50)
