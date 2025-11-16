# benchmark.py

import time
import psutil
import platform
import tensorflow as tf
import numpy as np

class EdgeAIBenchmark:

    def __init__(self, tflite_model):

        self.tflite_model = tflite_model

        self.interpreter = tf.lite.Interpreter(model_content=tflite_model)

        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()

        self.output_details = self.interpreter.get_output_details()

    

    def benchmark_inference(self, num_iterations=100):

        """Benchmark inference speed"""

        print("Running inference benchmark...")

        

        # Generate test input

        input_shape = self.input_details[0]['shape']

        test_input = np.random.random(input_shape).astype(np.float32)

        

        # Warm-up

        for _ in range(10):

            self.interpreter.set_tensor(self.input_details[0]['index'], test_input)

            self.interpreter.invoke()

        

        # Benchmark

        times = []

        for _ in range(num_iterations):

            start_time = time.time()

            self.interpreter.set_tensor(self.input_details[0]['index'], test_input)

            self.interpreter.invoke()

            end_time = time.time()

            times.append((end_time - start_time) * 1000)  # Convert to ms

        

        avg_time = np.mean(times)

        std_time = np.std(times)

        fps = 1000 / avg_time if avg_time > 0 else 0

        

        print(f"Average inference time: {avg_time:.2f} ms")

        print(f"Standard deviation: {std_time:.2f} ms")

        print(f"Maximum FPS: {fps:.1f}")

        

        return {

            'avg_inference_time_ms': avg_time,

            'std_inference_time_ms': std_time,

            'max_fps': fps

        }

    

    def get_system_info(self):

        """Get system information for benchmarking"""

        system_info = {

            'platform': platform.system(),

            'processor': platform.processor(),

            'memory_gb': psutil.virtual_memory().total / (1024**3),

            'python_version': platform.python_version(),

            'tensorflow_version': tf.__version__

        }

        

        print("\nSystem Information:")

        for key, value in system_info.items():

            print(f"{key}: {value}")

        

        return system_info

    

    def memory_usage(self):

        """Check memory usage of the model"""

        process = psutil.Process()

        memory_mb = process.memory_info().rss / 1024 / 1024

        print(f"Current memory usage: {memory_mb:.2f} MB")

        return memory_mb

