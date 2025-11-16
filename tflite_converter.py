# tflite_converter.py

import tensorflow as tf
import numpy as np

class TFLiteConverter:

    def __init__(self, model):

        self.model = model

        

    def convert_to_tflite(self, optimization='DEFAULT'):

        """Convert model to TensorFlow Lite with optimization"""

        

        # Basic conversion

        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)

        

        # Apply optimizations

        if optimization == 'DEFAULT':

            converter.optimizations = [tf.lite.Optimize.DEFAULT]

        elif optimization == 'FLOAT16':

            converter.optimizations = [tf.lite.Optimize.DEFAULT]

            converter.target_spec.supported_types = [tf.float16]

        elif optimization == 'INT8':

            # For INT8, we need a representative dataset

            converter.optimizations = [tf.lite.Optimize.DEFAULT]

            converter.representative_dataset = self._representative_dataset

            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

            converter.inference_input_type = tf.uint8

            converter.inference_output_type = tf.uint8

        

        tflite_model = converter.convert()

        

        return tflite_model

    

    def _representative_dataset(self):

        """Representative dataset for quantization"""

        for _ in range(100):

            data = np.random.rand(1, 128, 128, 3).astype(np.float32)

            yield [data]

    

    def save_tflite_model(self, tflite_model, filename):

        """Save TFLite model to file"""

        with open(filename, 'wb') as f:

            f.write(tflite_model)

        

        file_size = len(tflite_model) / 1024  # Size in KB

        print(f"TFLite model saved: {filename}")

        print(f"Model size: {file_size:.2f} KB")

        

        return file_size

    

    def test_tflite_model(self, tflite_model, test_dataset, class_names):

        """Test TFLite model performance"""

        # Initialize interpreter

        interpreter = tf.lite.Interpreter(model_content=tflite_model)

        interpreter.allocate_tensors()

        

        # Get input and output details

        input_details = interpreter.get_input_details()

        output_details = interpreter.get_output_details()

        

        print("\nTFLite Model Details:")

        print(f"Input shape: {input_details[0]['shape']}")

        print(f"Input type: {input_details[0]['dtype']}")

        print(f"Output shape: {output_details[0]['shape']}")

        print(f"Output type: {output_details[0]['dtype']}")

        

        # Test inference

        correct_predictions = 0

        total_samples = 0

        

        for images, labels in test_dataset.take(10):

            for i in range(len(images)):

                # Preprocess input

                input_data = images[i].numpy().reshape(1, 128, 128, 3).astype(np.float32)

                

                # Set input tensor

                interpreter.set_tensor(input_details[0]['index'], input_data)

                

                # Run inference

                interpreter.invoke()

                

                # Get output

                output_data = interpreter.get_tensor(output_details[0]['index'])

                predicted_class = np.argmax(output_data[0])

                true_class = labels[i].numpy()

                

                if predicted_class == true_class:

                    correct_predictions += 1

                total_samples += 1

        

        accuracy = correct_predictions / total_samples if total_samples > 0 else 0

        print(f"TFLite Test Accuracy: {accuracy:.3f}")

        

        return accuracy

