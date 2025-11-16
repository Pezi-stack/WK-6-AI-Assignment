# data_loader.py

import tensorflow as tf
import numpy as np
import os

class RecyclingDataset:

    def __init__(self, img_size=(128, 128)):

        self.img_size = img_size

        self.class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

        

    def create_synthetic_dataset(self, num_samples=2000):

        """Create synthetic dataset for demonstration"""

        print("Creating synthetic recycling dataset...")

        

        # In real scenario, you would use:

        # dataset = tfds.load('trashnet', split='train')

        # For now, we'll create synthetic data

        

        # Create directory structure

        os.makedirs('synthetic_dataset', exist_ok=True)

        for class_name in self.class_names:

            os.makedirs(f'synthetic_dataset/{class_name}', exist_ok=True)

        

        return self.class_names

    

    def load_and_preprocess_image(self, image_path):

        """Load and preprocess image for model"""

        image = tf.io.read_file(image_path)

        image = tf.image.decode_image(image, channels=3)

        image = tf.image.resize(image, self.img_size)

        image = tf.cast(image, tf.float32) / 255.0

        return image

    

    def create_tf_dataset(self, batch_size=32):

        """Create TensorFlow dataset pipeline"""

        # This would be replaced with actual dataset loading

        # For demonstration, creating synthetic dataset structure

        

        # Generate synthetic data

        x_train = np.random.random((1000, *self.img_size, 3)).astype(np.float32)

        y_train = np.random.randint(0, len(self.class_names), 1000)

        

        x_val = np.random.random((200, *self.img_size, 3)).astype(np.float32)

        y_val = np.random.randint(0, len(self.class_names), 200)

        

        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))

        val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))

        

        train_dataset = train_dataset.shuffle(1000).batch(batch_size)

        val_dataset = val_dataset.batch(batch_size)

        

        return train_dataset, val_dataset

