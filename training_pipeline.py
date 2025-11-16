# training_pipeline.py

import tensorflow as tf
import numpy as np

class RecyclingTrainer:

    def __init__(self, model, model_name="recycling_classifier"):

        self.model = model

        self.model_name = model_name

        self.history = None

        

    def compile_model(self, learning_rate=0.001):

        """Compile the model with appropriate settings"""

        self.model.compile(

            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),

            loss='sparse_categorical_crossentropy',

            metrics=['accuracy']

        )

        

        print("Model compiled successfully!")

        self.model.summary()

    

    def train(self, train_dataset, val_dataset, epochs=50):

        """Train the model with callbacks"""

        callbacks = [

            tf.keras.callbacks.EarlyStopping(

                monitor='val_loss',

                patience=10,

                restore_best_weights=True

            ),

            tf.keras.callbacks.ReduceLROnPlateau(

                monitor='val_loss',

                factor=0.2,

                patience=5,

                min_lr=1e-7

            ),

            tf.keras.callbacks.ModelCheckpoint(

                f'best_{self.model_name}.h5',

                monitor='val_accuracy',

                save_best_only=True,

                mode='max'

            )

        ]

        

        print("Starting training...")

        self.history = self.model.fit(

            train_dataset,

            epochs=epochs,

            validation_data=val_dataset,

            callbacks=callbacks,

            verbose=1

        )

        

        return self.history

    

    def evaluate(self, test_dataset):

        """Evaluate model performance"""

        print("Evaluating model...")

        test_loss, test_accuracy = self.model.evaluate(test_dataset, verbose=0)

        

        # Generate predictions

        y_pred = []

        y_true = []

        

        for images, labels in test_dataset:

            preds = self.model.predict(images, verbose=0)

            y_pred.extend(np.argmax(preds, axis=1))

            y_true.extend(labels.numpy())

        

        return {

            'test_loss': test_loss,

            'test_accuracy': test_accuracy,

            'y_true': y_true,

            'y_pred': y_pred

        }

