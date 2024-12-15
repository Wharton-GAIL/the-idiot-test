from modelManager import ModelManager
import tensorflow as tf
import os
import psutil
from time import sleep
import gc
gc.collect()

process = psutil.Process(os.getpid())
print(f"Memory usage: {process.memory_info().rss / 1024**2:.2f} MB")

model_manager = ModelManager("https://tfhub.dev/google/universal-sentence-encoder/4")

ideas = ["test idea 1", "test idea 2", "test idea 3", "test idea 4", "test idea 5"]
vectors = model_manager.embed(ideas)
print(vectors)
print(f"Memory usage: {process.memory_info().rss / 1024**2:.2f} MB")
model_manager.free_model()
sleep(5)
print(f"Memory usage: {process.memory_info().rss / 1024**2:.2f} MB")
tf.keras.backend.clear_session()
sleep(5)
print(f"Memory usage: {process.memory_info().rss / 1024**2:.2f} MB")
gc.collect()
sleep(5)
print(f"Memory usage: {process.memory_info().rss / 1024**2:.2f} MB")

vectors = model_manager.embed(ideas)
print(vectors)
print(f"Memory usage: {process.memory_info().rss / 1024**2:.2f} MB")