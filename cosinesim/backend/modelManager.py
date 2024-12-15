import tensorflow_hub as hub
import tensorflow as tf
import time
import threading
import os
import psutil

class ModelManager:
    def __init__(self, model_url: str, idle_timeout: int = 300):
        self.model_url = model_url
        self.model = None
        self.last_used = None
        self.idle_timeout = idle_timeout  # 5 minutes in seconds
        self.lock = threading.Lock()
        self._start_cleanup_thread()
    
    def _start_cleanup_thread(self):
        def cleanup_loop():
            while True:
                time.sleep(60)  # Check every minute
                self._cleanup_if_idle()
        
        thread = threading.Thread(target=cleanup_loop, daemon=True)
        thread.start()
    
    def _cleanup_if_idle(self):
        with self.lock:
            if (self.model is not None and 
                self.last_used is not None and 
                time.time() - self.last_used > self.idle_timeout):
                self.model = None
                tf.keras.backend.clear_session()
                print("Model unloaded due to inactivity")
    
    def get_model(self):
        with self.lock:
            if self.model is None:
                print("Loading model...")
                self.model = hub.load(self.model_url)
                print("Model loaded")
            self.last_used = time.time()
            return self.model

    def free_model(self):
        with self.lock:
            if self.model is not None:
                del self.model
                self.model = None
                print("Model unloaded")

    def embed(self, texts):
        model = self.get_model()
        return model(texts)