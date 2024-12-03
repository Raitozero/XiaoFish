import os
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from XiaoFishBot import Transformer, CustomTransformer, CustomSchedule
from utils import predict_sequence_step_by_step
from keras.models import save_model
# 定义保存的路径
model_dir = "model_2"
tokenizer_path = os.path.join(model_dir, "tokenizers")
saved_model_path = os.path.join(model_dir, "transformer_model")

# 加载模型
loaded_model = load_model(saved_model_path, custom_objects={
    "Transformer": Transformer,
    "CustomTransformer": CustomTransformer,
    "CustomSchedule": CustomSchedule
})

print("Model loaded successfully!")
loaded_model.save("saved_model_path", save_format="tf")
print("Model saved successfully!")