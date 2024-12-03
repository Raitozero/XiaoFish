import os
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from XiaoFishBot import Transformer, CustomTransformer, CustomSchedule
from utils import preprocess_text, sequence_to_text, predict_sequence_step_by_step, get_bleu
from functionality_test import test_functionality, get_accuracy
import numpy as np

# 加载模型
model_dir = "latest_model"
tokenizer_path = os.path.join(model_dir, "tokenizers")
saved_model_path = os.path.join(model_dir, "saved_model")
SEED = 42
np.random.seed(SEED)
print("--------------Start Loading Model--------------")
# 定义保存的路径load
# 加载模型
loaded_model = load_model(saved_model_path, custom_objects={
    "Transformer": Transformer,
    "CustomTransformer": CustomTransformer,
    "CustomSchedule": CustomSchedule
})
print("--------------Model Loaded Successfully--------------")

# 加载分词器
print("--------------Start Loading Tokenizer--------------")
with open(os.path.join(tokenizer_path, 'source_tokenizer.pickle'), 'rb') as handle:
    source_tokenizer = pickle.load(handle)

with open(os.path.join(tokenizer_path, 'target_tokenizer.pickle'), 'rb') as handle:
    target_tokenizer = pickle.load(handle)
print("--------------Tokenizer Loaded Successfully--------------")

# Test the generated commands running on Linux

with open(r'data/all.nl', 'r', encoding='utf-8') as f_nl:
    input_sentences = f_nl.readlines()
with open(r'data/all.cm', 'r', encoding='utf-8') as f_cm:
    reference_commands = f_cm.readlines()



# Generate details of first 10 samples
input_sentences_test1 = input_sentences[:10]
generated_commands_test1 = []
for input_sentence in input_sentences_test1:
    generated_command = predict_sequence_step_by_step(input_sentence, loaded_model, source_tokenizer, target_tokenizer)
    generated_commands_test1.append(generated_command)

reference_commands_test1 = reference_commands[:10]
results = test_functionality(input_sentences_test1, reference_commands_test1, generated_commands_test1)

for res in results:
    print(f"Input: {res['input']}")
    print(f"Reference: {res['reference']}")
    print(f"Generated: {res['generated']}")
    print(f"Correct: {res['correct']}\n")

# BLEU
get_bleu(generated_commands_test1, reference_commands_test1)


# Real Test
size = 50
input_sentences_test2 = input_sentences[:size]
generated_commands_test2 = []
for input_sentence in input_sentences_test2:
    generated_command = predict_sequence_step_by_step(input_sentence, loaded_model, source_tokenizer, target_tokenizer)
    generated_commands_test2.append(generated_command)

reference_commands_test1 = reference_commands[:size]
results = test_functionality(input_sentences_test2, reference_commands_test1, generated_commands_test2)
accuracy = get_accuracy(results)
print("Result: ")
print(f"Input Sentences Tested: {accuracy[0]}")
print(f"Passed: {accuracy[1]}")
print(f"Accuracy: {accuracy[1] / accuracy[0] * 100:.2f}%")


