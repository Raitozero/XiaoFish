import re
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate.bleu_score import sentence_bleu
from tqdm import tqdm

def predict_sequence_step_by_step(input_sentence, model, source_tokenizer, target_tokenizer, max_length=50,
                                  beam_width=5):
    # 1. 预处理输入句子
    input_sentence = preprocess_text(input_sentence)
    input_sequence = source_tokenizer.texts_to_sequences([input_sentence])
    input_sequence = pad_sequences(input_sequence, padding='post')

    # 2. 初始化解码器输入序列（以 <START> 开始）
    start_token = target_tokenizer.word_index.get('<start>', None)
    end_token = target_tokenizer.word_index.get('<end>', None)
    if start_token is None or end_token is None:
        raise ValueError("<START> or <END> token is missing in the target tokenizer.")

    target_sequence = [start_token]  # 初始目标序列

    # 3. 逐步解码
    for _ in range(max_length):
        tar_inp = pad_sequences([target_sequence], padding='post', maxlen=max_length)
        predictions = model.predict((input_sequence, tar_inp), verbose=0)
        predictions = tf.nn.softmax(predictions, axis=-1).numpy()
        next_token = np.argmax(predictions[0, len(target_sequence) - 1])  # 当前时间步的预测
        target_sequence.append(next_token)
        if next_token == end_token:
            break

    # 4. 将生成的序列转换为文本，移除 <start> 和 <end>
    decoded_sequence = " ".join(
        [target_tokenizer.index_word.get(idx, "<OOV>") for idx in target_sequence if idx > 0 and idx not in {start_token, end_token}]
    )
    return decoded_sequence

def sequence_to_text(sequence, tokenizer):
    return " ".join([tokenizer.index_word.get(idx, "<OOV>") for idx in sequence if idx > 0])

def preprocess_text(text):
    text = re.sub(r'[\t\n]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.lower()
    return text.strip()

# BLEU Warpper
def get_bleu(generated_commands, reference_commands):
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    import numpy as np

    start_token = '<start>'
    end_token = '<end>'

    def compute_bleu_score(reference, prediction):
        smoothie = SmoothingFunction().method1
        return sentence_bleu([reference], prediction, smoothing_function=smoothie)

    print("----------------BLEU EVALUATION START----------------")
    bleu_scores = []

    for pred_command, ref_command in zip(generated_commands, reference_commands):
        # Tokenize the predicted and reference commands
        pred_tokens = pred_command.strip().split()
        ref_tokens = ref_command.strip().split()

        # Remove <start> and <end> tokens from both predictions and references
        pred_tokens = [token for token in pred_tokens if token not in {start_token, end_token}]
        ref_tokens = [token for token in ref_tokens if token not in {start_token, end_token}]

        # Compute BLEU score
        bleu = compute_bleu_score(ref_tokens, pred_tokens)
        bleu_scores.append(bleu)

    average_bleu = np.mean(bleu_scores)
    print(f"Average BLEU score on test set: {average_bleu:.4f}")
    print("----------------BLEU EVALUATION END----------------")

