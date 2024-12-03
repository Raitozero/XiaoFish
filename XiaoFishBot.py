import tensorflow as tf




from tensorflow.keras import layers
import numpy as np


# 定义位置编码函数
def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(
        np.arange(position)[:, np.newaxis],
        np.arange(d_model)[np.newaxis, :],
        d_model)

    # 将 sin 应用于数组中的偶数索引（2i）
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    # 将 cos 应用于数组中的奇数索引（2i+1）
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)


# 定义多头注意力机制
class XFMultiHeadAttention(layers.Layer):
    def __init__(self, d_model, num_heads):
        super(XFMultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % num_heads == 0

        self.depth = d_model // num_heads

        self.wq = layers.Dense(d_model)
        self.wk = layers.Dense(d_model)
        self.wv = layers.Dense(d_model)

        self.dense = layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """分拆最后一个维度到 (num_heads, depth)。"""
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])  # (batch_size, num_heads, seq_len, depth)

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        scaled_attention_logits = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
        dk = tf.cast(self.depth, tf.float32)
        scaled_attention_logits /= tf.math.sqrt(dk)

        if mask is not None:
            scaled_attention_logits += (mask * -1e9)

        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)
        output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth)

        output = tf.transpose(output, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)
        concat_attention = tf.reshape(output, (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)
        return output


# 定义编码器层
class EncoderLayer(layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = XFMultiHeadAttention(d_model, num_heads)
        self.ffn = tf.keras.Sequential([
            layers.Dense(dff, activation='relu'),
            layers.Dense(d_model)
        ])

        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, x, training, mask):
        attn_output = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # 残差连接

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # 残差连接

        return out2


# 定义编码器
class Encoder(layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff,
                 input_vocab_size, maximum_position_encoding, rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dff = dff

        self.embedding = layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]
        self.dropout = layers.Dropout(rate)

    def call(self, x, training, mask=None):  # 添加 mask 参数
        seq_len = tf.shape(x)[1]

        # 添加嵌入和位置编码
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        # 逐层调用 EncoderLayer
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training=training, mask=mask)

        return x  # (batch_size, input_seq_len, d_model)



# 定义解码器层
class DecoderLayer(layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = XFMultiHeadAttention(d_model, num_heads)
        self.mha2 = XFMultiHeadAttention(d_model, num_heads)

        self.ffn = tf.keras.Sequential([
            layers.Dense(dff, activation='relu'),
            layers.Dense(d_model)
        ])

        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
        self.dropout3 = layers.Dropout(rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        attn1 = self.mha1(x, x, x, mask=look_ahead_mask)  # 修正关键字参数
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2 = self.mha2(enc_output, enc_output, out1, mask=padding_mask)  # 修正关键字参数
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)

        ffn_output = self.ffn(out2)  # 前馈网络
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)

        return out3


# 定义解码器
class Decoder(layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff,
                 target_vocab_size, maximum_position_encoding, rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)

        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]
        self.dropout = layers.Dropout(rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        seq_len = tf.shape(x)[1]

        attention_weights = {}

        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.dec_layers[i](x, enc_output=enc_output, training=training,
                                   look_ahead_mask=look_ahead_mask, padding_mask=padding_mask)

        return x  # (batch_size, target_seq_len, d_model)


# 定义 Transformer 模型
class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff,
                 input_vocab_size, target_vocab_size,
                 pe_input, pe_target, rate=0.1):
        super(Transformer, self).__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dff = dff
        self.input_vocab_size = input_vocab_size
        self.target_vocab_size = target_vocab_size
        self.pe_input = pe_input
        self.pe_target = pe_target
        self.rate = rate

        self.encoder = Encoder(num_layers, d_model, num_heads, dff,
                               input_vocab_size, pe_input, rate)

        self.decoder = Decoder(num_layers, d_model, num_heads, dff,
                               target_vocab_size, pe_target, rate)

        self.final_layer = layers.Dense(target_vocab_size)
    def call(self, inp, tar, training, enc_padding_mask, look_ahead_mask, dec_padding_mask):
        enc_output = self.encoder(inp, training=training, mask=enc_padding_mask)  # 传递 enc_padding_mask

        dec_output = self.decoder(
            tar, enc_output=enc_output, training=training,
            look_ahead_mask=look_ahead_mask, padding_mask=dec_padding_mask
        )  # 传递 look_ahead_mask 和 dec_padding_mask

        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

        return final_output

    def get_config(self):
        # 保存模型的配置
        return {
            "num_layers": self.num_layers,
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "dff": self.dff,
            "input_vocab_size": self.input_vocab_size,
            "target_vocab_size": self.target_vocab_size,
            "pe_input": self.pe_input,
            "pe_target": self.pe_target,
            "rate": self.rate
        }

    @classmethod
    def from_config(cls, config):
        # 从配置中重建模型
        return cls(**config)


# 定义自定义模型
class CustomTransformer(tf.keras.Model):
    def __init__(self, transformer):
        super(CustomTransformer, self).__init__()
        self.transformer = transformer

    def call(self, inputs, training=False):
        inp, tar = inputs  # 确保 inputs 是 (inp, tar)

        enc_padding_mask, look_ahead_mask, dec_padding_mask = self.create_masks(inp, tar)

        final_output = self.transformer(
            inp, tar, training=training, enc_padding_mask=enc_padding_mask, look_ahead_mask=look_ahead_mask, dec_padding_mask=dec_padding_mask
        )
        return final_output

    def create_padding_mask(self, seq):
        seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
        return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

    def create_look_ahead_mask(self, size):
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask  # (seq_len, seq_len)

    def create_masks(self, inp, tar):
        # 编码器填充遮罩
        enc_padding_mask = self.create_padding_mask(inp)

        # 在解码器的第二个注意力模块使用的填充遮罩
        dec_padding_mask = self.create_padding_mask(inp)

        # 在解码器的第一个注意力模块使用的前瞻遮罩
        look_ahead_mask = self.create_look_ahead_mask(tf.shape(tar)[1])
        dec_target_padding_mask = self.create_padding_mask(tar)
        combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

        return enc_padding_mask, combined_mask, dec_padding_mask

    def get_config(self):
        # 保存 CustomTransformer 的配置
        config = super(CustomTransformer, self).get_config()
        config.update({"transformer": self.transformer.get_config()})
        return config

    @classmethod
    def from_config(cls, config):
        # 从配置中重建 CustomTransformer
        transformer_config = config.pop("transformer")
        transformer = Transformer.from_config(transformer_config)
        return cls(transformer=transformer)

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps**-1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

    def get_config(self):
        # 返回类的参数，以便序列化时保存配置
        return {
            "d_model": self.d_model.numpy(),  # 将 Tensor 转为普通 Python 数值
            "warmup_steps": self.warmup_steps
        }

    @classmethod
    def from_config(cls, config):
        # 从配置中恢复实例
        return cls(**config)


# 超参数设置
if __name__ == '__main__':
    num_layers = 4
    d_model = 128
    dff = 512
    num_heads = 8

    input_vocab_size = 8500
    target_vocab_size = 8000
    dropout_rate = 0.1

    # 创建 Transformer 模型
    transformer = Transformer(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        dff=dff,
        input_vocab_size=input_vocab_size,
        target_vocab_size=target_vocab_size,
        pe_input=1000,
        pe_target=1000,
        rate=dropout_rate
    )

    # 创建自定义模型
    custom_transformer = CustomTransformer(transformer)

    # 编译模型
    custom_transformer.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    # 准备示例数据
    batch_size = 64
    sequence_length = 40  # 序列长度
    num_samples = 1000  # 样本数量

    # 生成随机的输入和目标数据
    inp_data = np.random.randint(0, input_vocab_size, size=(num_samples, sequence_length))
    tar_data = np.random.randint(0, target_vocab_size, size=(num_samples, sequence_length))

    # 创建数据集
    dataset = tf.data.Dataset.from_tensor_slices(((inp_data, tar_data), tar_data))
    dataset = dataset.shuffle(1000).batch(batch_size)

    # 训练模型
    history = custom_transformer.fit(
        dataset,
        epochs=10
    )