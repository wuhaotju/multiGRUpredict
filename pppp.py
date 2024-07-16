import jieba
import pandas as pd
import numpy as np
import tensorflow as tf
data = pd.read_csv('./label.csv')
text = pd.read_csv('./text1.csv')
tx_v = text['question_text']
from sklearn.model_selection import train_test_split


from keras.models import Sequential
from keras import layers

from gensim.models import Word2Vec

td = [list(jieba.cut(s)) for s in data['question_text']]
# 设置 Word2Vec 模型的参数
vector_size = 50  # 设置词向量的维度
window = 5  # 窗口大小，控制上下文窗口的大小
min_count = 1  # 最小词频，过滤掉低频词
sg = 1  # 0表示使用 CBOW 模型，1示使用 Skip-Gram 模型

print(td)
wv_model = Word2Vec(td, vector_size=vector_size, window=window, min_count=min_count, sg=sg)


def text_to_vector(text):
    vector = np.zeros(50)  # 初始化全零向量
    count = 0  # 统计有效词汇数
    for word in text:
        if word in wv_model.wv:
            vector += wv_model.wv[word]
            count += 1
    if count != 0:
        vector /= count
    return vector

print('aaaaaaaaaaaa')
X = [text_to_vector(t) for t in td]
train_x, val_x, train_y, val_y = train_test_split(X, data[['1', '2']], test_size=0.2, random_state=2019)
tx = [list(jieba.cut(s)) for s in tx_v]
tx_a = [text_to_vector(t) for t in tx]

train_x = tf.stack(train_x)
train_y = tf.stack(train_y)

val_x = tf.stack(val_x)
val_y = tf.stack(val_y)

print(np.array(train_x).shape)
print(np.array(train_y).shape)
vocab_size = 50

embedding_dim = 50

model = Sequential()
model.add(layers.Embedding(input_dim=vocab_size,
                           output_dim=embedding_dim
                           ))
# model.add(layers.GlobalAvgPool3D())
model.add(layers.GRU(units=50))
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dense(2, activation='softmax'))
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


history = model.fit(train_x, train_y,
                    epochs=50,
                    verbose=False,
                    validation_data=(val_x, val_y),
                    batch_size=10)
model.summary()
loss, accuracy = model.evaluate(val_x, val_y, verbose=False)
tx_a = tf.stack(tx_a)
a = model.predict(tx_a)
print("AAAAAAAAAAAAAAAAAAAAAa")
print(a)

print('Test accuracy: {:.4f}'.format(accuracy))
