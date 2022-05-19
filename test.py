import numpy as np
import pandas as pd 
import tensorflow as tf
import tensorflow_hub as hub 
import tensorflow_text as text 

from config import config
from sklearn.preprocessing import LabelBinarizer


physical_devices = tf.config.list_physical_devices('GPU')

try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass


original_df = pd.read_csv('Data/original_dataset.csv')
labels = pd.unique(original_df['intent'])
num_classes = len(labels)

for i in range(len(labels)): 
    labels[i] = str(labels[i])
    
binarizer = LabelBinarizer()
binarizer.fit_transform(labels)

tfhub_handle_encoder = config['map_name_to_handle'][config['bert_model_name']]
tfhub_handle_preprocess = config['map_model_to_preprocess'][config['bert_model_name']]

def predict(examples, model):
    results = tf.nn.softmax(model(tf.constant(examples)))
    intents = binarizer.inverse_transform(results.numpy())

    return intents

def print_my_examples(inputs, results):
    result_for_printing = [f'input: {inputs[i]:<30} : estimated intent: {results[i]}' for i in range(len(inputs))]
    print(*result_for_printing, sep='\n')

def build_classifier_model(num_classes):
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing')
    encoder_inputs = preprocessing_layer(text_input)
    encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=False, name='BERT_encoder')
    outputs = encoder(encoder_inputs)

    net = outputs['pooled_output']
    net = tf.keras.layers.Dropout(0.1)(net)
    net = tf.keras.layers.Dense(num_classes, activation=None, name='classifier')(net)

    return tf.keras.Model(text_input, net)

if __name__ == '__main__':  
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    metrics = tf.metrics.CategoricalAccuracy()
    optimizer = tf.keras.optimizers.Adam(1e-5)

    classifier_model = build_classifier_model(num_classes)
    classifier_model.summary()
    classifier_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    classifier_model.load_weights('models/intent_classifier.h5')

    examples = [
        'no i do not have it',
        'We can do it on June 4th',
        'Thanks I appreciate it',
        'Can we do it tomorrow?',
        'I do not understand', 
        '123421421',
        'Here you go - ',
        "no I don't",
        'I cannot do that',
        'Can we schedule it for tomorrow or day after?',
        "It was on April 20th last year",
        "Nope"
    ]

    intents = predict(examples, classifier_model)

    print_my_examples(examples, intents)