import warnings
from numpy import dtype

import pandas as pd 
import matplotlib.pyplot as plt 
import tensorflow as tf
import tensorflow_hub as hub 
import tensorflow_text as text 
import seaborn as sns

from config import config
from sklearn.preprocessing import LabelBinarizer
from pylab import rcParams

tf.get_logger().setLevel("ERROR")
warnings.filterwarnings('ignore')

physical_devices = tf.config.list_physical_devices('GPU')

try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass

binarizer = LabelBinarizer()
HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]
# rcParams['figure.figsize'] = 12,8

sns.set(style='whitegrid', palette='muted', font_scale=1.2)
sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))


def build_classifier_model(num_classes):
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')

    preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing')

    encoder_inputs = preprocessing_layer(text_input)

    encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name='BERT_encoder')

    outputs = encoder(encoder_inputs)

    net = outputs['pooled_output']
    net = tf.keras.layers.Dropout(0.1)(net)
    net = tf.keras.layers.Dense(num_classes, activation=None, name='classifier')(net)

    return tf.keras.Model(text_input, net)

def print_my_examples(inputs, results):
    result_for_printing = [f'input: {inputs[i]:<30} : estimated intent: {results[i]}' for i in range(len(inputs))]
    print(*result_for_printing, sep='\n')

def predict(examples): 
    results = tf.nn.softmax(classifier_model(tf.constant(examples)))
    intents = binarizer.inverse_transform(results.numpy())
    return intents


if __name__ == '__main__': 
    original_df = pd.read_csv('Data/original_dataset.csv')
    # train_df = pd.read_csv(config['train_file'])
    # valid_df = pd.read_csv(config['valid_file'])
    # test_df = pd.read_csv(config['test_file'])

    num_classes = len(pd.unique(original_df['intent']))

    print("no of classes in original set: ", num_classes)

    print(original_df.head())
    print(original_df.shape)

    # get all the training features (will include both text and intent-label)
    train_features = original_df.copy()
    train_labels = train_features.pop('intent')

    for i in range(len(train_labels)):  
        train_labels[i] = str(train_labels[i])

    print("train labels: ",train_labels)

    train_features = train_features.values

    # test_features = test_df.copy()
    # test_labels = test_features.pop("intent")

    # for i in range(len(test_labels)):  
    #     test_labels[i] = str(test_labels[i])

    # test_features = test_features.values

    # valid_features = valid_df.copy()
    # valid_labels = valid_features.pop("intent")

    # for i in range(len(valid_labels)):  
    #     valid_labels[i] = str(valid_labels[i])

    # valid_features = valid_features.values

    chart = sns.countplot(train_labels, palette=HAPPY_COLORS_PALETTE)
    plt.title("Number of texts per intent")
    chart.set_xticklabels(chart.get_xticklabels(), rotation=30, horizontalalignment='right')
    plt.show()

    train_labels = binarizer.fit_transform(train_labels.values)
    # train_labels = binarizer.transform(train_labels.values)
    # test_labels = binarizer.transform(test_labels.values)
    # valid_labels = binarizer.transform(valid_labels.values)

    print(train_labels.shape)

    tfhub_handle_encoder = config['map_name_to_handle'][config['bert_model_name']]
    tfhub_handle_preprocess = config['map_model_to_preprocess'][config['bert_model_name']]

    print(f'BERT model selected           : {tfhub_handle_encoder}')
    print(f'Preprocess model auto-selected: {tfhub_handle_preprocess}')

    bert_preprocess_model = hub.KerasLayer(tfhub_handle_preprocess)

    print(train_features[0])

    text_test = train_features[0]
    text_preprocessed = bert_preprocess_model(text_test)

    print(f'Keys       : {list(text_preprocessed.keys())}')
    print(f'Shape      : {text_preprocessed["input_word_ids"].shape}')
    print(f'Word Ids   : {text_preprocessed["input_word_ids"][0, :12]}')
    print(f'Input Mask : {text_preprocessed["input_mask"][0, :12]}')
    print(f'Type Ids   : {text_preprocessed["input_type_ids"][0, :12]}')

    bert_model = hub.KerasLayer(tfhub_handle_encoder)
    bert_results = bert_model(text_preprocessed)

    print(f'Loaded BERT: {tfhub_handle_encoder}')
    print(f'Pooled Outputs Shape:{bert_results["pooled_output"].shape}')
    print(f'Pooled Outputs Values:{bert_results["pooled_output"][0, :12]}')
    print(f'Sequence Outputs Shape:{bert_results["sequence_output"].shape}')
    print(f'Sequence Outputs Values:{bert_results["sequence_output"][0, :12]}')

    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    metrics = tf.metrics.CategoricalAccuracy()
    optimizer = tf.keras.optimizers.Adam(1e-5)

    classifier_model = build_classifier_model(num_classes)
    classifier_model.summary()
    classifier_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    print(f'Training model with {tfhub_handle_encoder}')

    history = classifier_model.fit(
        x = train_features, 
        y = train_labels, 
        # validation_data = (valid_features,valid_labels), 
        batch_size = config['batch_size'], 
        epochs = config['epochs']
    )

    classifier_model.save_weights('models/intent_classifier.h5')
    print('model saved to disk')

    loss, accuracy = classifier_model.evaluate(test_features, test_labels)
    print(f'Loss: {loss}')
    print(f'Accuracy: {accuracy}')

    history_dict = history.history
    print(history_dict.keys())

    acc = history_dict['categorical_accuracy']
    val_acc = history_dict['val_categorical_accuracy']
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']

    epochs = range(1, len(acc) + 1)
    fig = plt.figure(figsize=(10, 8))
    fig.tight_layout()

    plt.subplot(2, 1, 1)
    # "bo" is for "blue dot"
    plt.plot(epochs, loss, 'r', label='Training loss')
    # b is for "solid blue line"
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.grid(True)
    # plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(epochs, acc, 'r', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.grid(True)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.show()

    examples = [
        'no i do not have it',
        'We can do it on June 4th',
        'Thanks I appreciate it',
        'Can we do it tomorrow?',
        'I do not understand'
    ]

    intents = predict(examples)

    print_my_examples(examples, intents)