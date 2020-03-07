import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from keras.layers import Input, Embedding, Activation, Flatten, Dense
from keras.layers import Conv1D, MaxPooling1D, Dropout
from keras.models import Model
from keras.utils import to_categorical

def parseCSV_testing(file):
    data = pd.read_csv(file)
    data['Index'] = np.arange(len(data))
    data["Word"]= data["Word"].str.lower()
    return data[['Word', 'Language', "Index"]]

def parseCSV(file):
    data = pd.read_csv(file)
    data = data.sample(500)
    data["Word"]= data["Word"].str.lower()
    data['Index'] = np.arange(len(data))
    return data[['Word', 'Language', "Index"]]

class CNN():
    def __init__(self, train_data_source, test_data_source):
        self.train_df = parseCSV(train_data_source)
        self.test_df = parseCSV_testing(test_data_source)

        self.x_train = np.array(self.train_df["Word"])
        self.x_test = np.array(self.test_df["Word"])

        CHAR_STRING = 'abcdefghijklmnopqrstuvwxyzáéíóúüñàèìòùçâêîôûëïäöß()-āēīōū’ā̆ē̆ī̆ō̆ăĭḗū́u̯ṇ̃þʒ¹²/\ :;"!?¿¡".'
        char_dict = {}

        for i, char, in enumerate(CHAR_STRING):
            char_dict[char] = i + 1

        #print(char_dict)

        self.tk = Tokenizer(num_words=None, char_level=True, oov_token='UNK')
        self.tk.fit_on_texts(self.x_train)

        self.tk.word_index = char_dict.copy()
        self.tk.word_index[self.tk.oov_token] = max(char_dict.values()) + 1

        train_sequences = self.tk.texts_to_sequences(self.x_train)
        test_texts = self.tk.texts_to_sequences(self.x_test)

        #print(tk.word_index)

        # Padding
        train_data = pad_sequences(train_sequences, maxlen=1014, padding='post')
        test_data = pad_sequences(test_texts, maxlen=1014, padding='post')

        # Convert to numpy array
        train_data = np.array(train_data, dtype='float32')
        test_data = np.array(test_data, dtype='float32')

        self.class2indexes = dict((l, i) for i, l in enumerate(set(self.train_df["Language"])))
        self.index2class = dict((i, l) for i, l in enumerate(set(self.train_df["Language"])))

        print(self.class2indexes)

        train_classes = self.train_df["Language"]
        train_class_list = [self.class2indexes[x] for x in train_classes]

        test_classes = self.test_df["Language"]
        test_class_list = [self.class2indexes[x] for x in test_classes]

        train_classes = to_categorical(train_class_list)
        test_classes = to_categorical(test_class_list) #converts class to binary class martix 

        vocab_size = 93
        print(vocab_size)

        embedding_weights = [] 
        embedding_weights.append(np.zeros(vocab_size))

        for i in range(vocab_size):
            onehot = np.zeros(vocab_size)
            onehot[i] = 1
            embedding_weights.append(onehot)


        embedding_weights = np.array(embedding_weights)
        # print(embedding_weights.shape)
        # print(embedding_weights)
        print('Load')

        # Model Construction

        #parameters
        input_size = 1014
        embedding_size = 93
        conv_layers = [[256, 7, 3],
                        [256, 7, 3],
                        [256, 3, -1],
                        [256, 3, -1],
                        [256, 3, -1],
                        [256, 3, 3]]

        fully_connected_layers = [1024,1024]
        nums_of_classes = 7
        dropout_p = 0.5
        optimizer = "adam"
        loss = "categorical_crossentropy"

        # Embedding layer Initialization
        embedding_layer = Embedding(vocab_size + 1, embedding_size,
                                    input_length=input_size,
                                    weights=[embedding_weights])


        # Input
        inputs = Input(shape=(input_size,), 
                    name='input', dtype='int64')  # shape=(?, 1014)

        x = embedding_layer(inputs)
        #conv 
        for filter_num, filter_size, pooling_size in conv_layers:
            x = Conv1D(filter_num, filter_size)(x) #data_format = 'channels_first'
            x = Activation('relu')(x)
            if pooling_size != -1:
                x = MaxPooling1D(pool_size=pooling_size)(x) #prevents overfitting
        x = Flatten()(x) #turns in a martix into a 1D array 
        #Fully connected layers
        for dense_size in fully_connected_layers:
            x = Dense(dense_size, activation='relu')(x)
            x = Dropout(dropout_p)(x) #help reduce overfitting

        #Output Layer
        predictions = Dense(nums_of_classes, activation='softmax')(x)
        #Build Model
        self.model = Model(input=inputs, outputs=predictions)
        self.model.compile(optimizer=optimizer, loss=loss)
        self.model.summary()

        self.model.fit(train_data, train_classes, validation_data=(test_data, test_classes), batch_size=128, epochs=10, verbose=2)

    def predict(self, filename): #used to be false
        x  = parseCSV_testing(filename)  
        x_test = np.array(x["Word"])
        test_texts = self.tk.texts_to_sequences(x_test)
        test_data = pad_sequences(test_texts, maxlen=1014, padding='post')
        test_data = np.array(test_data, dtype='float32')
        predictions = self.model.predict(test_data)
        predict_results = predictions.argmax(axis=-1)
        for i in range(len(predict_results)):
            print("-------------------------------------------------")
            index = self.test_df["Index"][i]
            print(self.test_df["Word"][index])
            print("Language: ", self.index2class[predict_results[i]])
            print("Probability: ", predictions[i][predict_results[i]])
        return 

def main():
    train_data_source = 'language_dataset.csv'
    # train_data_source = 'language_dataset_with_stopwords.csv'
    # train_data_source = 'Difficult_Set.csv'
    test_data_source = 'test_data_v1.csv'
    cnn = CNN(train_data_source, test_data_source)    
    cnn.predict(test_data_source)

main()