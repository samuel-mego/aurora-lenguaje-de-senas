from keras.models import Sequential # type: ignore
from keras.layers import LSTM, Dense, Dropout, BatchNormalization # type: ignore
from keras.regularizers import l2 # type: ignore
from constants import LENGTH_KEYPOINTS

def get_model(max_length_frames, output_length: int):
    model = Sequential()
    
    # Primera capa LSTM con Batch Normalization
    model.add(LSTM(128, return_sequences=True, input_shape=(max_length_frames, LENGTH_KEYPOINTS), kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    
    # Segunda capa LSTM
    model.add(LSTM(128, return_sequences=True, kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    
    # Tercera capa LSTM
    model.add(LSTM(64, return_sequences=False, kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.5))
    
    # Capas densas adicionales
    model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.01)))
    
    # Capa de salida
    model.add(Dense(output_length, activation='softmax'))
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model



