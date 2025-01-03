import os
import cv2
import numpy as np
import requests
import time
from mediapipe.python.solutions.holistic import Holistic
from keras.models import load_model # type: ignore
from helpers import *
from constants import *
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
from text_to_speech import text_to_speech


esp32_ip = "http://192.168.223.42"

def send_to_esp32(text):
    
    try:
        response = requests.get(f"{esp32_ip}/?text={text}")
        if response.status_code == 200:
            print("Señal enviada al ESP32:", response.text)
        else:
            print(f"Error al enviar señal al ESP32: {response.status_code}")
    except Exception as e:
        print(f"Error en la conexión con el ESP32: {e}")

def obtener_distancia():
    try:
        # Realizar una petición GET al ESP32
        response = requests.get(f"{esp32_ip}/distancia")
        if response.status_code == 200:
            distancia = response.text.strip()
            #print(f"Distancia recibida en cm: {distancia}")
            return distancia
        else:
            #print(f"Error en la solicitud: {response.status_code}")
            return f"Error en la solicitud: {response.status_code}"
    except requests.exceptions.RequestException as e:
        #print(f"Error de conexión: {e}")
        return f"Error de conexión: {e}"

def send_translation_to_node_red(text):
    try:
        url = 'http://localhost:1880/translation'  # Endpoint de Node-RED
        data = {'translation': text}
        requests.post(url, json=data)
        print(f"Enviado a Node-RED: {text}")
    except Exception as e:
        print(f"Error al enviar a Node-RED: {e}")

def interpolate_keypoints(keypoints, target_length=15):
    current_length = len(keypoints)
    if current_length == target_length:
        return keypoints
    
    indices = np.linspace(0, current_length - 1, target_length)
    interpolated_keypoints = []
    for i in indices:
        lower_idx = int(np.floor(i))
        upper_idx = int(np.ceil(i))
        weight = i - lower_idx
        if lower_idx == upper_idx:
            interpolated_keypoints.append(keypoints[lower_idx])
        else:
            interpolated_point = (1 - weight) * np.array(keypoints[lower_idx]) + weight * np.array(keypoints[upper_idx])
            interpolated_keypoints.append(interpolated_point.tolist())
    
    return interpolated_keypoints

def normalize_keypoints(keypoints, target_length=15):
    current_length = len(keypoints)
    if current_length < target_length:
        return interpolate_keypoints(keypoints, target_length)
    elif current_length > target_length:
        step = current_length / target_length
        indices = np.arange(0, current_length, step).astype(int)[:target_length]
        return [keypoints[i] for i in indices]
    else:
        return keypoints
    
def evaluate_model(src=None, threshold=0.8, margin_frame=1, delay_frames=3):
    kp_seq, sentence = [], []
    word_ids = get_word_ids(WORDS_JSON_PATH)
    model = load_model(MODEL_PATH)
    count_frame = 0
    fix_frames = 0
    recording = False

    with Holistic() as holistic_model:
        video = cv2.VideoCapture(src or 0)
        contador2= 0 #-------------agreagado
        while video.isOpened():
            contador2 += 1 #----------agregado
            ret, frame = video.read()
            if not ret: break
            results = mediapipe_detection(frame, holistic_model)
        
            # TODO: colocar un máximo de frames para cada seña,
            # es decir, que traduzca incluso cuando hay mano si se llega a ese máximo.
            if there_hand(results) or recording:
                recording = False
                count_frame += 1
                if count_frame > margin_frame:
                    kp_frame = extract_keypoints(results)
                    kp_seq.append(kp_frame)
            
            else:
                if count_frame >= MIN_LENGTH_FRAMES + margin_frame:
                    fix_frames += 1
                    if fix_frames < delay_frames:
                        recording = True
                        continue
                    kp_seq = kp_seq[: - (margin_frame + delay_frames)]
                    kp_normalized = normalize_keypoints(kp_seq, int(MODEL_FRAMES))
                    res = model.predict(np.expand_dims(kp_normalized, axis=0))[0]
                    print("empieza el np.argmax(res)")
                    valor=np.argmax(res)
                    print(valor, f"({res[valor] * 100:.2f}%)")
                    if res[valor] > threshold :
                        print(valor+1) 
                        send_to_esp32(str(valor+1))
                        word_id = word_ids[valor].split('-')[0]
                        sent = words_text.get(word_id)
                        sentence.insert(0, sent)
                        print(sent)
                        #text_to_speech(sent) # ONLY LOCAL (NO SERVER)
                        send_translation_to_node_red(sent)  # Enviar a Node-RED
                
                recording = False
                fix_frames = 0
                count_frame = 0
                kp_seq = []
            
            if not src:
                cv2.rectangle(frame, (0, 0), (640, 35), (245, 117, 16), -1)
                cv2.putText(frame, ' | '.join(sentence), FONT_POS, FONT, FONT_SIZE, (255, 255, 255))
                
                draw_keypoints(frame, results)
                cv2.imshow('Traductor LSP', frame)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
                #------------------------
                if contador2 == 400 :  #contador % 400 si es multiplo el resultado es 0
                    print(contador2) 
                    contador2 = 0
                    live_distance = float(obtener_distancia())
                    print(live_distance)
                    if live_distance > 200:
                        break
                #-----------
                    
        video.release()
        cv2.destroyAllWindows()
        return sentence
    
if __name__ == "__main__":
    #while True:
        send_to_esp32("off")
        contador = 0
        while contador < 15:
            distancia=float(obtener_distancia())
            print(distancia)
            if distancia > 40 and distancia < 120 :
                contador += 1
            time.sleep(0.5) 
        send_to_esp32("on")
        evaluate_model()