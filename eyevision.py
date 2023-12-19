import cv2
import subprocess
from gtts import gTTS
import pywhatkit
import datetime
import numpy as np
import time
import pyjokes
import wikipedia
import webbrowser
import speech_recognition as sr
from transformers import CLIPProcessor, CLIPModel
import requests



def talk(text):
    tts = gTTS(text=text, lang='en')
    tts.save("output.mp3")

    subprocess.run(["start", "output.mp3"], shell=True)

    print(text)

def get_instruction():
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        print("Listening for a command...")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        audio = recognizer.listen(source)

    try:
        instruction = recognizer.recognize_google(audio).lower()
        print(f"You said: {instruction}")
        return instruction
    except sr.UnknownValueError:
        print("Sorry, I didn't catch that. Please try again.")
        return ""
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
        return ""

def process_instruction(instruction):
    if "play" in instruction:
        song = instruction.replace('play', "").strip()
        talk("Playing " + song)
        pywhatkit.playonyt(song)

    elif 'time' in instruction:
        current_time = datetime.datetime.now().strftime('%I:%M %p')
        talk('The current time is ' + current_time)

    elif 'camera' in instruction:
        talk('Opening camera...')
        capture_video()

    elif 'joke' in instruction:
        joke = pyjokes.get_joke()
        talk(joke)

    elif 'info' in instruction:
        query = instruction.replace('info', "").strip()
        try:
            info = wikipedia.summary(query, sentences=2)
            talk(f"Here is some information about {query}: {info}")
        except wikipedia.exceptions.DisambiguationError as e:
            talk(f"There are multiple options. Please be more specific.")
        except wikipedia.exceptions.PageError as e:
            talk(f"Sorry, I couldn't find information about {query}.")

    elif 'search' in instruction:
        search_query = instruction.replace('search', "").strip()
        search_on_google(search_query)

def search_on_google(query):
    talk(f"Searching Google for {query}")
    google_search_url = f"https://www.google.com/search?q={query}"
    webbrowser.open(google_search_url)

def capture_video():
    cap = cv2.VideoCapture(0) 

    
    net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')

    
    with open('coco.names', 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    cooldown_timer = 0
    cooldown_period = 10  
    last_identified_object = None

    while True:
        ret, frame = cap.read()

        
        object_info = detect_objects(frame, net, classes)

        if object_info and object_info != last_identified_object and cooldown_timer <= 0:
            x, y, x1, y1, class_name = object_info
            talk(f"I see a {class_name}")
            last_identified_object = object_info
            cooldown_timer = cooldown_period

        
        display_objects(frame, [object_info] if object_info else [])

        cv2.imshow('Camera Feed', frame)

        
        cooldown_timer = max(0, cooldown_timer - 1)

        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def detect_objects(frame, net, classes):
    height, width = frame.shape[:2]

    
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)

    net.setInput(blob)

    
    output_layers = net.getUnconnectedOutLayersNames()

    
    detections = net.forward(output_layers)

    
    object_info = None
    max_confidence = 0
    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.2 and confidence > max_confidence:  
                max_confidence = confidence
                center_x = int(obj[0] * width)
                center_y = int(obj[1] * height)
                w = int(obj[2] * width)
                h = int(obj[3] * height)

                x = int(center_x - w/2)
                y = int(center_y - h/2)

                class_name = classes[class_id]
                object_info = (x, y, x+w, y+h, class_name)

    return object_info

def display_objects(frame, objects):
    for (x, y, x1, y1, class_name) in objects:
        cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)
        cv2.putText(frame, class_name, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        
        if class_name.lower() == 'unknown':
            specify_unknown_object()

def specify_unknown_object():
    talk("I detected an unknown object. Please specify the object.")
    
   
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)
        audio = recognizer.listen(source)

    try:
        unknown_object = recognizer.recognize_google(audio).lower()
        talk(f"You specified the unknown object as {unknown_object}. Searching Google...")
        search_on_google(unknown_object)
    except sr.UnknownValueError:
        talk("Sorry, I didn't catch that. Please try again.")
        return
    except sr.RequestError as e:
        talk(f"Could not request results from Google Speech Recognition service; {e}")
        return



if __name__ == "__main__":
    while True:
        instruction = get_instruction()

        if instruction:
            process_instruction(instruction)
