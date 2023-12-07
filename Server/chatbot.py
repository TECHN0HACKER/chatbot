import random
import json
import pickle
import numpy as np
import asyncio
import websockets
import nltk
import os
from nltk.stem import WordNetLemmatizer
from keras.models import load_model


lemmatizer = WordNetLemmatizer()

intents = json.loads(open("intents.json").read())

words = pickle.load(open("words.pkl",'rb'))
classes =pickle.load(open("classes.pkl",'rb'))
model = load_model("chatbotmodel.keras")

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_word = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for w in sentence_word:
        for i,word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1],reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent':classes[r[0]],'probability':str(r[1])})
    return return_list
    
def get_response(intents_list, intents_json, user_input, context):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    result = None
    for i in list_of_intents:
        if i['tag'] == tag:
            magnitude = None
            order = None
            unknown=0
            for response in i['responses']:
                unknown+=1
                if 'magnitude' in response and isinstance(response, dict) == True:
                    for word in user_input.split():
                        for synonym in response['magnitude']:
                            if synonym == word:
                                magnitude = response['magnitude']
                                break
                    if magnitude:
                        break
                    if len(i['responses'])-unknown == 1:
                        magnitude=['default']
                        break
            if magnitude:
                if any(mag in magnitude for mag in ["high", "a lot", "lots", "too much", "huge", "quite", "significant", "extreme", "large"]):
                    order = [0, 1, 2]
                elif any(mag in magnitude for mag in ["typical", "ordinary", "normal", "fine", "just ok", "medium"]):
                    order = [1, 2, 0]
                elif any(mag in magnitude for mag in ["low", "less", "little", "not much", "least", "bare", "minimum"]):
                    order = [2, 1, 0]
                else:
                    order = [3]
                sorted_responses = [i['responses'][idx]['response'] for idx in order]
                result = " ".join(sorted_responses)
                break  
            else:
                if 'context_set' in i:
                    context = i["context_set"]
                    result = random.choice(i['responses'])
                elif context in i['responses']:
                    result=random.choice(i['responses'][context])
                    context = 'None'
                elif context != None and context != 'None':
                    if context in i['responses']:                        
                        #print(context)
                        result = i['responses']['None']
                    else:
                        result = random.choice(i["responses"])
                        #print(i)                        
                else:
                    result = random.choice(i["responses"])
                    #print(i)
    if not result:
        result = "I'm not sure about that."
    #print(context)
    if context == None:
        context = 'None'
    return context, result
print("GO Bot is running!!!")
async def server(websocket, path):
    print('first try to connect')
    context = 'None'
    x=0
    id=""
    while True:
        message = await websocket.recv()
        message=message.lower()
        x+=1
        if x==1:
                #print(message)
                filename = message+".txt"
                savedmsg = ""
                id=filename
                if os.path.exists(filename):
                    file = open(filename, "r")
                    content = file.read()
                    file.close()
                    parts = content.split("-*-----------------------------------------------------*-")
                    for part in parts:
                        part = part.strip()
                        if part:
                            savedmsg=part
                            #print(part)
                            await websocket.send(savedmsg)
                    await websocket.send("*-*-*END_MESSAGE*-*-*")
                else:
                    with open(id, "w") as file:
                        file.write("\n-*-----------------------------------------------------*-\n")
                    await websocket.send("*-*-*END_MESSAGE*-*-*")
        elif x>1:
            ints = predict_class(message)
            recontext, res = get_response(ints,intents,message,context)
            #print(recontext)
            context=recontext
            #print(res)
            await websocket.send(res)
            with open(id, "a") as file:
                file.write(message)
                file.write("\n-*-----------------------------------------------------*-\n")
                file.write(res)
                file.write("\n-*-----------------------------------------------------*-\n")
start_server = websockets.serve(server, "localhost", 9000)
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
