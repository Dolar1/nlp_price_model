from sklearn.externals import joblib
import numpy as np
model = joblib.load('model.pkl')
tokenizer = joblib.load('tokenizer.pkl')
pad_sequences = joblib.load('pad_sequences.pkl')
import os


while True:
    
    if os.path.isfile('input.txt'):
        #print("*")
        with open('input.txt','r') as af:
            text = af.read()
            #print(text)
        input_text = []
        input_text.append(text)
        con = tokenizer.texts_to_sequences(input_text)
        con = pad_sequences(con,maxlen=250)
    
        y_pred = model.predict(con)
    
        i = np.argmax(y_pred)
        lower = (i+1-y_pred[0][i])*10000
        upper = (i+1+y_pred[0][i])*10000 + 20000
        
        with open('output.txt','w') as f:
            answer = ((lower,upper))
            f.write(str(int(lower)))
            f.write('\n')
            f.write(str(int(upper)))
        f.close()
    
        
    #input_text = ['I have a old dell vostro gaming laptop with 16 gb ram, 2tb harddisk windows 10 which is about 2 years old and is has a i7 7th gen in it processor']
    #input_text = ['i want 2 tb hard disk with 16 gb ram and high end graphics card like 4 gb nvidea or an 2 gb amd radeon']
    