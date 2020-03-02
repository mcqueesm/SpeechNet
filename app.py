

import record_tools as rec
import tensorflow as tf
from keras.models import load_model 
from keras.backend import clear_session
from tkinter import *
from PIL import ImageTk, Image
import time
import numpy as np
import multiprocessing
import threading
import queue
import wave
import pyaudio

class GuiPart:
    def __init__(self, parent, q, end_application):
        pad=3
        self._geom="{0}x{1}+0+0".format(
            parent.winfo_screenwidth()-pad, parent.winfo_screenheight()-pad)
        parent.geometry('500x700+400+50')
        parent.configure(background='black')
        parent.title("Jonetics")
        parent.bind('<Escape>',self.toggle_geom)  
        root.protocol("WM_DELETE_WINDOW", end_application)
         
        #open image
        self.img = Image.open('/Users/joni.jpg')
        self.img = self.img.resize((500, 500), Image.ANTIALIAS)
        self.pic = ImageTk.PhotoImage(self.img)
        #sound prediction variable
        self.result = StringVar()
        #root
        self.root = parent
        #queue
        self.q = q

        #label for image
        self.l1= Label(parent, image=self.pic, bg='black').pack()
        #label instructions
        self.l2 = Label(parent, text="Let's hear some Joni-speak!", bg='black', fg='white',font=('Helvetica', 25)).pack()
        #label for prediction result
        self.l3 = Label(parent, textvariable=self.result, bg='black', fg='white',font=('Helvetica', 80)).pack()
        self.score = StringVar()
        self.l4 = Label(parent, textvariable=self.score, bg='black', fg='white',font=('Helvetica', 40)).pack()
    def toggle_geom(self,event):
        geom=self.root.winfo_geometry()
        self.root.geometry(self._geom)
        self._geom=geom

    def CheckQueuePoll(self):
        try:
            self.result.set(self.q.get_nowait())
         
        except queue.Empty:
            pass
      
class ThreadedClient:
    def __init__(self, parent):
        self.parent = parent
        self.q = queue.Queue()
        self.gui = GuiPart(parent, self.q, self.end_application)
        self.model = load_model('joni_master_8-23.h5')
        self.model._make_predict_function()
        self.graph = tf.get_default_graph()
        self.running =True
        self.thread1 = threading.Thread(target=self.predict)
        self.thread1.start()

        self.periodic_call()

    def periodic_call(self):

        self.parent.after(200, self.periodic_call)
        self.gui.CheckQueuePoll()
        if not self.running:
            # This is the brutal stop of the system.  You may want to do
            # some cleanup before actually shutting it down.
            import sys
            sys.exit(1)

    
    def predict(self):
        while self.running:
            
            sw, r = rec.record()
            
            r = rec.norm_for_cnn(r)

            data, r = rec.process_wav_file(r)
            with self.graph.as_default():
                pred = np.argmax(self.model.predict(np.expand_dims(r, axis=0))[0])
                self.gui.score.set(self.model.predict(np.expand_dims(r, axis=0))[0][pred])
            if self.model.predict(np.expand_dims(r, axis=0))[0][pred] >= .7:
                if pred==0: 
                    self.q.put('Joni')
                    #self.play_sound('/Users/jonirecord/joni_english.wav')
                elif pred==1:
                    self.q.put('Bird')
                elif pred==2:
                    self.q.put('Burger')
                elif pred==3:
                    self.q.put('Nemo')
                elif pred==4:
                    self.q.put('Sean')
                elif pred==5:
                    self.q.put('Snow')
                elif pred==6:
                    self.q.put('Motorcycle')
                elif pred==7:
                    self.q.put('Indian')
                elif pred==8:
                    self.q.put('Train')
            self.predict()
    
    def play_sound(self, wavpath):
        chunk = 1024
        wf = wave.open(wavpath, 'rb')

        # create an audio object
        p = pyaudio.PyAudio()

        # open stream based on the wave object which has been input.
        stream = p.open(format =
                        p.get_format_from_width(wf.getsampwidth()),
                        channels = wf.getnchannels(),
                        rate = wf.getframerate(),
                        output = True)

        # read data (based on the chunk size)
        data = wf.readframes(chunk)

        # play stream (looping from beginning of file to the end)
        while data != '':
            # writing to the stream is what *actually* plays the sound.
            stream.write(data)
            data = wf.readframes(chunk)

        # cleanup stuff.
        stream.close()    
        p.terminate()
    def end_application(self):
        self.running = False
root = Tk()
client = ThreadedClient(root)
root.mainloop()


'''
our_model = load_model('joni_model4.h5')

    
    
count = 1



try:
    while True:
        print("Let's hear some Joni-speak")
        #print("Round " + str(count))
        sw, r = rec.record()

        r = rec.norm_for_cnn(r)

        data, r = rec.process_wav_file(r)

        #wavfile.write('/Users/jonirecord/sean_sign' + str(count) + '.wav', RATE, data)

        count += 1
        
        pred = np.argmax(our_model.predict(np.expand_dims(r, axis=0))[0])
        print("Prediction:")
        if pred==0: 
            print('Joni')
        elif pred==1:
            print('Bird')
        elif pred==2:
            print('Nemo')
        
except KeyboardInterrupt:
    pass
'''