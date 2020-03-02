words=('befelled','recalled','expelled','swelled','tested','marked','scott','brutt') 
def firstif(words):  
    for w in words:  
        if w.endswith('ed'):  
             return (w[:-2]) 
        else:  
            return(w) 
print(firstif(words))
words2=tuple(firstif(words)) 
def secondif(words2):  
    for w2 in words2:  
        if w2.endswith('ll'):  
            return (w2[:-1]) 
        else:  
            return(w2) 
#secondif(w2)