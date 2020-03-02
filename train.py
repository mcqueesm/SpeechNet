#Model Results:
#'joni_master_8-30.h5': final test score: 96.8 (9 signs, no data roll)
#'joni_master_8-30b.h5': final test score 97.8 (9 signs, no data roll)
#'joni_master_8-31.h5': final test score 99.15 (9 signs, with data roll)
from models import deep, deep_cnn

import numpy as np
from sklearn.metrics import accuracy_score
from keras.callbacks import EarlyStopping

from dataset import DatasetGenerator

import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

DIR1 = '/Users/jonifinal/*/*.wav' # unzipped train and test data
DIR2 = '/Users/jonifinal'
INPUT_SHAPE = (177,98,1)
BATCH = 200
EPOCHS = 50

LABELS = 'joni bird burger nemo sean snow motorcycle'.split()
NUM_CLASSES = len(LABELS)

#==============================================================================
# Prepare data      
#==============================================================================
dsGen = DatasetGenerator(label_set=LABELS) 
# Load DataFrame with paths/labels for training and validation data 
# and paths for testing data 
df = dsGen.load_data(DIR1, DIR2)


dsGen.apply_train_test_split(test_size=0.3, random_state=2018)
dsGen.apply_train_val_split(val_size=0.2, random_state=2018)

print('Test: ', dsGen.df_test)
print('Train: ', dsGen.df_train)
print('Val: ', dsGen.df_val)
#==============================================================================
# Train
#==============================================================================              
model = deep_cnn(INPUT_SHAPE, NUM_CLASSES)
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['acc'])

callbacks = [EarlyStopping(monitor='val_acc', patience=2, verbose=1, mode='max')]

history = model.fit_generator(generator=dsGen.generator(BATCH, mode='train'),
                              steps_per_epoch=int(np.ceil(len(dsGen.df_train)/BATCH)),
                              epochs=EPOCHS,
                              verbose=1,
                              callbacks=callbacks,
                              validation_data=dsGen.generator(BATCH, mode='val'),
                              validation_steps=int(np.ceil(len(dsGen.df_val)/BATCH)))

model.save('joni_master_9-7b.h5')
#==============================================================================
# Predict
#==============================================================================
y_pred_proba = model.predict_generator(dsGen.generator(BATCH, mode='test'), 
                                     int(np.ceil(len(dsGen.df_test)/BATCH)), 
                                     verbose=1)
y_pred = np.argmax(y_pred_proba, axis=1)

y_true = dsGen.df_test['label_id'].values

acc_score = accuracy_score(y_true, y_pred)
print(acc_score)