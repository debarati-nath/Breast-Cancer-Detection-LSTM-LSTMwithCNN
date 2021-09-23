#Import libraries
from keras.models import Sequential
from keras.layers import LSTM, Activation, Dense, Dropout, Convolution1D, MaxPooling1D, TimeDistributed

#CNN-LSTM
for i, (train_idx, valid_idx) in enumerate(splits):
    X_train_fold = np.array(train_data[train_idx.astype(np.int)])
    y_train_fold = np.array(train_labels[train_idx.astype(np.int), np.newaxis])
    X_val_fold = np.array(train_data[valid_idx.astype(np.int)])
    y_val_fold = np.array(train_labels[valid_idx.astype(np.int), np.newaxis])
    model = Sequential()
    model.add(Convolution1D(34, 4, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=4))
    # model.add(TimeDistributed(Dense(10)))
    model.add(LSTM(60, return_sequences=True, input_shape=(train_data.shape[1], train_data.shape[2])))
    # model.add(Dropout(0.2))
    # model.add(LSTM(32, activation='relu', return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(10))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='RMSprop', metrics=['accuracy'])
    print(f'Fold {i + 1}')
    history = model.fit(X_train_fold, y_train_fold, epochs=5, batch_size=20, validation_data=(X_val_fold, y_val_fold),
                        verbose=1, shuffle=False)
    from sklearn.metrics import classification_report

    target = ['class0', 'class1']

    from sklearn.metrics import roc_curve

    y_pred = model.predict(X_val_fold)
    fpr, tpr, thresold = roc_curve(y_val_fold, y_pred)
    from sklearn.metrics import auc

    auc_keras = auc(fpr, tpr)
    print(classification_report(y_val_fold, y_pred.round(), target_names=target))
    print('AUC:', auc_keras)
    from sklearn.metrics import confusion_matrix

    tn, fp, fn, tp = confusion_matrix(y_val_fold, y_pred.round()).ravel()
    specificity = tn / (tn + fp)
    print('specificity:', specificity)
    sens = tp / (tp + fn)
    print('sensitivity:', sens)
    plt.figure(1)
    plt.plot(history.history['loss'], color='b', label='Training Loss')
    plt.plot(history.history['val_loss'], color='r', label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss')
    plt.legend(loc='best')
    plt.show()
    plt.figure(2)
    plt.plot(history.history['accuracy'], color='b', label='Training Accuarcy')
    plt.plot(history.history['val_accuracy'], color='r', label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy')
    plt.legend(loc='best')
    plt.show()
    plt.figure(3)
    plt.plot(fpr, tpr, label='Area={:.3f}'.format(auc_keras))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='best')
    plt.show()
