# -*- coding: utf-8 -*-
"""
@Author    : xrh
@Time      : 2021/6/18 15:13
"""
import time
import wave

import pyaudio
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (QGroupBox, QVBoxLayout, QGridLayout, QWidget, QPushButton, QDesktopWidget, QApplication,
                             QLabel, QHBoxLayout, QProgressBar)
from PyQt5.QtGui import QIcon, QPixmap, QPalette, QBrush, QFont
import sys
import sounddevice as sd
import soundfile as sf
import librosa
import tensorflow as tf
from keras.models import load_model
import numpy as np
from tqdm import tqdm


class RecordingWindow(QWidget):
    # First window the user opens
    def __init__(self):
        super(RecordingWindow, self).__init__()
        self.initUI()

    def initUI(self):
        self.model = load_model('model/best_model.hdf5')
        self.resize(1000, 600)
        self.center()
        self.setWindowIcon(QIcon('./image/icon.png'))
        self.setWindowTitle("Speech Recognition System")
        palette = QPalette()
        palette.setBrush(QPalette.Background, QBrush(QPixmap("./image/background.png")))
        self.setPalette(palette)
        self.setStyleSheet("QLabel{"
                           # "color:rgb(0, 0, 0, 180);"
                           # "background-color: rgb(0, 0, 0, 20);"
                           "color: rgb(59, 161, 218);" 
                           "font-family:Kristen ITC; font-size:200px"
                           "}"
                           "QPushButton{"
                           "color: rgb(255, 136, 0);"
                           "background-color:rgb(255, 255, 255, 0);"
                           "border:none;"
                           "font-family:Kristen ITC; font-size:40px;"
                           "}"
                           "QPushButton::hover{"
                           "color: rgb(255, 195, 0);"
                           "font-size:45px;"
                           "}")


        # Label of welcome text
        self.stateTitle = QLabel(self)
        self.stateTitle.setText("🎧Ready to Recording! ( •̀ ω •́ )✧")
        self.stateTitle.setAlignment(QtCore.Qt.AlignCenter)
        self.stateTitle.setStyleSheet("font-size:60px")


        # PlaceHolder for recording figure
        self.img = QPixmap("./image/recognition.png")
        h = self.img.height()
        w = self.img.width()
        self.img.scaled(w*3, h*3)
        self.labelImg = QLabel()
        self.labelImg.setPixmap(self.img)
        self.labelImg.setAlignment(QtCore.Qt.AlignCenter)
        # self.labelImg.setScaledContents(True)

        # Label for Tip title
        self.tipTitle = QLabel()
        self.tipTitle.setText("\n💡Tips ฅʕ•̫͡•ʔฅ")
        self.tipTitle.setAlignment(QtCore.Qt.AlignCenter)
        self.tipTitle.setStyleSheet("font-size:60px")


        # Label for tips
        self.tipsLabel = QLabel()
        self.tipsLabel.setText("\n[STEP 1]👇\nClick \"Start Recording\" button to start record;\n"
                               "\n[STEP 2]👇\nSpeak a word form {'up', 'down', 'yes', 'no'}in 1s;\n"
                               "\n[STEP 3]👇\nResults are displayed immediately!"
                               "\n")
        self.tipsLabel.setStyleSheet("font-size:30px")


        # Label to show recognition result
        self.resultLabel = QLabel()
        self.resultLabel.setText("")
        self.resultLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.resultLabel.setStyleSheet("color:rgb(231, 76, 60)")


        # Button to start Recording
        self.recordingBt = QPushButton(self)
        self.recordingBt.setText("🎤[Start Recording]")
        self.recordingBt.clicked.connect(self.startRecording)

        # Button to quit the game
        self.quitBt = QPushButton(self)
        self.quitBt.setText("👋[Quit System]")
        self.quitBt.clicked.connect(self.GameQuit)

        # # Progressbar for recording
        # self.progressbar = QProgressBar(self)
        # self.progressbar.setMinimum(0)
        # self.progressbar.setMaximum(100)
        # self.step = 0  # 3
        # self.timer = QTimer(self)  # 4
        # self.timer.timeout.connect(self.update_func)
        #
        # self.thread = None

        # layout
        left = QVBoxLayout()
        left.addWidget(self.stateTitle)
        # left.addWidget(self.progressbar)
        left.addWidget(self.labelImg)
        left.addWidget(self.resultLabel)
        left.addWidget(self.recordingBt)

        right = QVBoxLayout()
        right.addWidget(self.tipTitle)
        right.addWidget(self.tipsLabel)
        right.addWidget(self.quitBt)

        left_w = QWidget()
        right_r = QWidget()
        right_r.setStyleSheet(
            "background-color: rgb(236, 240, 241)"
        )
        left_w.setLayout(left)
        right_r.setLayout(right)

        layout = QHBoxLayout()
        layout.addWidget(left_w, 4)
        layout.addWidget(right_r, 2)

        self.setLayout(layout)
        self.resultLabel.hide()
        # self.progressbar.hide()

        self.showMaximized()

    # def update_func(self):
    #     self.step += 1
    #     self.progressbar.setValue(self.step)
    #     QApplication.processEvents()
    #     if self.step >= 100:
    #         # self.recordingBt.setText('🎤[Start Recording]')
    #         self.timer.stop()
    #         self.step = 0

    def center(self):
        # Function to center window
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    # def start_stop_func(self):
    #     self.progressbar.show()
    #     if self.recordingBt.text() == '🎤[Start Recording]':
    #         self.recordingBt.setText('📍[Stop Recording]')
    #         self.timer.start(1)
    #         QApplication.processEvents()
    #
    #     else:
    #         self.recordingBt.setText('🎤[Start Recording]')
    #         self.timer.stop()
    #     # self.startRecording()

    def startRecording(self):
        # self.start_stop_func()
        samplerate = 16000
        duration = 1  # seconds
        self.filename = 'testsound.wav'
        print("start")
        self.stateTitle.setText('🎙Recording...')
        # #1e90ff
        # self.timer.start(1)
        # self.progressbar.show()
        self.resultLabel.hide()
        self.labelImg.show()
        QApplication.processEvents()
        mydata = sd.rec(int(samplerate * duration), samplerate=samplerate,
                        channels=1, blocking=True)
        print("end")
        sd.wait()
        sf.write(self.filename, mydata, samplerate)
        print('----------Recognition----------')
        self.result = self.speechRecognition()
        print(self.result)
        self.stateTitle.setText('📢Recognition Result:')
        self.resultLabel.setText(self.result)
        self.resultLabel.show()
        self.labelImg.hide()

        QApplication.processEvents()

    def speechRecognition(self):
        print('----------Recognition----------')
        classes = ['down', 'no', 'up', 'yes']
        samples, sample_rate = librosa.load(self.filename, sr=16000)
        samples = librosa.resample(samples, sample_rate, 8000)
        prob = self.model.predict(samples.reshape(1, 8000, 1))
        index = np.argmax(prob[0])
        print('---------------result:',classes[index])
        return classes[index]


    def GameQuit(self):
        self.close()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = RecordingWindow()
    win.show()
    sys.exit(app.exec_())



