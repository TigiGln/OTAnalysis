#!/usr/bin/python

import threading

class myThread(threading.Thread):
    def __init__(self, threadID, name):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name

    def run(self):
        print("Starting " + self.name)
    
if __name__ == "__name__":
    thread1 = myThread(1, "Thread-1")
    thread2 = myThread(2, "Thread-2")

    thread1.start()
    thread2.start()
