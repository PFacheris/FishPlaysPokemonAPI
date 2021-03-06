from config import config

import cv2
import math
import numpy as np
import urllib
import time
import sys
import os

import tornado.escape
import tornado.ioloop
import tornado.web
from tornado import gen
from tornado.escape import json_encode
from multiprocessing import Process, Pipe, Manager

def get_percentage_from_pixels(dim, type="width"):
  if type == "width":
    return str(100 * (dim / config.CONTROLLER.WIDTH))
  if type == "height":
    print dim
    return str(100 * (dim / config.CONTROLLER.HEIGHT))
  else:
    return None

class DetectorProcess(Process):
  @staticmethod
  def processFrame(bytes):
    return cv2.imdecode(np.fromstring(bytes, dtype=np.uint8),cv2.CV_LOAD_IMAGE_COLOR)

  @staticmethod
  def diffImg(t1, t2):
    gray1 = cv2.cvtColor(t1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(t2, cv2.COLOR_BGR2GRAY)
    d1 = cv2.absdiff(gray1, gray2)
    blurred = cv2.blur(d1, (15, 15))
    return cv2.threshold(blurred,5,255,cv2.THRESH_BINARY)[1]

  def __init__(self, x, y):
    Process.__init__(self)
    self.x = x
    self.y = y

  def run(self):
    stream = urllib.urlopen(config.VIDEO.INPUT)
    running_average = np.zeros((config.CONTROLLER.HEIGHT,config.CONTROLLER.WIDTH, 3), np.float64) # image to store running avg

    bytes = ''
    while True:
      bytes+=stream.read(1024)
      a = bytes.find('\xff\xd8')
      b = bytes.find('\xff\xd9')
      if a!=-1 and b!=-1:
        img = DetectorProcess.processFrame(bytes[a:b+2])
        bytes = bytes[b+2:]

        cv2.accumulateWeighted(img, running_average, .2, None)
        diff = DetectorProcess.diffImg(img, running_average.astype(np.uint8))
        contours, hierarchy = cv2.findContours(diff, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        contourAreas = {cv2.contourArea(c):c for c in contours}
        if len(contours) != 0:
          maxarea = max(contourAreas.keys())
          if maxarea > config.VIDEO.AREA_THRESHOLD:
            largestContour = contourAreas[maxarea]
            ccenter, cradius = cv2.minEnclosingCircle(largestContour)
            self.x.value = ccenter[0]
            self.y.value = ccenter[1]

class StreamHandler(tornado.web.RequestHandler):
  def initialize(self, x, y):
    self.x = x
    self.y = y

  @tornado.web.asynchronous
  def get(self):
    while True:
      gen.Task(tornado.ioloop.IOLoop.instance().add_timeout, time.time() + 1)
      self.write(get_percentage_from_pixels(self.x.value, type="width") + ", " + get_percentage_from_pixels(self.y.value, type="height") + "\r\n")
      self.flush()
    self.finish()

class PositionHandler(tornado.web.RequestHandler):
  def initialize(self, x, y):
    self.x = x
    self.y = y

  @tornado.web.asynchronous
  def get(self):
    response = {
        "x": get_percentage_from_pixels(self.x.value, type="width"),
        "y": get_percentage_from_pixels(self.y.value, type="height")
    }
    self.write(json_encode(response))
    self.set_header("Content-Type", "application/json")
    self.finish()

if __name__ == "__main__":
    static_path = os.path.join('./', 'static')
    manager = Manager()
    x = manager.Value('f', 0)
    y = manager.Value('f', 0)
    process = DetectorProcess(x, y)
    handlers = [
      (r'/static/(.*)', tornado.web.StaticFileHandler, {'path': static_path}),
      (r'/', tornado.web.RedirectHandler,
        dict(url="/static/index.html"),
      ),
      # (r"/stream", StreamHandler, dict(x=x, y=y)),
      (r"/position", PositionHandler, dict(x=x, y=y))
    ]
    application1 = tornado.web.Application(handlers)
    application1.listen(8000, address='127.0.0.1')
    application2 = tornado.web.Application(handlers)
    application2.listen(8001, address='127.0.0.1')
    application3 = tornado.web.Application(handlers)
    application3.listen(8002, address='127.0.0.1')
    application4 = tornado.web.Application(handlers)
    application4.listen(8003, address='127.0.0.1')
    process.start()
    tornado.ioloop.IOLoop.instance().start()
