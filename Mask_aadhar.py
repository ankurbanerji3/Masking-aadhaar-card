import cv2
from PIL import Image
import pytesseract
from pytesseract import Output
import re
import os
import math

class Aadhaar_Card():
  def __init__(self,config = {'orient' : True,'skew' : True,'crop': True,'contrast' : True,'psm': [3,4,6],'mask_color': (0, 165, 255), 'brut_psm': [6]}):
    self.config = config
  def validate(self,aadhaarNum):
    mult = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 0, 6, 7, 8, 9, 5], [2, 3, 4, 0, 1, 7, 8, 9, 5, 6], [3, 4, 0, 1, 2, 8, 9, 5, 6, 7], [4, 0, 1, 2, 3, 9, 5, 6, 7, 8], [5, 9, 8, 7, 6, 0, 4, 3, 2, 1], [6, 5, 9, 8, 7, 1, 0, 4, 3, 2], [7, 6, 5, 9, 8, 2, 1, 0, 4, 3], [8, 7, 6, 5, 9, 3, 2, 1, 0, 4], [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]]
    perm = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 5, 7, 6, 2, 8, 3, 0, 9, 4], [5, 8, 0, 3, 7, 9, 6, 1, 4, 2], [8, 9, 1, 6, 0, 4, 3, 5, 2, 7], [9, 4, 5, 3, 1, 2, 6, 8, 7, 0], [4, 2, 8, 6, 5, 7, 3, 9, 0, 1], [2, 7, 9, 3, 8, 0, 6, 4, 1, 5], [7, 0, 4, 6, 9, 1, 3, 2, 5, 8]]
    try:
      i = len(aadhaarNum)
      j = 0
      x = 0
      while i > 0:
        i -= 1
        x = mult[x][perm[(j % 8)][int(aadhaarNum[i])]]
        j += 1
      if x == 0:
        return 1
      else:
        return 0
    except ValueError:
      return 0
    except IndexError:
      return 0
  def extract(self, path):
    self.image_path = path
    self.read_image_cv()
    if self.config['orient']:
      self.cv_img = self.rotate(self.cv_img)
    if self.config['skew']:
      print("skewness correction not available")
    if self.config['crop']:
      print("Smart Crop not available")
    if self.config['contrast']:
      self.cv_img  = self.contrast_image(self.cv_img)
      print("correcting contrast")
    aadhaars = set()
    for i in range(len(self.config['psm'])):
      t = self.text_extractor(self.cv_img,self.config['psm'][i])
      anum = self.is_aadhaar_card(t)
      uid = self.find_uid(t)
      if anum != "Not Found" and len(uid) == 0:
        if len(anum) - anum.count(' ') == 12:
          aadhaars.add(anum.replace(" ", ""))
      if anum == "Not Found" and len(uid) != 0:
        aadhaars.add(uid[0].replace(" ", ""))
      if anum != "Not Found" and len(uid) != 0:
        if len(anum) - anum.count(' ') == 12:
          aadhaars.add(anum.replace(" ", ""))
        aadhaars.add(uid[0].replace(" ", ""))
    return list(aadhaars)
  def mask_image(self, path, write, aadhaar_list):
    self.mask_count = 0
    self.mask = cv2.imread(str(path), cv2.IMREAD_COLOR)
    for j in range(len(self.config['psm'])):
      for i in range(len(aadhaar_list)):
        if (self.mask_aadhaar(aadhaar_list[i],write,self.config['psm'][j]))>0:
          self.mask_count = self.mask_count + 1
    cv2.imwrite(write,self.mask)
    return self.mask_count
  def mask_aadhaar(self, uid, out_path, psm):
    d = self.box_extractor(self.mask, psm)
    n_boxes = len(d['level'])
    color = self.config['mask_color']
    count_of_match = 0
    uid_8 = uid[:8]
    for i in range(n_boxes):
      string = d['text'][i].strip()
      if string.isdigit() and string in uid_8 and len(string)>=2:
        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
        cv2.rectangle(self.mask, (x, y), (x + w, y + h), color, cv2.FILLED)
        count_of_match = count_of_match + 1
      else:
        count_of_match = count_of_match + 0
    return count_of_match
  def read_image_cv(self):
    self.cv_img = cv2.imread(str(self.image_path), cv2.IMREAD_COLOR)
  def mask_nums(self, input_file, output_file):
    img = cv2.imread(str(input_file), cv2.IMREAD_COLOR)
    j = 0
    for i in range(len(self.config['brut_psm'])):
      d = self.box_extractor(img,self.config['brut_psm'][i])
      n_boxes = len(d['level'])
      color = self.config['mask_color']
      for i in range(n_boxes):
        string = d['text'][i].strip()
        if string.isdigit() and len(string)==4 and j<2:
          j = j + 1
          (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
          cv2.rectangle(img, (x, y), (x + w, y + h), color, cv2.FILLED)
        else:
          j = 0
    cv2.imwrite(output_file,img)
    return "Done"
  def rotate_only(self, img, angle_in_degrees):
    self.img = img
    self.angle_in_degrees = angle_in_degrees
    rotated = ndimage.rotate(self.img, self.angle_in_degrees)
    return rotated
  def is_image_upside_down(self, img):
    self.img = img
    face_locations = face_recognition.face_locations(self.img)
    encodings = face_recognition.face_encodings(self.img, face_locations)
    image_is_upside_down = (len(encodings) == 0)
    return image_is_upside_down
  def rotate(self,img):
    self.img = img
    img_gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
    img_edges = cv2.Canny(img_gray, 100, 100, apertureSize=3)
    lines = cv2.HoughLinesP(img_edges, 1, math.pi / 180.0, 100, minLineLength=100, maxLineGap=5)
    img_copy = self.img.copy()
    for x in range(0, len(lines)):
      for x1,y1,x2,y2 in lines[x]:
        cv2.line(img_copy,(x1,y1),(x2,y2),(0,255,0),2)
    angles = []
    for x1, y1, x2, y2 in lines[0]:
      angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
      angles.append(angle)
    median_angle = np.median(angles)
    img_rotated = self.rotate_only(self.img, median_angle)
    if self.is_image_upside_down(img_rotated):
      print("rotate to 180 degree")
      angle = -180
      img_rotated_final = self.rotate_only(img_rotated, angle)
      if self.is_image_upside_down(img_rotated_final):
        print("Kindly check the uploaded image, face encodings still not found!")
        return img_rotated
      else:
        print("image is now straight")
        return img_rotated_final
    else:
      print("image is straight")
      return img_rotated
  def contrast_image(self, img):
    self.img = img
    gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    return thresh
  def text_extractor(self, img, psm):
    config  = ('-l eng --oem 3 --psm '+ str(psm))
    t = pytesseract.image_to_string(img, lang='eng', config = config)
    return t
  def box_extractor(self, img, psm):
    config  = ('-l eng --oem 3 --psm '+ str(psm))
    t = pytesseract.image_to_data(img, lang='eng', output_type=Output.DICT, config=config)
    return t
  def find_uid(self,text2):
    uid = set()
    try:
      newlist = []
      for xx in text2.split('\n'):
        newlist.append(xx)
      newlist = list(filter(lambda x: len(x) > 12, newlist))
      for no in newlist:
        if re.match("^[0-9 ]+$", no):
          uid.add(no)
    except Exception:
      pass
    return list(uid)
  def is_aadhaar_card(self, text):
    res=text.split()
    aadhaar_number=''
    for word in res:
      if len(word) == 4 and word.isdigit():
        aadhaar_number=aadhaar_number  + word + ' '
    if len(aadhaar_number)>=14:
      return aadhaar_number
    else:
      return "Not Found"

if __name__=='__main__':      
	config = {'orient' : True,   #corrects orientation of image default -> True
        	  'skew' : True,     #corrects skewness of image default -> True
        	  'crop': True,      #crops document out of image default -> True
        	  'contrast' : True, #Bnw for Better OCR default -> True
        	  'psm': [3,4,6],    #Google Tesseract psm modes default -> 3,4,6 
        	  'mask_color': (0, 165, 255),  #Masking color BGR Format
        	  'brut_psm': [6]    #Keep only one for brut mask (6) is good to start
        	  }
	objj = Aadhaar_Card(config)
	aadhaar_list = objj.extract("image_path")
	flag = objj.mask_image("input_image_path", "output_image_path", aadhaar_list)
	#the method mask_image is used if there's only one location of the aadhaar number on the aadhaar card
	flag2 = objj.mask_nums("input_image_path", "output_image_path")
	#the method mask_nums is used if there are multiple locations of the aadhaar number on the aadhaar card

