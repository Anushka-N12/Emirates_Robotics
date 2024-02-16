# Script to get & save images from url
# https://www.digikey.com/en/maker/projects/esp32-cam-python-stream-opencv-example/840608badd0f4a5eb21b1be25ecb42cb

# Importing required packages
import cv2
import numpy as np
import requests

def set_resolution(url: str, index: int=1, verbose: bool=False):
    try:
        if verbose:
            resolutions = "10: UXGA(1600x1200)\n9: SXGA(1280x1024)\n8: XGA(1024x768)\n7: SVGA(800x600)\n6: VGA(640x480)\n5: CIF(400x296)\n4: QVGA(320x240)\n3: HQVGA(240x176)\n0: QQVGA(160x120)"
            print("available resolutions\n{}".format(resolutions))

        if index in [10, 9, 8, 7, 6, 5, 4, 3, 0]:
            requests.get(url + "/control?var=framesize&val={}".format(index))
        else:
            print("Wrong index")
    except:
        print("SET_RESOLUTION: something went wrong")

def set_quality(url: str, value: int=1, verbose: bool=False):
    try:
        if value >= 10 and value <=63:
            requests.get(url + "/control?var=quality&val={}".format(value))
    except:
        print("SET_QUALITY: something went wrong")

def set_awb(url: str, awb: int=1):
    try:
        awb = not awb
        requests.get(url + "/control?var=awb&val={}".format(1 if awb else 0))
    except:
        print("SET_QUALITY: something went wrong")
    return awb

# Function used by app.py to get image
def getimg(url):
    cap = cv2.VideoCapture(url + ":81/stream")
    set_resolution(url, index=8)
    n,x = 0,[]
    if True:
        try:
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 0)
                ret, frame = cap.read()
                if ret:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    gray = cv2.equalizeHist(gray)
                n += 1
                cv2.imwrite("frame.jpg", frame)  
                # frame.set(5,640)
                # frame.set(6,480)
        except:
            pass

    # cv2.destroyAllWindows()
    cap.release()
    return 'Image saved.'

# Test to execute only if this file is executed directly
if __name__ == '__main__':
    print(getimg('http://10.5.35.138'))

