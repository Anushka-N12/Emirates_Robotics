from flask import Flask, make_response, Response
from ultralytics import YOLO
import cam
import cv2, json

esp_cam = ''

app = Flask(__name__) 

c_model = YOLO("best.pt")
c_model.to('cuda')
c_classes = {0: 'cardboard', 1: 'black_garbage_bag', 2: 'glass_bottle', 3: 'metal', 4: 'paper', 5: 'plastic_bag', 6: 'plastic_bottle', 7: 'toy_car', 8: 'wrapper', 9: 'hammer', 10: 'screwdriver', 11: 'wrench'}
pt_classes = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}

pt_model = YOLO("YOLOv8n.pt")
pt_model.to('cuda')

@app.route('/pred', methods=['GET'])
def pred():
    # Take image
    print(cam.getimg(esp_cam))
    # Run through custom model 
    c_results = c_model('frame.jpg')
    # Filter by size
    objs = []
    for result in c_results:
        result = result.cpu().boxes.numpy()
        if len(result.cls) > 0 and result.conf > 0.6:  # If detected, with decent confidence
            # print(result)
            [x1,y1,x2,y2] = result.xyxy[0]
            og_shape = result.orig_shape

            # Confirm with pre-trained model
            passed = True
            image = cv2.imread("frame.jpg")
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Draw a rectangle around obj
            cropped_image = image[y1:y2, x1:x2]    # Crop the image to the rectangle
            cv2.imwrite("cropped.jpg", cropped_image)    # Save in       
            cr_results = pt_model("cropped.jpg")
            for result in cr_results:
                result = result.cpu().boxes.numpy()
                if pt_classes[result.cls[0]] in ['person', 'bicycle', 'fire hydrant', 'stop sign', 'bench', 'bird', 'cat', 'dog', 'chair', 'potted plant', 'dining table']:
                    passed = False

            if passed == True and x2-x1 < og_shape[1]*0.75 and y2-y1 < og_shape[0]*0.75:  # Making sure obj is not too big
                objs.append(((x2+x1)//2, (y2+y1)//2))  # Adding center coords of object to list

    # Add pt inference
    pt_results = pt_model('frame.jpg')
    for result in pt_results:
        result = result.cpu().boxes.numpy()
        [x1,y1,x2,y2] = result.xyxy[0]
    objs.append(((x2+x1)//2, (y2+y1)//2))

    # Choose closest object
    if len(objs) > 0:    
        closest = (objs[0])
        # ideal = (og_shape[1]//2, og_shape[0]*0.75)
        for i in objs:
            if i[1] < closest[1]:
                closest = i
        print(closest)
    else:
        closest = (-1, -1)

    # Send result
    # Create a JSON response and set custom header
    d = {'coords': closest}
    response = json.dumps([d])
    response = Response(response, status=200, content_type='application/json')
    response.headers['X-My-Header'] = 'foo'
    return response, 200

if __name__ == '__main__':
    app.run(debug=True)
    # print(model.names)

# results = model(r'C:\Users\anush\Projects\Emirates_Robotics\venv\data\test\trash.mp4', show=True)
