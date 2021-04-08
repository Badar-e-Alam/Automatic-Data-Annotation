# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import glob
import cv2
import numpy as np
import argparse
import tqdm

def bounding_box(img_pth, lbl_path):
    for i in range(len(img_pth)):
        image = cv2.imread(img_pth[i])
        height, width, _ = image.shape
        print(lbl_path[i] + "AND" + img_pth[i])

        lbl = open(lbl_path[i], 'r')
        boxes = []
        for i in lbl:
            clean = i.split()

            boxes.append(clean[1:])
        for box in boxes:
            x, y, w, h = box
            w = int(float(w) * width)
            h = int(float(h) * height)
            x = int((float(x) * width) - (w / 2))
            y = int((float(y) * height) - (h / 2))
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 1)
        cv2.namedWindow("enhanced", 0)
        cv2.resizeWindow("enhanced", 1920, 1080)
        cv2.imshow("enhanced", image)
        cv2.waitKey(500)
        cv2.destroyAllWindows()

def load_vide(index):
    print("function shift")
    vid_pth = 'videos/'
    vid_lis = glob.glob(vid_pth + '*.mp4')
    print("index number ___________________________________", index)

    print("printing ",vid_lis[index])
    cap = cv2.VideoCapture(vid_lis[index])
    if not cap.isOpened():
        print("Error In loading Image")
    return cap


def frame_by_frame(vid_pth):
    # Create a VideoCapture object and read from input file
    # If the input is the camera, pass 0 instead of the video file name
    vid_lis = glob.glob(vid_pth + '*.mp4')
    count = 0;
    count1=0
    index = 0;
    i = 0;
    frame1=0
    flage = False;
    forwd = 14400;
    lenght = len(vid_lis)
    cap=[]
    cap = cv2.VideoCapture(vid_lis[index])
    if not cap.isOpened():
        print("Error In loading Image")

    while True:
        if  index<lenght:
            if flage :
                index += 1
                print("index number ___________________________________",index)
                # cv2.destroyAllWindows()
                cap = load_vide(index)
                flage=False
        else:
            print("NOthing left ")
            break

        frame1+=1
        ret, frame = cap.read()
        if index==0:
            if ret and frame1 == 72:
                # Display the resulting frame
                cv2.imwrite('train/%s.jpg' % "{0:010d}".format(count1), frame)
                count1 += 1
                frame1 = 0  # Press Q on keyboard to  exit
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break
        else:
            if ret and frame1 == 63:
                # Display the resulting frame
                cv2.imwrite('train/%s.jpg' % "{0:010d}".format(count1), frame)
                count1 += 1
                frame1 = 0  # Press Q on keyboard to  exit
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break

        count += 1
        if count >= forwd:
            flage = True
            forwd += 14400
            print("Caputring next Video")


            if forwd==43200:
                break


    # cap = cv2.VideoCapture(vid_pth)
    # # Check if camera opened successfully
    # if (cap.isOpened() == False):
    #     print("Error opening video stream or file")
    # # Read until video is completed
    # count=0
    # while (cap.isOpened()):
    #     # Capture frame-by-frame
    #     ret, frame = cap.read()
    #     if ret == True:
    #         # Display the resulting frame
    #         cv2.imshow('Frame', frame)
    #         # Press Q on keyboard to  exit
    #         if cv2.waitKey(25) & 0xFF == ord('q'):
    #             break
    #     # Break the loop
    #     else:
    #         break
    # # When verything done, release the video capture object
    # cap.release()
    # # Closes all the frames
    # cv2.destroyAllWindows()
    
def predection(imagepth):
    count=0
    for img in tqdm.tqdm(imagepth):
        txt_name = str(str(img.replace('Frames/', 'Labels/')).replace('.jpg', '.txt').replace('jpg', 'txt'))
        txt_name = 'SCALED_LBL/' + "{0:010d}".format(count) + ".txt"
        image= cv2.imread(img)
        Width = image.shape[1]
        Height = image.shape[0]
        scale = 0.00392
        with open('coco.names', 'r') as f:
            classes = [line.strip() for line in f.readlines()]
        COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
            # read pre-trained model and config file
        net = cv2.dnn.readNet('yolov4-csp.weights', 'yolov4-x.cfg')
            # create input blob
        blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)

            # set input blob for the network
        net.setInput(blob)
        layer_names = net.getLayerNames()

        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

        outs = net.forward(output_layers)

            # initialization
        class_ids = []
        confidences = []
        boxes = []
        conf_threshold = 0.5
        nms_threshold = 0.4

            # for each detetion from each output layer
            # get the confidence, class id, bounding box params
            # and ignore weak detections (confidence < 0.5)
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.2:
                    center_x = int(detection[0] * Width)
                    center_y = int(detection[1] * Height)
                    w = int(detection[2] * Width)
                    h = int(detection[3] * Height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append(detection[0:4])
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
        indices = np.reshape(indices, (-1))
        boxes1=[]
        for i in indices:
            boxes1.append(boxes[i])

        with open(txt_name,'w') as f:
            for box in boxes1:
                line = '0 ' + str(box[0]) + ' ' + str(box[1]) + ' ' + str(box[2]) + ' ' + str(box[3])+"\n"
                f.write(line)
        count += 1
        # print(indexes)
        for i in range(len(boxes)):
            if i in indices:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                print(type(x))
                cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 15)
                cv2.putText(image, label, (int(x), int(y) + 10), cv2.FONT_HERSHEY_PLAIN, 6, (0, 0, 255), 8, cv2.LINE_AA)

            # display output image
        cv2.namedWindow("enhanced", 0)
        cv2.resizeWindow("enhanced", 1920, 1080)
        cv2.imshow("enhanced", image)
        cv2.waitKey(500)
        cv2.destroyAllWindows()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='img.jpg', help="Path for input image (optional)")
    parser.add_argument('-o', '--output', type=str, default='out.jpg', help="path to output image (optional)")
    parser.add_argument('-c', '--confidence', type=float, default=0.4, help='minimum confidence value')
    parser.add_argument('-t', '--threshold', type=float, default=0.4, help="Threshold for non maxima spression")
    parser.add_argument('-v', '--video_path', type=str, default='video_f_100/', help='Videos path optional')
    args = parser.parse_args()
    print(args.input)
    frame_by_frame('videos/')
    #img_path=sorted( glob.glob('100_farms/*.jpg'))
    # pth_lbl=sorted(glob.glob('lbl_100/*.txt'))
    # bounding_box(img_path,pth_lbl)
    #predection(img_path)
    #
    # # cv2.imshow('img',predection(cv2.imread('img.png')))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

