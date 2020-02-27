import cv2
import dlib
import numpy as np
import math

predictorPath = "shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(predictorPath)
faceDetector = dlib.get_frontal_face_detector()


def findLandmarks(im):
    rects = faceDetector(im, 1)

    if len(rects) > 1:
        return "error"
    if len(rects) == 0:
        return "error"
    return np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])


def markLandmarks(im, landmarks):
    image = im.copy()
    for index, point in enumerate(landmarks):
        position = (point[0, 0], point[0, 1])
        cv2.putText(image, str(index), position, fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, fontScale=0.4,
                    color=(0, 0, 255))
        cv2.circle(image, position, 3, color=(0, 255, 255))
    return image


def topLip(landmarks):
    topLipPts = []
    for i in range(50, 53):
        topLipPts.append(landmarks[i])
    for i in range(61, 64):
        topLipPts.append(landmarks[i])
    topLipAllPts = np.squeeze(np.asarray(topLipPts))
    topLipMean = np.mean(topLipPts, axis=0)
    return int(topLipMean[:, 1])


def bottomLip(landmarks):
    bottomLipPts = []
    for i in range(65, 68):
        bottomLipPts.append(landmarks[i])
    for i in range(56, 59):
        bottomLipPts.append(landmarks[i])
    bottomLipAllPts = np.squeeze(np.asarray(bottomLipPts))
    bottomLipMean = np.mean(bottomLipPts, axis=0)
    print(bottomLipMean)
    return int(bottomLipMean[:, 1])


def mouth_open(image):
    landmarks = findLandmarks(image)

    if landmarks == "error":
        return image, 0, 0, 0, 0

    imageWithLandmarks = markLandmarks(image, landmarks)
    top_lip_center = topLip(landmarks)
    bottom_lip_center = bottomLip(landmarks)
    lip_distance = abs(top_lip_center - bottom_lip_center)

    ########## eyebrow #####################
    Brow_x1 = landmarks[21, 0]
    Brow_y1 = landmarks[21, 1]
    Brow_x2 = landmarks[22, 0]
    Brow_y2 = landmarks[22, 1]
    BrowEndPointDistance = math.sqrt(pow(Brow_x2 - Brow_x1, 2) + pow(Brow_y2 - Brow_y1, 2))

    ############### left eyebrow and eye dist ###########
    left_eye_brow_x1 = landmarks[37, 0]
    left_eye_brow_y1 = landmarks[37, 1]
    left_eye_brow_x2 = landmarks[19, 0]
    left_eye_brow_y2 = landmarks[19, 1]
    LeftEye_Brow_Distance = math.sqrt(
        pow(left_eye_brow_x2 - left_eye_brow_x1, 2) + pow(left_eye_brow_y2 - left_eye_brow_y1, 2))

    ############### right eyebrow and eye dist ###########
    right_eye_brow_x1 = landmarks[44, 0]
    right_eye_brow_y1 = landmarks[44, 1]
    right_eye_brow_x2 = landmarks[24, 0]
    right_eye_brow_y2 = landmarks[24, 1]
    RightEye_Brow_Distance = math.sqrt(
        pow(right_eye_brow_x2 - right_eye_brow_x1, 2) + pow(right_eye_brow_y2 - right_eye_brow_y1, 2))

    return imageWithLandmarks, lip_distance, BrowEndPointDistance, LeftEye_Brow_Distance, RightEye_Brow_Distance


def detectSmile(image):
    landmarks = findLandmarks(image)

    if landmarks == "error":
        return image, 0, 0

    imageWithLandmarks = markLandmarks(image, landmarks)

    pointx1 = landmarks[48, 0]
    pointy1 = landmarks[48, 1]
    pointx2 = landmarks[54, 0]
    pointy2 = landmarks[54, 1]
    lipEndPointDistance = math.sqrt(pow(pointx2 - pointx1, 2) + pow(pointy2 - pointy1, 2))

    ############ mouth open #######
    top_lip_center = topLip(landmarks)
    bottom_lip_center = bottomLip(landmarks)
    lip_distance = abs(top_lip_center - bottom_lip_center)

    return imageWithLandmarks, lipEndPointDistance, lip_distance


def detectAnger(image):
    landmarks = findLandmarks(image)
    if landmarks == "error":
        return image, 0, 0, 0, 0, 0, 0, 0, 0, 0

    imageWithLandmarks = markLandmarks(image, landmarks)

    ############# lips #####################
    pointx1 = landmarks[48, 0]
    pointy1 = landmarks[48, 1]
    pointx2 = landmarks[54, 0]
    pointy2 = landmarks[54, 1]
    lipEndPointDistance = math.sqrt(pow(pointx2 - pointx1, 2) + pow(pointy2 - pointy1, 2))

    ########## eyebrow #####################
    Brow_x1 = landmarks[21, 0]
    Brow_y1 = landmarks[21, 1]
    Brow_x2 = landmarks[22, 0]
    Brow_y2 = landmarks[22, 1]
    BrowEndPointDistance = math.sqrt(pow(Brow_x2 - Brow_x1, 2) + pow(Brow_y2 - Brow_y1, 2))

    ############## left eye ###################
    left_eye_x1 = landmarks[37, 0]
    left_eye_y1 = landmarks[37, 1]
    left_eye_x2 = landmarks[41, 0]
    left_eye_y2 = landmarks[41, 1]
    LeftEyePointDistance = math.sqrt(pow(left_eye_x2 - left_eye_x1, 2) + pow(left_eye_y2 - left_eye_y1, 2))

    ############### right eye ###################
    right_eye_x1 = landmarks[44, 0]
    right_eye_y1 = landmarks[44, 1]
    right_eye_x2 = landmarks[46, 0]
    right_eye_y2 = landmarks[46, 1]
    RightEyePointDistance = math.sqrt(pow(right_eye_x2 - right_eye_x1, 2) + pow(right_eye_y2 - right_eye_y1, 2))

    ############## nose #########################
    leftNosePoint_x1 = landmarks[31, 0]
    leftNosePoint_y1 = landmarks[31, 1]
    rightNosePoint_x1 = landmarks[35, 0]
    rightNosePoint_y1 = landmarks[35, 1]

    ############ mouth open #######
    top_lip_center = topLip(landmarks)
    bottom_lip_center = bottomLip(landmarks)
    lip_distance = abs(top_lip_center - bottom_lip_center)

    return imageWithLandmarks, lipEndPointDistance, BrowEndPointDistance, LeftEyePointDistance, RightEyePointDistance, leftNosePoint_x1, leftNosePoint_y1, rightNosePoint_x1, rightNosePoint_y1, lip_distance


def detectSuprise(image):
    landmarks = findLandmarks(image)
    if landmarks == "error":
        return image, 0, 0, 0, 0, 0, 0

    imageWithLandmarks = markLandmarks(image, landmarks)
    ############## left eye ###################
    left_eye_x1 = landmarks[37, 0]
    left_eye_y1 = landmarks[37, 1]
    left_eye_x2 = landmarks[41, 0]
    left_eye_y2 = landmarks[41, 1]
    LeftEyePointDistance = math.sqrt(pow(left_eye_x2 - left_eye_x1, 2) + pow(left_eye_y2 - left_eye_y1, 2))

    ############### right eye ###################
    right_eye_x1 = landmarks[44, 0]
    right_eye_y1 = landmarks[44, 1]
    right_eye_x2 = landmarks[46, 0]
    right_eye_y2 = landmarks[46, 1]
    RightEyePointDistance = math.sqrt(pow(right_eye_x2 - right_eye_x1, 2) + pow(right_eye_y2 - right_eye_y1, 2))

    ############### left eyebrow and eye dist ###########
    left_eye_brow_x1 = landmarks[37, 0]
    left_eye_brow_y1 = landmarks[37, 1]
    left_eye_brow_x2 = landmarks[19, 0]
    left_eye_brow_y2 = landmarks[19, 1]
    LeftEye_Brow_Distance = math.sqrt(
        pow(left_eye_brow_x2 - left_eye_brow_x1, 2) + pow(left_eye_brow_y2 - left_eye_brow_y1, 2))

    ############### right eyebrow and eye dist ###########
    right_eye_brow_x1 = landmarks[44, 0]
    right_eye_brow_y1 = landmarks[44, 1]
    right_eye_brow_x2 = landmarks[24, 0]
    right_eye_brow_y2 = landmarks[24, 1]
    RightEye_Brow_Distance = math.sqrt(
        pow(right_eye_brow_x2 - right_eye_brow_x1, 2) + pow(right_eye_brow_y2 - right_eye_brow_y1, 2))

    ############ mouth open #######
    top_lip_center = topLip(landmarks)
    bottom_lip_center = bottomLip(landmarks)
    lip_distance = abs(top_lip_center - bottom_lip_center)

    pointx1 = landmarks[48, 0]
    pointy1 = landmarks[48, 1]
    pointx2 = landmarks[54, 0]
    pointy2 = landmarks[54, 1]
    lipEndPointDistance = math.sqrt(pow(pointx2 - pointx1, 2) + pow(pointy2 - pointy1, 2))

    return imageWithLandmarks, LeftEyePointDistance, RightEyePointDistance, LeftEye_Brow_Distance, RightEye_Brow_Distance, lip_distance, lipEndPointDistance


def detectSad(image):
    landmarks = findLandmarks(image)
    if landmarks == "error":
        return image, 0, 0, 0, 0, 0

    imageWithLandmarks = markLandmarks(image, landmarks)
    pointx1 = landmarks[48, 0]
    pointy1 = landmarks[48, 1]
    pointx2 = landmarks[54, 0]
    pointy2 = landmarks[54, 1]
    lipEndPointDistance = math.sqrt(pow(pointx2 - pointx1, 2) + pow(pointy2 - pointy1, 2))

    ############## left eye ###################
    left_eye_x1 = landmarks[37, 0]
    left_eye_y1 = landmarks[37, 1]
    left_eye_x2 = landmarks[41, 0]
    left_eye_y2 = landmarks[41, 1]
    LeftEyePointDistance = math.sqrt(pow(left_eye_x2 - left_eye_x1, 2) + pow(left_eye_y2 - left_eye_y1, 2))

    ############### right eye ###################
    right_eye_x1 = landmarks[44, 0]
    right_eye_y1 = landmarks[44, 1]
    right_eye_x2 = landmarks[46, 0]
    right_eye_y2 = landmarks[46, 1]
    RightEyePointDistance = math.sqrt(pow(right_eye_x2 - right_eye_x1, 2) + pow(right_eye_y2 - right_eye_y1, 2))

    ########## eyebrow #####################
    Brow_x1 = landmarks[21, 0]
    Brow_y1 = landmarks[21, 1]
    Brow_x2 = landmarks[22, 0]
    Brow_y2 = landmarks[22, 1]
    BrowEndPointDistance = math.sqrt(pow(Brow_x2 - Brow_x1, 2) + pow(Brow_y2 - Brow_y1, 2))

    ############ mouth open #######
    top_lip_center = topLip(landmarks)
    bottom_lip_center = bottomLip(landmarks)
    lip_distance = abs(top_lip_center - bottom_lip_center)

    return imageWithLandmarks, lipEndPointDistance, LeftEyePointDistance, RightEyePointDistance, BrowEndPointDistance, lip_distance


def NeutralFace(image):
    landmarks = findLandmarks(image)
    if landmarks == "error":
        return image, 0, 0, 0, 0, 0, 0, 0

    ################ lips distance for smile ##############
    imageWithLandmarks = markLandmarks(image, landmarks)
    pointx1 = landmarks[48, 0]
    pointy1 = landmarks[48, 1]
    pointx2 = landmarks[54, 0]
    pointy2 = landmarks[54, 1]
    neutralLipEndPointDistance = math.sqrt(pow(pointx2 - pointx1, 2) + pow(pointy2 - pointy1, 2))

    ################ eyebrows ##############
    eyebrow_x1 = landmarks[22, 0]
    eyebrow_y1 = landmarks[22, 1]
    eyebrow_x2 = landmarks[21, 0]
    eyebrow_y2 = landmarks[21, 1]
    BrowEndPointDistance = math.sqrt(pow(eyebrow_x2 - eyebrow_x1, 2) + pow(eyebrow_y2 - eyebrow_y1, 2))

    ############### left eye ###################
    left_eye_x1 = landmarks[37, 0]
    left_eye_y1 = landmarks[37, 1]
    left_eye_x2 = landmarks[41, 0]
    left_eye_y2 = landmarks[41, 1]
    LeftEyePointDistance = math.sqrt(pow(left_eye_x2 - left_eye_x1, 2) + pow(left_eye_y2 - left_eye_y1, 2))

    ############### right eye ###################
    right_eye_x1 = landmarks[44, 0]
    right_eye_y1 = landmarks[44, 1]
    right_eye_x2 = landmarks[46, 0]
    right_eye_y2 = landmarks[46, 1]
    RightEyePointDistance = math.sqrt(pow(right_eye_x2 - right_eye_x1, 2) + pow(right_eye_y2 - right_eye_y1, 2))

    ############### left eyebrow and eye dist ###########
    left_eye_brow_x1 = landmarks[37, 0]
    left_eye_brow_y1 = landmarks[37, 1]
    left_eye_brow_x2 = landmarks[19, 0]
    left_eye_brow_y2 = landmarks[19, 1]
    LeftEye_Brow_Distance = math.sqrt(
        pow(left_eye_brow_x2 - left_eye_brow_x1, 2) + pow(left_eye_brow_y2 - left_eye_brow_y1, 2))

    ############### right eyebrow and eye dist ###########
    right_eye_brow_x1 = landmarks[44, 0]
    right_eye_brow_y1 = landmarks[44, 1]
    right_eye_brow_x2 = landmarks[24, 0]
    right_eye_brow_y2 = landmarks[24, 1]
    RightEye_Brow_Distance = math.sqrt(
        pow(right_eye_brow_x2 - right_eye_brow_x1, 2) + pow(right_eye_brow_y2 - right_eye_brow_y1, 2))

    ############ mouth open #######
    top_lip_center = topLip(landmarks)
    bottom_lip_center = bottomLip(landmarks)
    Orglip_distance = abs(top_lip_center - bottom_lip_center)

    return imageWithLandmarks,neutralLipEndPointDistance, BrowEndPointDistance, LeftEyePointDistance, RightEyePointDistance, LeftEye_Brow_Distance, RightEye_Brow_Distance, Orglip_distance


cap = cv2.VideoCapture(0)
i = 0
output_text2=''
while True:

    ret, frame = cap.read()
    # capture for initialization --neutral face
    if i == 0:
        orgImageWithLandmarks,originalLipEndPoint_Distance, orig_BrowEndPointDistance, orig_LeftEyeDist, orig_RightEyeDist, orig_LeftEye_Brow_Dist, orig_RightEye_Brow_Dist, orig_lip_distance = NeutralFace(
            frame)
        i = 1
    else:
        flag=False
        flag1=False
        output_text2=''

        ### Smile Detection ###

        imageLandmarks, Smile_lipEndPoint_Distance, Smile_lip_distance = detectSmile(frame)

        if (Smile_lipEndPoint_Distance > originalLipEndPoint_Distance) and (
        abs(Smile_lipEndPoint_Distance - originalLipEndPoint_Distance)) > 5:
            output_text2 = "Smile Detected"
            x_offset = y_offset = 0
            p, q, r = frame.shape
            cv2.putText(frame, output_text2, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 127), 2)

            print("Smile Detection")

        ### Surprise Detection ###

        imageLandmarks, LeftEyePointDistance, RightEyePointDistance, LeftEye_Brow_Distance, RightEye_Brow_Distance, lip_distance, lipEndPointDistance = detectSuprise(
            frame)

        if (abs(orig_LeftEye_Brow_Dist - LeftEye_Brow_Distance)) > 1.5 and (
                (abs(orig_RightEye_Brow_Dist - RightEye_Brow_Distance)) > 1.5
        and (abs(orig_LeftEyeDist - LeftEyePointDistance)) > 1) and (
            abs(orig_RightEyeDist - RightEyePointDistance) > 1) and lip_distance > 15 and (lipEndPointDistance - originalLipEndPoint_Distance)<0 :

            output_text2 = " Suprise Detected"
            cv2.putText(frame, output_text2, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 127), 2)
            flag1=True
            print("Surprise Detection")

        ### Anger Detection ###
        imageLandmarks, lipEndPointDistance, BrowEndDist, AngerLeftEyeDist, AngerRightEyeDist, leftNosePoint_x1, leftNosePoint_y1, rightNosePoint_x1, rightNosePoint_y1, Angerlip_distance = detectAnger(
            frame)

        if (BrowEndDist < orig_BrowEndPointDistance and AngerLeftEyeDist < orig_LeftEyeDist and AngerRightEyeDist < orig_RightEyeDist  and lipEndPointDistance <= originalLipEndPoint_Distance) or \
        (BrowEndDist < orig_BrowEndPointDistance and AngerLeftEyeDist < orig_LeftEyeDist and AngerRightEyeDist < orig_RightEyeDist and lip_distance > orig_lip_distance):
            output_text2 = "Anger Detected"
            cv2.putText(frame, output_text2, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 127), 2)
            flag=True

            print("Anger Detection")


        ### Sad Detection ###
        imageLandmarks, SadlipEndPointDistance, SadLeftEyePointDistance, SadRightEyePointDistance, SadBrowEndPointDistance, lip_distance = detectSad(
            frame)
        if flag== False:
            if (SadlipEndPointDistance < originalLipEndPoint_Distance and SadBrowEndPointDistance < orig_BrowEndPointDistance and lip_distance <= orig_lip_distance) or \
                    (SadLeftEyePointDistance < orig_LeftEyeDist and SadRightEyePointDistance < orig_RightEyeDist and lip_distance <= orig_lip_distance) or \
                    (SadLeftEyePointDistance == 0 and SadRightEyePointDistance == 0 and lip_distance <= orig_lip_distance):
                output_text2 = "Sad Detected"
                cv2.putText(frame, output_text2, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 127), 2)
                print("Sad Detection")

        ### Yawning Detection ###
        imageLandmarks, lip_distance, BrowEndDist, LeftEye_Brow_Distance, RightEye_Brow_Distance = mouth_open(frame)
        if flag1==False:
            if (lip_distance > 25 and BrowEndDist >= orig_BrowEndPointDistance) and (
            LeftEye_Brow_Distance <= orig_LeftEye_Brow_Dist) and (RightEye_Brow_Distance <= orig_RightEye_Brow_Dist):
                output_text2 = "Yawn Detected"
                cv2.putText(frame, output_text2, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 127), 2)
                print("Yawn Detection")



        cv2.imshow('Live Landmarks', imageLandmarks)
        cv2.imshow('Emotion Detection', frame)
        i = 1

    ### 13- for enter key ######
    if cv2.waitKey(1) == 13:
        break

cap.release()
cv2.destroyAllWindows()
