import cv2
import mediapipe as mp
import pyautogui
import time


cam = cv2.VideoCapture(0)
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
mp_drawing = mp.solutions.drawing_utils
screen_w, screen_h = pyautogui.size()


mouth_opened = False
start_time = None


screen_width, screen_height = pyautogui.size()
min_scale_x = 2
max_scale_x = 5
min_scale_y = 2
max_scale_y = 5
scale_factor_x = 1.4
scale_factor_y = 1.9


while True:
    _, frame = cam.read()
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = face_mesh.process(rgb_frame)
    landmark_points = output.multi_face_landmarks
    frame_h, frame_w, _ = frame.shape


    if landmark_points:
        landmarks = landmark_points[0].landmark


        lip_upper = int(landmarks[13].y * frame_h)
        lip_lower = int(landmarks[14].y * frame_h)
        lip_distance = lip_lower - lip_upper


        if lip_distance > 10:
            if not mouth_opened:
                start_time = time.time()
            mouth_opened = True
        else:
            if mouth_opened:
                end_time = time.time()
                if end_time - start_time < 0.5:
                    pyautogui.click()
                elif end_time - start_time > 1.0:
                    pyautogui.rightClick()
            mouth_opened = False


        left_eye_closed = landmarks[159].y > landmarks[145].y
        right_eye_closed = landmarks[386].y > landmarks[374].y


        if left_eye_closed and right_eye_closed:
            pyautogui.click()


        nose = landmark_points[0].landmark[6]
        x = int(nose.x * screen_width * scale_factor_x) - 1100
        y = int(nose.y * screen_height * scale_factor_y) - 550


        scale_factor_x = max(min_scale_x, min(max_scale_x, scale_factor_x))
        scale_factor_y = max(min_scale_y, min(max_scale_y, scale_factor_y))


        pyautogui.moveTo(x, y)

    cv2.imshow('Control Combinat', frame)
    cv2.waitKey(1)
    k = cv2.waitKey(30)
    if k == 27:
        break
    

cam.release()
cv2.destroyAllWindows()