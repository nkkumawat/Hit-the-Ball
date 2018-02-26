import cv2
import random
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
path = "/home/sonu/.local/lib/python3.6/site-packages/cv2/data/"
palm_cascade = cv2.CascadeClassifier(path + 'palm.xml')
ball_images = ['ball.png' , 'ball1.png' , 'ball2.png']
ball_image_index = 0
x_balls =[]
y_balls = []
score = 0
ball_length = 20
balls =[]
for i in range(0 , ball_length):
    x_balls.insert(len(x_balls) , random.randint(50 , 500))
    y_balls.insert(len(y_balls) , 0)
    balls.insert(len(balls) , ball_image_index)
    ball_image_index += 1
    ball_image_index %= len(ball_images)
while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = palm_cascade.detectMultiScale(gray, 1.3, 5)
    for i in range(0 , ball_length):
        add = random.randint(1 , 5)
        s_img = cv2.imread(ball_images[balls[i]], -1)
        s_img = cv2.resize(s_img, (30, 30))
        alpha_s = s_img[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s
        if y_balls[i] + s_img.shape[1] +add < frame.shape[0] :
            y_balls[i] += add
        else:
            y_balls[i] = 0
            x_balls[i] = random.randint(50 , 500)
            score -= 1
        y1, y2 = y_balls[i], y_balls[i] + s_img.shape[0]
        x1, x2 = x_balls[i], x_balls[i] + s_img.shape[1]
        for c in range(0, 3):
            frame[y1:y2, x1:x2, c] = (alpha_s * s_img[:, :, c] + alpha_l * frame[y1:y2, x1:x2, c])
    cv2.putText(frame, "Total Score : " + str(score), (19, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, 255 ,3)
    for (x,y,w,h) in faces:
        for i in range(0, ball_length):
            if x_balls[i] >= x and x_balls[i] <= x + w and y_balls[i] >= y and y_balls[i] <= y +h :
                y_balls[i] = 0
                score += 1
        # cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.imshow('frame',frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
