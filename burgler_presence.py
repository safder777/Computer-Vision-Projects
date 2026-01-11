# import cv2

# cap=cv2.VideoCapture('Burglary.mp4')

# frames=[]

# gap=5

# count=0

# while True:
#     ret,frame=cap.read()

#     if not ret:
#         break

#     gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
#     frames.append(gray)

#     if len(frames)>gap+1:
#         frames.pop(0)

#     cv2.putText(frame,f"Frame Count:{count}",(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

#     if len(frames)>gap:
#         diff=cv2.absdiff(frames[0],frames[-1])
#         _, thresh=cv2.threshold(thresh,30,255,cv2.THRESH_BINARY)

#         contours, _=cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

#         for c in contours:
#             if cv2.contourArea(c)<1000:
#                 continue
#             x,y,w,h =cv2.boundingRect(c)
#             cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

#     # count +=1
#     # cv2.imshow("Frame",frame)
#     # cv2.imshow("Frame",frame)

#         motion=any(cv2.contourArea(c)>1000 for c in contours) #discarding those contours whose area in less than 1000

#         if motion:
#             cv2.putText(frame,"Motion Detected!",(10,60),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
#             cv2.imwrite(f"motion_frame_{count}.jpg",frame)
#             print(f"Saved:motion_frame_{count}.jpg")

#         cv2.imshow("Motion Detectection",frame)
#         count+=1

#         if cv2.waitKey(1) & 0xFF==27:
#             break

# cap.release()
# cv2.destroyAllWindows()

#(video link) https://www.youtube.com/watch?v=8FG-wBXs0Es&list=PLPTV0NXA_ZSgmWYoSpY_2EJzPJjkke4Az&index=21
import cv2
cap = cv2.VideoCapture('Burglary.mp4') #put zero for camera
if not cap.isOpened():
    print("[Error] Could not open video file 'Burglary.mp4'")
    raise SystemExit

frames = []     # rolling buffer of grayscale frames
gap = 5         # compare frames 'gap' apart
count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale and denoise (helps contour stability)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    frames.append(gray)
    # Keep buffer at most 'gap' frames; we'll compare oldest vs newest
    if len(frames) > gap:
        frames.pop(0)

    cv2.putText(frame, f"Frame Count: {count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    contours = []  # ensure defined even if we skip motion logic

    if len(frames) == gap:
        diff = cv2.absdiff(frames[0], frames[-1])

        # Threshold: choose values based on your footage’s lighting/noise
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

        # Optional: connect small regions
        thresh = cv2.dilate(thresh, None, iterations=2)

        # Find motion contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            if cv2.contourArea(c) < 1000:
                continue
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Evaluate motion only when contours exist
    motion = any(cv2.contourArea(c) > 1000 for c in contours) if contours else False


    
    N = 60  # save 1 out of every 60 frames if motion is present
    if motion and (count % N == 0):

        if motion:
            cv2.putText(frame, "Motion Detected!", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imwrite(f"motion_frame_{count}.jpg", frame)
            print(f"Saved: motion_frame_{count}.jpg")

    cv2.imshow("Motion Detection", frame)
    count += 1

    # ESC to quit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
