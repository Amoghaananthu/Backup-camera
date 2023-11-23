import numpy as np
import cv2

#############################################################################################################

#Vehicle parameters
height = 1.4351 #m
width = 1.78054 #m
wheelbase = 2.7 #m
wheel_radius = 0.315 #m
cam_pos = np.array([0, 0.5, 0.5994]) 
front_track_width = 1.4351 #m
rear_track_width = 1.53162 #m
steering_ratio = 18

#############################################################################################################

#Camera calibration
camera_matrix = np.array([[586.85,   0.,         636.59],
                          [  0.,         582.09, 333.92],
                          [  0.,           0.,           1.        ]])

#Rotation Matrix
R1 = np.array([[0, 1, 0],
              [-1, 0, 0],
              [0, 0, 1]])    #Rotation about z-axis by -90deg

R2 = np.array([[0, 0, 1],
              [0, 1, 0],
              [-1, 0, 0]])   #Rotation about y-axis by 90deg

R = np.matmul(R1,R2)

#Transformation Matrix
transformation_matrix = np.zeros([4, 4])

transformation_matrix[:3, :3] = R

transformation_matrix[:3, 3] = cam_pos.T

transformation_matrix[3,3] = 1

#Projection Matrix
P = camera_matrix @ np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]) @ transformation_matrix

#############################################################################################################

#Function forAckermann Steering dynamic trajectory
def calculate_wheel_steer_angles(pinion_angle):
    
    # Calculate ackermann angle
    pinion_angle = np.deg2rad(pinion_angle)
    ackermann_steer_angle = pinion_angle/steering_ratio

    # Calculate left wheel steer angle
    left_wheel_steer_angle = np.arctan((wheelbase * np.tan(ackermann_steer_angle)) / (wheelbase + 0.5 * front_track_width * np.tan(ackermann_steer_angle)))

    # Calculate right wheel steer angle
    right_wheel_steer_angle = np.arctan((wheelbase * np.tan(ackermann_steer_angle)) / (wheelbase - 0.5 * front_track_width * np.tan(ackermann_steer_angle)))

    return left_wheel_steer_angle, right_wheel_steer_angle

#############################################################################################################

#Function to find trajectory using state space equations
def wheel_trajectory(left_wheel_steer_angle, right_wheel_steer_angle, xn, yn, theta_n, num_points):

    us = -1 #m/s
    delta_t = 0.001 #s

    trajectory_points = np.zeros([num_points, 3])
    trajectory_points[0] = np.array([xn, yn, theta_n])

    left_wheel_trajectory = np.zeros([num_points, 4]) 
    right_wheel_trajectory = np.zeros([num_points, 4])

    left_wheel_cam_frame = np.zeros([num_points, 3])
    right_wheel_cam_frame = np.zeros([num_points, 3])

    for i in range(num_points):

        up = (left_wheel_steer_angle + right_wheel_steer_angle)/2

        xn1 = xn + delta_t * (us * np.cos(theta_n))
        yn1 = yn + delta_t * (us * np.sin(theta_n))
        theta_n1 = theta_n + delta_t * ((us * np.tan(up))/wheelbase)

        trajectory_points[i] = [xn1, yn1, theta_n1]

        #finding wheel trajectory
        left_wheel_trajectory[i] = [xn1 , yn1  + (rear_track_width/2) , 0, 1]
        right_wheel_trajectory[i] = [xn1 , yn1 - (rear_track_width/2) , 0, 1]

        #Transforming trajectory points to camera frame
        left_wheel_cam_frame[i] = P @ left_wheel_trajectory[i].T
        right_wheel_cam_frame[i] = P @ right_wheel_trajectory[i].T

        xn = xn1
        yn = yn1
        theta_n = theta_n1

    return left_wheel_cam_frame, right_wheel_cam_frame

#############################################################################################################

#Function to convert to camera pixel coordinates
def pixel_coordinates(left_wheel_cam_frame, right_wheel_cam_frame, num_points):
    for i in range(num_points):
        if left_wheel_cam_frame[i,2] !=0 and right_wheel_cam_frame[i,2] != 0:
            left_wheel_cam_frame[i] /= left_wheel_cam_frame[i,2]
            right_wheel_cam_frame[i] /= right_wheel_cam_frame[i,2]

    left_wheel_homogeneous_points = left_wheel_cam_frame[:,:2]
    right_wheel_homogeneous_points = right_wheel_cam_frame[:,:2]

    left_pixel_coordinates = np.round(left_wheel_homogeneous_points).astype(int)
    right_pixel_coordinates = np.round(right_wheel_homogeneous_points).astype(int)

    return left_pixel_coordinates, right_pixel_coordinates

#############################################################################################################

#Function to change pinion angle
def on_trackbar(val):
    pass

cv2.namedWindow('Frame')

trackbar_name = 'Pinion Angle'
cv2.createTrackbar(trackbar_name, 'Frame', 540, 1080, on_trackbar)

#############################################################################################################

#Function to draw lines using the pixel coordinates
def draw_points(frame, points, color, thickness = 2):
    for i in range(1, len(points)):
        cv2.line(frame, points[i - 1], points[i], color, thickness)

#############################################################################################################

#Function to draw horizontal lines using the pixel coordinates
def draw_horizontal_points(frame, left_points, right_points, color, thickness = 2):
    no_horizontal_lines = 3
    interval = len(left_points) // no_horizontal_lines
    for i in range(0, len(left_points), interval):
        if i < len(right_points):
            cv2.line(frame, tuple(left_points[i]), tuple(right_points[i]), color, thickness)

#############################################################################################################

#Setting up the video and saving options
video_path = 'Video.mp4'
cap = cv2.VideoCapture(video_path)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (1280, 720))
frame_id = 0

#############################################################################################################

#Viewing the video and drawing the lines
while cap.isOpened():
    ret, frame = cap.read()
   
    if not ret:
        break
    
    #For static lines
    left_wheel_steer_angle, right_wheel_steer_angle = calculate_wheel_steer_angles(0) 

    left_wheel_cam_frame, right_wheel_cam_frame = wheel_trajectory(left_wheel_steer_angle, right_wheel_steer_angle, 0, 0, 0, 1000)
    
    left_pixel_coordinates, right_pixel_coordinates = pixel_coordinates(left_wheel_cam_frame, right_wheel_cam_frame, 1000)

    draw_points(frame, left_pixel_coordinates, color = (255, 0, 0))
    draw_points(frame, right_pixel_coordinates, color = (255, 0, 0))
    draw_horizontal_points(frame, left_pixel_coordinates, right_pixel_coordinates, color = (255, 0, 0))

    #For dynamic lines
    # pinion_angle = get_pinion_angle
    pinion_angle = cv2.getTrackbarPos(trackbar_name, 'Frame') - 540

    left_wheel_steer_angle, right_wheel_steer_angle = calculate_wheel_steer_angles(pinion_angle) 

    left_wheel_cam_frame, right_wheel_cam_frame = wheel_trajectory(left_wheel_steer_angle, right_wheel_steer_angle, 0, 0, 0, 1500)

    left_pixel_coordinates, right_pixel_coordinates = pixel_coordinates(left_wheel_cam_frame, right_wheel_cam_frame, 1500)
    
    draw_points(frame, left_pixel_coordinates, color = (0, 0, 255))
    draw_points(frame, right_pixel_coordinates, color = (0, 0, 255))

    # Show the result
    cv2.imshow('Frame', frame)
    out.write(frame)  # Write out frame to video

    # Increment frame_id
    frame_id += 1

    # Break loop on 'q' key press
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Release everything
cap.release()
out.release()
cv2.destroyAllWindows()