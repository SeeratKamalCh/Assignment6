import cv2
import numpy as np
import os
from matplotlib import pyplot as plt 
from scipy import linalg


"""
This code takes an image as input and attempts to apply caliberation
technique to find the projection matrix and instrinsic and extrinsic 
parameters of the camera
"""

# Speicfy number of points taken and number of knowns in matrix C
NUM_POINTS = 25
NUM_KNOWNS = 12

# function to read the image 
def read_image():
    try:
        filepath = "rubic.jpg"
        image = cv2.imread(filepath)
    except:
        print("Error occurred try again")
    return image

# function to display an image
def show_image(image):
        imS = cv2.resize(image, (960, 900)) 
        cv2.imshow("image", imS)
        cv2.waitKey(0)
        return

# Mark the image points i.e mark pixel locations on the image    
def mark_image_points(image):
    image_points = []
    image_points.append([300, 502])
    image_points.append([370, 467])
    image_points.append([434, 432])
    image_points.append([491, 402])
    image_points.append([298, 427])
    image_points.append([360, 398])
    image_points.append([429, 364])
    image_points.append([491, 332])
    image_points.append([295, 333])
    image_points.append([363, 310])
    image_points.append([428, 276])
    image_points.append([497, 245])
    image_points.append([300, 245])
    image_points.append([369, 210])
    image_points.append([440, 182])
    image_points.append([498, 154])
    image_points.append([230, 467])
    image_points.append([165, 429])
    image_points.append([109, 400])
    image_points.append([238, 398])
    image_points.append([168, 366])
    image_points.append([108, 329])
    image_points.append([228, 304])
    image_points.append([165, 274])
    image_points.append([102, 243])
    
    # show the points on the image
    for i in range(len(image_points)): 
        x= image_points[i][0]
        y = image_points[i][1]
        cv2.circle(image, (x, y), 3, 255, -1)
    show_image(image)
    return image_points

# function to mark the corresponding real word points of the image points
def mark_world_points():
    cube_size = 19
    world_points = []
    world_points.append([0, 0, 0])
    world_points.append([1, 0, 0])
    world_points.append([2, 0, 0])
    world_points.append([3, 0, 0])
    world_points.append([0, 1, 0])
    world_points.append([1, 1, 0])
    world_points.append([2, 1, 0])
    world_points.append([3, 1, 0])
    world_points.append([0, 2, 0])
    world_points.append([1, 2, 0])
    world_points.append([2, 2, 0])
    world_points.append([3, 2, 0])
    world_points.append([0, 3, 0])
    world_points.append([1, 3, 0])
    world_points.append([2, 3, 0])
    world_points.append([3, 3, 0])
    world_points.append([0, 0 ,1])
    world_points.append([0, 0 ,2])
    world_points.append([0, 0, 3])
    world_points.append([0, 1, 1])
    world_points.append([0, 1, 2])
    world_points.append([0, 1, 3])
    world_points.append([0, 2, 1])
    world_points.append([0, 2, 2])
    world_points.append([0, 2, 3])
    # scale the coordinates with cube size to scale it to millimeters
    for i in world_points:
        i[0] = i[0] * cube_size
        i[1] = i[1] * cube_size
        i[2] = i[2] * cube_size
    return world_points

def generate_c_matrix(image):
    # generate c matrix to form the equation Ca = 0
    # where C is the matrix of knowns and a is the matrix of unknowns
    image_points = mark_image_points(image)
    world_points = mark_world_points()
    i = 0
    c_matrix = np.zeros((NUM_POINTS * 2, NUM_KNOWNS))
    for index in range(NUM_POINTS):
        # get values of world coordinates and image points for current index
        X, Y, Z, x, y = get_values(image_points[index], world_points[index])
        # set row for x coordinate in image point x,y 
        array = np.array([X, Y, Z, 1, 0, 0, 0, 0, -x*X, -x*Y, -x*Z, -x])
        c_matrix[i] = array
        # Append row for y coordinate in current image point x, y
        array = np.array([0, 0, 0, 0, X, Y, Z, 1, -y*X, -y*Y, -y*Z, -y])
        c_matrix[i + 1] = array
        i = i + 2
    return c_matrix

# helper function to extract the world and image point values
def get_values(image_points, world_points):
    X = world_points[0]
    Y = world_points[1]
    Z = world_points[2]
    x = image_points[0]
    y = image_points[1]
    return X, Y, Z, x, y
    
# calculate right null vector of C matrix
def SVD_right_null(matrix):
    # Calculate SVD of c_matrix 
    U, D, V = np.linalg.svd(matrix)
    # and get last column from V as it corresponds to the smallest eigenvalue
    minval = np.argmin(np.ma.masked_where(D==0, D))
    # reshape the matrix to 3x4 format
    V = np.transpose(V)
    null_vector = V[:, minval].reshape(3, 4)
    return null_vector


# function to calculate center of camera matrix
def find_center(matrix_P):
    # solve SVD and get null vector which gives the center C
    U, D, V = linalg.svd(matrix_P)
    columns = U.shape[1]
    null_vector = U[:, columns - 1 ]
    return null_vector


# function for QR decomposition of matrix P to get K and R
def QR_decomposition(matrix_P):
    # Extract P3x3 and do its QR decomposition as P3x3 = KR
    matrix = matrix_P[0:3,0:3]
    K, R = linalg.rq(matrix)
    K = K/float(K[2,2])
    inv = np.linalg.inv(K)
    # Calculate Translation vector
    T = np.matmul(-inv, matrix_P[:, 3])
    return K, R

# Calculate focal length of camera
def find_focal_length(image, matrix_K):
    sensor_y = 23.4
    sensor_x = 15.4
    image_y = image.shape[0]
    image_x = image.shape[1]
    # calculate mx and my 
    mx = image_x / sensor_x
    my = image_y / sensor_y
    # divide the values by mx and my to get focal length
    focal_length = (matrix_K[0,0]) / mx
    focal_length_second = (matrix_K[1, 1]) / my
    return focal_length * - 1, focal_length_second * -1


# Calculae points with projection matrix matrix_P
def calculate_points(world_points, matrix_P, image):
    original_points = np.zeros((NUM_POINTS, 2))
    for i in range(NUM_POINTS):
        # get world points
        x = world_points[i][0]
        y = world_points[i][1]
        z = world_points[i][2]
        w = 1
        point = np.array([x, y, z, w]).reshape(4, 1)
        # Calculate homogenous points by hp(3x1) = matrix_P(3x4) * world_point[X,Y,Z,1]
        homogenous_point = np.matmul(matrix_P, point)
        # now calculate euclidean points from homegenous points
        x_original = homogenous_point[0] / homogenous_point[2]
        y_original = homogenous_point[1] / homogenous_point[2]
        original_points[i][0] = int(x_original)
        original_points[i][1] = int(y_original)
    # show the calculated points on the image
    for i in range(len(original_points)): 
        x= original_points[i][0]
        y = original_points[i][1]
        cv2.circle(image, (int(x), int(y)), 3, (0,0,255), -1)
    show_image(image)
    return original_points


# Calculate projection error between image_points and calculated image points
def projection_error(image_points, original_points):
    error = 0
    for i in range(NUM_POINTS):
        # calculate difference
        x = image_points[i][0]
        y = image_points[i][1]
        x_cal = original_points[i][0]
        y_cal = original_points[i][1]
        # Calculate magnitude difference
        diff = np.square(x - x_cal) + np.square(y - y_cal)
        diff = np.sqrt(diff)
        error = error + diff    
    # get average of all the points
    error = error / NUM_POINTS    
    return error


# main driver function for caliberation technique
def caliberation():
    image = read_image()
    c_matrix = generate_c_matrix(image)
    p_matrix = SVD_right_null(c_matrix)
    center = find_center(p_matrix)
    K, R = QR_decomposition(p_matrix)
    focal, focal2 = find_focal_length(image, K)
    image_points = mark_image_points(image)
    world_points = mark_world_points()
    original_points = calculate_points(world_points, p_matrix, image)
    error = projection_error(image_points, original_points)
    print("Intrinsic K\n",K)
    print("********************")
    print("Extrinsic R\n", R)
    print("********************")
    print("Projection matrix\n", p_matrix)
    print("********************")
    # should be between 27 to 83
    print("focal length 1: ", focal)
    print("focal length 2: ", focal2)
    

caliberation()
        
    


