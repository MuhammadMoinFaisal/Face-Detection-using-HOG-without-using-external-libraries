#Importing Libraries

import cv2

#OpenCV-Python is a library of Python  designed to solve image processing and computer vision tasks. 

#Why we are using this in this task?

#We are using cv2.imread(), cv2.rectangle and cv2.putText() in our program and their functionality is listed below.

#1. cv2.imread() method loads an image from the specified file.

#2. cv2.rectangle() method  draws a rectangle on any image.

#3. cv2.putText() method is used to draw a text string on any image.

import numpy
import numpy as np

#Numpy library helps us to large multi-dimensional arrays and matrices


import matplotlib.pyplot as plt

#Matplotlib is a data visualization and graphical plotting library for Python

#Why we are using this in our task?

#We are using plt.figure(), plt.imshow(), plt.savefig('') in our code.

 #   1. plt.figure(), The figure() function in pyplot module of matplotlib library is used to create a new figure
    
 #  2. The imshow() function in pyplot module of matplotlib library is used to display data as an image
    
 #   3. plt.savefig(''), savefig() method is used to save the figure created after plotting data.


import skimage.feature as skim

#skimage.feature is used for the image pre-processing


import sys

#To access system specific parameters and functionWhy we are using this in our task here?

#When we want to print the array of our feature vectors, as it is of length 3780, so the issue which comes is the result which appears is in the form 0.224, 0.2345, ....., 0., 0.23. To print the full length of numpy array as shown in the image below we need to do import sys.
# # Detailed Analysis about Histogram of Oriented Gradients
#Lets first discuss about Histogram of Oriented Gradients (HOG) feature descriptorWhat is Feature Descriptor?

#Feature Descriptor is representation of an image or image patch that simplies the image by extracting useful information and throwing away the useless information

#What does Feature Descriptor do?

#Feature Descriptor converts an image of size width x height x 3 (channels) to feature vector/ array of length n


#What does HOG feature descriptor do?

#In case of HOG feature descriptor, the input image is of size 64x128x3 and the output feature vector is of length 3780


#Feature Vector is not useful for the purpose of viewing the image. But it is useful for the tasks like object detection and image recognition

#What kind of features do we need?

#As we are building an face detector that detect faces, We will run an edge detector on the image which contains face and easily tell if it is an face by simply looking at the edge image alone.


#In the HOG feature descriptor, the distribution ( histograms ) of directions of gradients ( oriented gradients ) are used as features. Gradients ( x and y derivatives ) of an image are useful because the magnitude of gradients is large around edges and corners ( regions of abrupt intensity changes ) and we know that edges and corners pack in a lot more information about object shape than flat regions.How to calculate Histogram of Oriented GradientsFirst Step: Calculate the Gradient ImagesTo calculate the HOG descriptor in the first step we calculate the Gradient Images, we need to first calculate the horizontal and vertical gradients, as, we want to calculate the histogram of gradients. 

#We can  achieve this, by using Sobel operator in OpenCV with kernel size 1.

#But as in this task, we are only allowed to used OpenCv for Computing the gradient, Performing convolution and 
#Drawing a rectangle, So we will be doing this by creating a function not by using the sobel operator in OpenCV

#First we are calculating the horizontal and vertical gradients and in the next step we are filtering the image


def sobel_filter():
    # Creating an Numpy array of 3x3 and naming the variable as filter x
    filter_x = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
    # Creating an Numpy array of 3x3 and naming the variable as filter y
    filter_y = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    return filter_x, filter_y






def image_filtering(im, filter):
    #Printing an array of zeros with the same shape 
    im_filtered = np.zeros_like(im)
    #.flatten return a copy of the array collapsed into one dimension.
    filter_flattened = filter.flatten()
    height = len(im)
    width = len(im[0])
    #np.pad pads the numpy arrays
    im_padded = np.pad(im,1)

    for i in range(height):
        for j in range(width):
            im_window = im_padded[i:i+3,j:j+3].flatten()
            im_filtered[i][j] = np.dot(im_window, filter_flattened)

    return im_filtered

#Step:2: 

#Next, we can find the magnitude and direction of gradient using the following the magnitude and direction of gradient formula

#We can also do this using OpenCV by using using the function cartToPolar, but here as we can use OpenCV for only Computing the gradient, Performing convolution and  Drawing a rectangle, So we will be doing this by creating a function and applying the magnituide and direction of the gradient formula, not by using the cartToPolar operator in OpenCV.

#So we are using this formula inside the function to calculate the magnitude and direction of gradient



def magnitude_direction_gradient(im_dx, im_dy):
    height = len(im_dx)
    width = len(im_dy[0])
    #np.zeros_like return an array of zeros with the same shape and type as a given array.
    grad_mag = np.zeros_like(im_dx,dtype=np.float)
    grad_angle = np.zeros_like(im_dx,dtype=np.float)
    for i in range(height):
        #Applying the formula as listed in the above image
        grad_mag[i] = [np.linalg.norm([dX,dY]) for (dX,dY) in zip(im_dx[i],im_dy[i])]
        for j in range(width):
            dX = im_dx[i][j]
            dY = im_dy[i][j]
            if abs(dX) > 0.00001:
                grad_angle[i][j] = np.arctan(float(dY/dX)) + (np.pi/2)
            else:
                if dY < 0 and dX < 0:
                    grad_angle[i][j] = 0
                else:
                    grad_angle[i][j] = np.pi
    return grad_mag, grad_angle

#Step:3: 

#Calculate the Histogram of Gradients in an 8x8 cellsIn this part we basically divide the image into 8x8 cells and then we calculate the histogram of gradient  for the 8x8 cells.

#Question : Why we divide the image into 8x8 cells?

#As we use HOG feature descriptor to extract features from the image that contains meaningful information. An 8x8 image patch basically has 8x8x3 = 192 pixel values while the gradient would have about 2 values (magnitude and direction) per pixel which adds upto 8x8x2 = 128 numbers are then represented using an n bin histogram



def create_histogram(grad_mag, grad_angle, cell_size):
    #Defining the height, width, bins, x corner and the y corner
    height = len(grad_mag)
    width = len(grad_mag[0])
    nBins = 6
    x_corner = 0
    y_corner = 0
    ori_histo = np.zeros((int(height / cell_size), int(width / cell_size), nBins), dtype=float)
    #Applyint a while loop
    while (x_corner + cell_size) <= height:
        while (y_corner + cell_size) <= width:
            hist = np.zeros((6), dtype=float)
            magROI = grad_mag[x_corner:x_corner+cell_size, y_corner:y_corner+cell_size].flatten()
            angleROI = grad_angle[x_corner:x_corner+cell_size, y_corner:y_corner+cell_size].flatten()
            for i in range(cell_size*cell_size):
                angleInDeg = angleROI[i] * (180 / np.pi)
                if angleInDeg >=0 and angleInDeg < 30:
                    hist[0] += magROI[i]
                elif angleInDeg >=30 and angleInDeg < 60:
                    hist[1] += magROI[i]
                elif angleInDeg >=60 and angleInDeg < 90:
                    hist[2] += magROI[i]
                elif angleInDeg >=90 and angleInDeg < 120:
                    hist[3] += magROI[i]
                elif angleInDeg >=120 and angleInDeg < 150:
                    hist[4] += magROI[i]
                else:
                    hist[5] += magROI[i]
            ori_histo[int(x_corner/cell_size),int(y_corner/cell_size),:] = hist
            y_corner += cell_size
        x_corner += cell_size
        y_corner = 0
    return ori_histo

#Step:4: Block NormalizationQuestion: Why do we need Block Normalization?As the gradients of an image are sensitive to overall lighting. If you make the image darker by dividing all pixel values by 2, the gradient magnitude will change by half, and therefore the histogram values will change by half.

#Ideally, we want our descriptor to be independent of lighting variations. In other words, we would like to “normalize” the histogram so they are not affected by lighting variations.


def block_normalization(ori_histo, block_size):
    x_window = 0
    y_window = 0
    height = len(ori_histo)
    width = len(ori_histo[0])
    ori_histo_normalized = np.zeros(((height-(block_size-1)), (width-(block_size-1)), (6*(block_size**2))), dtype=float)
    while x_window + block_size <= height:
        while y_window + block_size <= width:
            concatednatedHist = ori_histo[x_window:x_window + block_size, y_window:y_window + block_size,:].flatten()
            histNorm = np.sqrt(np.sum(np.square(concatednatedHist)) + 0.001)
            ori_histo_normalized[x_window,y_window,:] = [(h_i / histNorm) for h_i in concatednatedHist]
            y_window += block_size
        x_window += block_size
        y_window = 0
    return ori_histo_normalized

#Step 5: Extract Histogram of Oriented GradientsTo calculate the final feature vector for the entire image patch, the 36×1 vectors are concatenated into one giant vector.

#Each 16×16 block is represented by a 36×1 vector. So when we concatenate them all into one gaint vector we obtain a 36×105 = 3780 dimensional vector


def extract_histogram_of_oriented_gradients(im):
    im = im.astype('float') / 255.0
    im = (im - np.min(im)) / np.max(im)
    x_filter, y_filter = sobel_filter()
    dx = image_filtering(im, x_filter)
    dy = image_filtering(im, y_filter)
    mag_matrix, angle_mat = magnitude_direction_gradient(dx, dy)
    histogramMat = create_histogram(mag_matrix, angle_mat, 8)
    hog_descriptors = block_normalization(histogramMat, 2)
    hog = hog_descriptors.flatten()
    #visualize_hog_each_block(im, hog, 8, 2)
    return hog

#Step 6: Visualize the Histogram of Each BlockThe HOG descriptor of an image patch is usually visualized by plotting the 9×1 normalized histograms in the 8×8 cells


def visualize_hog_each_block(im, hog, cell_size, block_size):
    num_bins = 6
    max_len = 7  
    # controlling sum of segment lengths for visualized histogram bin of each block
    im_h, im_w = im.shape
    num_cell_h, num_cell_w = int(im_h / cell_size), int(im_w / cell_size)
    num_blocks_h, num_blocks_w = num_cell_h - block_size + 1, num_cell_w - block_size + 1
    histo_normalized = hog.reshape((num_blocks_h, num_blocks_w, block_size**2, num_bins))
    # num_blocks_h x num_blocks_w x num_bins
    histo_normalized_vis = np.sum(histo_normalized**2, axis=2) * max_len  
    angles = np.arange(0, np.pi, np.pi/num_bins)
    mesh_x, mesh_y = np.meshgrid(np.r_[cell_size: cell_size*num_cell_w: cell_size], np.r_[cell_size: cell_size*num_cell_h: cell_size])
    # expand to same dims as histo_normalized
    mesh_u = histo_normalized_vis * np.sin(angles).reshape((1, 1, num_bins))  
    # expand to same dims as histo_normalized
    mesh_v = histo_normalized_vis * -np.cos(angles).reshape((1, 1, num_bins))  
    plt.imshow(im, cmap='gray', vmin=0, vmax=1)
    for i in range(num_bins):
        plt.quiver(mesh_x - 0.5 * mesh_u[:, :, i], mesh_y - 0.5 * mesh_v[:, :, i], mesh_u[:, :, i], mesh_v[:, :, i],
                   color='white', headaxislength=0, headlength=0, scale_units='xy', scale=1, width=0.002, angles='xy')
    plt.show()

#Step 7: To test our face detection algorithm on large images, we check our detected bounding boxes against the coordinates given for the ground truth rectangles.

#I will be creating an function and using the same formula given below as 



def iou(box1,box2, boxSize):
    sumOfAreas = 2*(boxSize**2)
    box_1 = [box1[0], box1[1], box1[0] + boxSize, box1[1] + boxSize]
    box_2 = [box2[0], box2[1], box2[0] + boxSize, box2[1] + boxSize]
    intersectionArea = (min(box_1[2],box_2[2]) - max(box_1[0], box_2[0])) * (min(box_1[3],box_2[3]) - max(box_1[1], box_2[1]))
    return float(intersectionArea / (sumOfAreas - intersectionArea))

#Step:08:  Perform Face DetectionAfter extracting all the useful features from the images, i am finding all the bounding boxes and setting an stride of 3, which is believe to give some good results and then i will be applying thresholding and doing non maximum supression


def facedetection(I_target, I_template):
    template_HOG = extract_histogram_of_oriented_gradients(I_template)
    template_HOG = template_HOG - np.mean(template_HOG) 
    template_HOG_norm = np.linalg.norm(template_HOG)
    img_h, img_w = I_target.shape
    box_h, box_w = I_template.shape
    x = 0
    y = 0
    all_bounding_boxes = []
    while x + box_h <= img_h:
        print("Searching in row", x)
        while y + box_w <= img_w:
            img_window = I_target[x:x+box_h, y:y+box_w]
            img_HOG = extract_histogram_of_oriented_gradients(img_window)
            img_HOG = img_HOG - np.mean(img_HOG) # Normalize
            img_HOG_norm = np.linalg.norm(img_HOG)
            score = float(np.dot(template_HOG, img_HOG) / (template_HOG_norm*img_HOG_norm))
            all_bounding_boxes.append([y,x,score])
            # A stride of 3 is enough to produce a good result for the target. 
            y += 3
        x += 3
        y = 0
    print("Computing scores.")
    bounding_boxes = []
    all_bounding_boxes = sorted(list(all_bounding_boxes),key=lambda box: box[2], reverse=True)
    # Doing Thresholding
    thresholded_boxes = []
    for box in all_bounding_boxes:
        if box[2] >= 0.6: 
            thresholded_boxes.append(box)
    #Printing the Number of Boxes After Thresholding
    print("Number of boxes after thresholding: ", len(thresholded_boxes))
    # Applying Non Maximum Supression
    while thresholded_boxes != []:
        currBox = thresholded_boxes[0]
        bounding_boxes.append(currBox)
        toRemove = []
        for box in thresholded_boxes:
            if iou(currBox, box, box_w) >= 0.5:
                toRemove.append(box)
        for remBox in toRemove:
            thresholded_boxes.remove(remBox)
    return np.array(bounding_boxes)

#Visualizing Face DetectionNow i will be visualizing the resulting faces detected from the images,  after the face detection algorithm implementation


def visualize_face_detection(I_target,bounding_boxes,box_size):

    hh,ww,cc=I_target.shape

    fimg=I_target.copy()
    # Defining the Range as per the shape of the bounding boxes
    for ii in range(bounding_boxes.shape[0]):
        # Defining the Bounding Boxes and applying the conditions
        x1 = bounding_boxes[ii,0]
        x2 = bounding_boxes[ii, 0] + box_size 
        y1 = bounding_boxes[ii, 1]
        y2 = bounding_boxes[ii, 1] + box_size

        if x1<0:
            x1=0
        if x1>ww-1:
            x1=ww-1
        if x2<0:
            x2=0
        if x2>ww-1:
            x2=ww-1
        if y1<0:
            y1=0
        if y1>hh-1:
            y1=hh-1
        if y2<0:
            y2=0
        if y2>hh-1:
            y2=hh-1
        # cv2.rectangle draws a rectangle on any image.
        fimg = cv2.rectangle(fimg, (int(x1),int(y1)), (int(x2),int(y2)), (255, 0, 0), 1)
        #cv2.putText() method is used to draw a text string on any image.
        cv2.putText(fimg, "%.2f"%bounding_boxes[ii,2], (int(x1)+1, int(y1)+2), cv2.FONT_HERSHEY_SIMPLEX , 0.5, (0, 255, 0), 2, cv2.LINE_AA)

    plt.figure(3)
    plt.imshow(fimg, vmin=0, vmax=1)
    plt.savefig('output_face_detection.png')




im = cv2.imread(r"C:\UpworkProjects\ComputerVisionProject\Images\sampleinput.png", 0)
hog = extract_histogram_of_oriented_gradients(im)
print(hog)


numpy.set_printoptions(threshold=sys.maxsize)


# In[364]:


print(hog)




I_target= cv2.imread(r'C:\UpworkProjects\ComputerVisionProject\Images\First.png', 0)
I_template = cv2.imread(r'C:\UpworkProjects\ComputerVisionProject\Images\sampleinput.png', 0)
bounding_boxes=facedetection(I_target, I_template)
I_target_c= cv2.imread(r'C:\UpworkProjects\ComputerVisionProject\Images\First.png')
visualize_face_detection(I_target_c, bounding_boxes, I_template.shape[0])






