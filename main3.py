
import main2
import os
import gc
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from PIL import Image
from skimage import filters
from imutils import contours
from scipy.special import softmax
from reportlab.pdfgen import canvas
# from google.colab.patches import cv2_imshow
from IPython.display import Markdown, display
from keras.models import Sequential, model_from_json
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report,confusion_matrix
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout , BatchNormalization



# Load Model Architecture (JSON File)
json_file = open("D:/My Projects/BITHACK/static/model.json", 'r')
loaded_model_json = json_file.read()
json_file.close()
# Load the trained model weights (h5 file)
loaded_model = model_from_json(loaded_model_json)
# Initiate the model with the loaded weights
loaded_model.load_weights("D:\My Projects\BITHACK\static\model_weights.h5")
print("Loaded model from disk")

loaded_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# def saveimg(word):
#     from PIL import Image, ImageDraw, ImageFont
#     image = Image.new('RGB', (500, 200), color='white')
#     draw = ImageDraw.Draw(image)
#     text =word
#     font = ImageFont.load_default()
#     draw.text((10, 10), text, font=font, fill='black')
#     image.save('text_image.png')
#     print("I'm saved")
def saveimg(word):
    from PIL import Image, ImageDraw, ImageFont
    # word = 'அகரா'
    width, height = 200, 100  
    image = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(image)
    font_size = 36
    font_path = "D:/My Projects/BITHACK/latha.ttf"  
    font = ImageFont.truetype(font_path, font_size)
    text_width, text_height = draw.textsize(word, font=font)
    x = (width - text_width) / 2
    y = (height - text_height) / 2
    draw.text((x, y), word, font=font, fill='black')
    image.save('tamil_image.png')
def print_big(char):    
    display(Markdown(f"<h1>{char}</h1>"))

def printchar(index, word):
    # TamilChar.csv is the dataset which contains the unicode equivalents for a given character class
    df = pd.read_csv('D:/My Projects/BITHACK/static/tamil_156.csv', header=0)

    # Get the Unicode in hexadecimal string format for the previous index
    prev_index = index
    unicode_hex_string = df['Unicode'].values[prev_index]
    unicode_hex_values = unicode_hex_string.split()

    chars_together = ""
    for hex_value in unicode_hex_values:
        # Convert hexadecimal string to integer
        char_int = int(hex_value, 16)
        # Convert integer to corresponding character
        character = chr(char_int)
        chars_together += character

    word += chars_together
    font_path = 'D:/My Projects/BITHACK/latha.ttf'  # Replace with the path to your Tamil font file
    font_properties = fm.FontProperties(fname=font_path)

    plt.text(0.5, 0.5, word, fontproperties=font_properties, fontsize=24, ha='center', va='center')
    plt.axis('off')
    # plt.show()
    print(word)
    
    return word




def predictions_word(img2,word):
    # Convert the grayImage to (128,128) dimension and normalize it
    grayImage = (cv2.resize(img2,(128,128))/255.0).astype(np.float32)
    sample_img = [grayImage]
    sample_img = np.array(sample_img)


    # Predict the class of the word
    predictions = loaded_model.predict(sample_img)
    # Take the class with maximum probability
    predictions_abs = np.array([np.argmax(i) for i in predictions])

    # Print raw predictions and class indices for debugging

    print("Predicted Class Indices:", predictions_abs)

    # f, axarr = plt.subplots(1,1)
    # axarr[0].imshow(img_arr,cmap="gray")
    # axarr.imshow(grayImage,cmap="gray")

    # Retrieve the corresponding character of the class
    word = printchar(predictions_abs[0],word)
    return word



def sharpening(img,word):
  # Create a blank 3-channel image of the same size as the grayscale image
  height, width = img.shape
  rgb = np.zeros((height, width, 3), dtype=np.uint8)

  # Copy the grayscale image to all 3 channels of the RGB image
  rgb[:,:,0] = img
  rgb[:,:,1] = img
  rgb[:,:,2] = img

  # Threshold the image to create a binary mask
  _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

  # Find the contours of the binary mask
  contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

  # Create a blank white image of the same size as the input image
  white = np.full_like(rgb, (255, 255, 255))

  # Draw the contours onto the white image with a thickness of 5 pixels
  cv2.drawContours(white, contours, -1, (0, 0, 0), -1)

  # Apply a sharpening filter to the image
  kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
  sharpened = cv2.filter2D(white, -1, kernel)

  # Display the sharpened image
  cv2.imshow("sharpened",sharpened)

  # Use the reconstructed sharpened image for character recognition
  word = predictions_word(sharpened,word)

  # Return predicted classes
  cv2.waitKey(0)
  return word




def bounding_box(path):

  # Read the image in grayscale
  img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
  # Use Contrast Limited Adaptive Histogram Equalization to contrast the image and prevent over-amplification
  clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
  img = clahe.apply(img)
  # Smoothen the image using Gaussian and Median Blue to remove Image noises
  img = cv2.GaussianBlur(img, (3, 3), 0)
  img = cv2.medianBlur(img, 3)
  # Apply Binary Thresholding to sharpen the image
  _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
  # Apply Connected Component Analysis to Split the text into its constituent characters
  num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8, ltype=cv2.CV_32S)
  # Set a minimum size threshold to filter out small connected components

  min_size = 200  # adjust as needed based on the expected size of the characters
  # Loop over the connected components and save each character image without the bounding box
  bboxes = []
  for i in range(1, num_labels):
      area = stats[i, cv2.CC_STAT_AREA]
      if area < min_size:
          continue
      x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
      # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
      bbox = (x, y, w, h)
      bboxes.append(bbox)

  cv2.imshow("Image",img)
  # Sort the bounding boxes by their x-coordinate values
  bboxes = sorted(bboxes, key=lambda x: x[0])

  aspect_ratio_threshold = 0.2
  # Apply Padding to the images by adjusting the aspect ratio
  word = ""
  for i, bbox in enumerate(bboxes):
      x, y, w, h = bbox
      aspect_ratio = float(w) / h
      if aspect_ratio < aspect_ratio_threshold or aspect_ratio > 1/aspect_ratio_threshold:
            continue
      char_img = img[y:y+h, x:x+w]
      sharpening(char_img)



def bounding_box2(path):
  # Read the image in grayscale
  img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
  # Use Contrast Limited Adaptive Histogram Equalization to contrast the image and prevent over-amplification
  clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
  img = clahe.apply(img)
  # Smoothen the image using Gaussian and Median Blue to remove Image noises
  img = cv2.GaussianBlur(img, (3, 3), 0)
  img = cv2.medianBlur(img, 3)
  # Apply Binary Thresholding to sharpen the image
  _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
  # Apply Connected Component Analysis to Split the text into its constituent characters
  num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8, ltype=cv2.CV_32S)
  # Set a minimum size threshold to filter out small connected components
  min_size = 200  # adjust as needed based on the expected size of the characters
  # Loop over the connected components and save each character image without the bounding box
  bboxes = []
  for i in range(1, num_labels):
      area = stats[i, cv2.CC_STAT_AREA]
      if area < min_size:
          continue
      x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
      # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
      bbox = (x, y, w, h)
      bboxes.append(bbox)
  cv2.imshow("Image",img)
  # Sort the bounding boxes by their x-coordinate values
  bboxes = sorted(bboxes, key=lambda x: x[0])
  # Apply Padding to the images to set the aspect ratio and prevent over-magnification of the images
  aspect_ratio_threshold = 0.2
  word = ""
  indices = []

  for i in range(len(bboxes)-1):
    if(i in indices):
      continue
    x1, y1, w1, h1 = bboxes[i]
    x2, y2, w2, h2 = bboxes[i+1]
    # Ignore proper images
    aspect_ratio = float(w1) / h1
    if aspect_ratio < aspect_ratio_threshold or aspect_ratio > 1/aspect_ratio_threshold:
          continue
    # Pad the over magnified images with the repsective aspect ratio threshold
    if(x2>x1 and x2<x1+w1 and x2+w2>x1 and x2+w2<x1+w1):
      x,y,w,h = x1,y2,w1,h1+h2+25
      # print(i,i+1)
      # print(i,x1,x1+w1,y1,y1+h1,i+1,x2,x2+w2,y2,y2+h2)
      # print(i,x1,y1,w1,h1,i+1,x2,y2,w2,h2)
      char_img = img[y:y+h, x:x+w]
      word = sharpening(char_img,word)
      indices.append(i+1)
    else:
      # Make predictions with proper images
      char_img = img[y1:y1+h1, x1:x1+w1]
      word = sharpening(char_img,word)
  # Make predictions with padded images
  if(len(bboxes)-1 not in indices):
    x1, y1, w1, h1 = bboxes[-1]
    char_img = img[y1:y1+h1, x1:x1+w1]
    word = sharpening(char_img,word)
    print("I'm gonna end")
  return word




def segment_text_lines(image_path, output_folder, line_margin=15, desired_width=4800):
    # Read the original color image
    img = cv2.imread(image_path)

    # Convert the image to grayscale for thresholding
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise and improve text extraction
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply adaptive thresholding to create a binary image
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # Calculate the horizontal projection (histogram)
    horizontal_projection = binary.sum(axis=1)

    # Apply thresholding to the horizontal projection to find peaks
    peak_threshold = 0.2 * np.max(horizontal_projection)
    peaks = np.where(horizontal_projection < peak_threshold)[0]

    # Initialize variables for line segmentation
    lines = []
    start = 0

    # Split the image into lines based on detected peaks
    for end in peaks:
        if end - start > 10:  # Ignore small gaps
            # Add a margin to the top and bottom of the line image
            start_with_margin = max(start - line_margin, 0)
            end_with_margin = min(end + line_margin, img.shape[0])
            lines.append((start_with_margin, end_with_margin))
        start = end

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate through the lines and save each one as a separate image with the desired width
    for i, (start, end) in enumerate(lines):
        line_image = img[start:end, :]  # Extract the color line from the original image

        # Calculate the new height while maintaining the aspect ratio
        aspect_ratio = line_image.shape[1] / line_image.shape[0]
        desired_height = int(desired_width / aspect_ratio)

        # Resize the line_image to the desired size
        line_image_resized = cv2.resize(line_image, (desired_width, desired_height))

        # Define the file path for the segmented image
        save_path = os.path.join(output_folder, f"segmented_line_{i + 1}.png")

        # Save the resized image as a color image
        cv2.imwrite(save_path, line_image_resized)

        # Call the print_big(bounding_box_2) function for the saved image
        print_big(bounding_box2(save_path))

        print(f"Saved segmented line {i + 1} to {save_path} with size {desired_width}x{desired_height}")

    return lines

def is_single_line(image_path):
    # Load the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Threshold the image to create a binary image
    _, binary_image = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)

    # Find contours in the binary image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Check the number of contours to determine if it's a single line or multiple lines
    if len(contours) == 1:
        return True
    else:
        return False

def cleanup_output_folder(output_folder):
    # Remove all files in the output folder
    for filename in os.listdir(output_folder):
        file_path = os.path.join(output_folder, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")

def input(path):
    # image_path = "D:/My Projects/BITHACK/6.png"
    output_folder = "D:/My Projects/BITHACK/static/segmented_images_1"
    desired_width = 1200
    if is_single_line(path):
        segmented_lines=(bounding_box2(path))
        print_big(segmented_lines)
    else:
        print("This is a multi-line image. Deleting previously stored images and calling line segmentation function.")
        cleanup_output_folder(output_folder)
        segmented_lines = segment_text_lines(path, output_folder)
    saveimg(segmented_lines)
    return segmented_lines  
# input("D:/My Projects/BITHACK/9.jpg")






