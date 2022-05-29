# Make all other necessary imports.
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import numpy as np
import json
import argparse
from PIL import Image

import warnings
warnings.filterwarnings('ignore')

import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

# Function to process the image to the correct shape and size
def process_image(image):
    image_size = 224
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    image = tf.image.resize(image, (image_size, image_size))
    image /= 255
    return image.numpy()

# Function to predict the class of the image
def predict(image_path, model, top_k):
    from PIL import Image
    import numpy as np
    
    # Open the image to be predicted
    img = Image.open(image_path)
    pre_image = np.asarray(img)
    
    # Process the image to be fed into the prediction model
    processed_image = process_image(pre_image)
    processed_image = np.expand_dims(processed_image, axis = 0)

    # Predict the image
    predicted_image = model.predict(processed_image)

    # Get the prediction probabilities
    probs, indices = tf.nn.top_k(predicted_image, k=top_k)
    
    indices = indices.numpy()
    probs = probs.numpy()
    
    classes = [str(each+1) for each in indices[0]]

    return probs, classes

def parse_args():
    
    # Initialize the parser
    parser = argparse.ArgumentParser(description = 'Predict the class of an image')
    
    parser.add_argument('image_path', action='store', type=str, help='input image path') 
    parser.add_argument('model_path', action='store', type=str, help='prediction model to be used') 
    parser.add_argument('-k','--top_k', action='store', type=int, default=1, metavar='', help='top k probable class')
    parser.add_argument('-c','--class_names', action='store',type=str, metavar='', required=True, help='map class labels to flower names')
    
    args = parser.parse_args()
    return args
    
    
def main():
    #parse arguments
    args = parse_args()
      
    #load model
    model = tf.keras.models.load_model('./'+args.model_path, custom_objects={'KerasLayer': hub.KerasLayer}, compile=False)
       
    # load class labels from label map.json
    with open(args.class_names, 'r') as f:
        class_names = json.load(f)
      
    #path to image
    img_path = args.image_path
        
    #call predict function, which will call process image function
    probs, classes = predict(img_path, model, int(args.top_k))
      
    # map labels to class names
    labels = [class_names[str(index)] for index in classes]
    probability = probs[0]
    #print('File selected: ' + img_path)
    #print(labels)
    #print(probability)
    
    for i, j in zip(labels, probability):
        print('\nThe image class is:',i)
        print('\u2022The probability of the image class is:',j) 
              
    
if __name__ == '__main__':
    main()
