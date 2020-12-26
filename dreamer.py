#!/usr/bin/env python
# coding: utf-8

# In[91]:


import tensorflow as tf


# In[137]:


import numpy as np 

import os

import matplotlib as mpl

import IPython.display as display
import PIL.Image

from tensorflow.keras.preprocessing import image

import glob

import random

path = "imgs/img1.jpg"


def open_image(path, max_dim=None):
   img = PIL.Image.open(path)
   if max_dim:
      img.thumbnail((max_dim, max_dim))
   return np.array(img)



# Download an image and read it into a NumPy array.
def download(url, max_dim=None):
   name = url.split('/')[-1]
   image_path = tf.keras.utils.get_file(name, origin=url)
   img = PIL.Image.open(image_path)
   if max_dim:
      img.thumbnail((max_dim, max_dim))
   return np.array(img)


# Normalize an image
def deprocess(img):
  img = 255*(img + 1.0)/2.0
  return tf.cast(img, tf.uint8)

#Save an image
def save_image(image, filename):
    # Ensure the pixel-values are between 0 and 255.
    image = np.clip(image, 0.0, 255.0)
    
    # Convert to bytes.
    image = image.astype(np.uint8)
    
    with open(filename, 'wb') as file:
        PIL.Image.fromarray(image).save(filename, 'png')
        
      
 
# Display an image
def show(img):
  display.display(PIL.Image.fromarray(np.array(img)))


#original_img = download(url, max_dim=850)
original_img = open_image(path, max_dim=None)

base_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')


# Maximize the activations of these layers
names = ['mixed8']
layers = [base_model.get_layer(name).output for name in names]

# Create the feature extraction model
dream_model = tf.keras.Model(inputs=base_model.input, outputs=layers)


# ## Calculate loss

def calc_loss(img, model):
  # Pass forward the image through the model to retrieve the activations.
  # Converts the image into a batch of size 1.
  img_batch = tf.expand_dims(img, axis=0)
  layer_activations = model(img_batch)
  if len(layer_activations) == 1:
    layer_activations = [layer_activations]

  losses = []
  for act in layer_activations:
    loss = tf.math.reduce_mean(act)
    losses.append(loss)

  return  tf.reduce_sum(losses)


# ## Gradient ascent

class DeepDream(tf.Module):
  def __init__(self, model):
    self.model = model

  @tf.function(
      input_signature=(
        tf.TensorSpec(shape=[None,None,3], dtype=tf.float32),
        tf.TensorSpec(shape=[], dtype=tf.int32),
        tf.TensorSpec(shape=[], dtype=tf.float32),)
  )
  def __call__(self, img, steps, step_size):
      print("Tracing")
      loss = tf.constant(0.0)
      for n in tf.range(steps):
        with tf.GradientTape() as tape:
          # This needs gradients relative to `img`
          # `GradientTape` only watches `tf.Variable`s by default
          tape.watch(img)
          loss = calc_loss(img, self.model)

        # Calculate the gradient of the loss with respect to the pixels of the input image.
        gradients = tape.gradient(loss, img)

        # Normalize the gradients.
        gradients /= tf.math.reduce_std(gradients) + 1e-8 
        
        # In gradient ascent, the "loss" is maximized so that the input image increasingly "excites" the layers.
        # You can update the image by directly adding the gradients (because they're the same shape!)
        img = img + gradients*step_size
        img = tf.clip_by_value(img, -1, 1)
        

      return loss, img

deepdream = DeepDream(dream_model)

# ## Main Loop

def run_deep_dream_simple(img, steps=150, step_size=0.01, num_frames = 10):
  # Convert from uint8 to the range expected by the model.
  img = tf.keras.applications.inception_v3.preprocess_input(img)
  img = tf.convert_to_tensor(img)
  step_size = tf.convert_to_tensor(step_size)
  steps_remaining = steps
  step = 0
  rand_name = str(random.randint(0,999))
  outputname = "\\Users\Le Orel\deepdream\\" + rand_name
  os.mkdir(outputname)
  print(outputname)
    
  while steps_remaining:
    if steps_remaining>100:
      run_steps = tf.constant(100)
    else:
      run_steps = tf.constant(steps_remaining)
    steps_remaining -= run_steps
    step += run_steps
    frames = []
    for k in range(num_frames):
        loss, img = deepdream(img, run_steps, tf.constant(step_size))
        result = deprocess(img)
 
        save_image(result, outputname+ "\\"+str(k)+ ".png")
        frames.append(result)
    
    display.clear_output(wait=True)
    print ("Step {}, loss {}".format(step, loss))


  display.clear_output(wait=True)
  show(result)

  return result,rand_name


num_frames_calc = 50
dream_img, filepath = run_deep_dream_simple(img=original_img, steps=3, step_size=0.01, num_frames = num_frames_calc)


#Save the frames as a gif
make_movie = "ffmpeg -r " + str(num_frames_calc) + " -i " + filepath + "\\%01d.png -y " + filepath+ "\\vid.gif"
os.system(make_movie)

