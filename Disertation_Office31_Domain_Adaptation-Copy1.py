#!/usr/bin/env python
# coding: utf-8

# # Setup

# In[1]:


from IPython.display import clear_output
import matplotlib.pyplot as plt

import pickle
import tensorflow as tf
import keras
from keras import layers
import numpy as np
import os


# In[2]:


DIR_AMAZON = "office31/amazon"
DIR_DSLR = "office31/dslr"
DIR_WEBCAM = "office31/webcam"


# In[3]:


BATCH_SIZE = 64
IMG_HEIGHT = 128
IMG_WIDTH = 128
IMG_DIM = 3

EPOCHS = 100

# W_CATEGORICAL = 0.1
# W_ADVERSARIAL = 0
# W_DOMAIN      = 1

SOURCE = DIR_DSLR
TARGET = DIR_WEBCAM

latent_dim = 128


# # Preparation Part

# In[4]:


dataset_source = tf.keras.utils.image_dataset_from_directory(
  SOURCE,                                                     # change DIR according to the dataset
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(IMG_HEIGHT, IMG_WIDTH),
  batch_size=BATCH_SIZE)


# In[5]:


dataset_source_val = tf.keras.utils.image_dataset_from_directory(
  SOURCE,                                                     # change DIR according to the dataset
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(IMG_HEIGHT, IMG_WIDTH),
  batch_size=BATCH_SIZE)


# In[6]:


dataset_target = tf.keras.utils.image_dataset_from_directory(
  TARGET,                                                     # change DIR according to the dataset
  seed=123,
  image_size=(IMG_HEIGHT, IMG_WIDTH),
  batch_size=BATCH_SIZE)


# In[7]:


normalization_layer = tf.keras.layers.Rescaling(1./255)

dataset_source = dataset_source.map(lambda x, y: (normalization_layer(x), y))
dataset_source_val = dataset_source_val.map(lambda x, y: (normalization_layer(x), y))
dataset_target = dataset_target.map(lambda x, y: (normalization_layer(x), y))


# In[8]:


dataset = tf.data.Dataset.zip((dataset_source, dataset_source_val, dataset_target))


# In[9]:


@tf.custom_gradient
def grad_reverse(x):
    y = tf.identity(x)
    def custom_grad(dy):
        return -dy
    return y, custom_grad

class GradReverse(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, x):
        return grad_reverse(x)


# In[10]:


@tf.function
def train_step(real_images, real_label, test_images, test_label, target_images, target_label):
    # Sample random points in the latent space
    random_latent_vectors = tf.random.normal(shape=(BATCH_SIZE, latent_dim))
    # Decode them to fake images
    generated_images = generator(random_latent_vectors)
    # Combine them with real images
    combined_images = tf.concat([generated_images, real_images], axis=0)

    # Assemble labels discriminating real from fake images
    labels = tf.concat(
        [tf.ones((BATCH_SIZE, 1)), tf.zeros((real_images.shape[0], 1))], axis=0
    )
    # Add random noise to the labels - important trick!
    labels += 0.05 * tf.random.uniform(labels.shape)

    combined_domain = tf.concat([target_images, real_images], axis=0)

    # Assemble labels classifying target from source images
    labels_domain = tf.concat(
        [tf.ones((target_images.shape[0], 1)), tf.zeros((real_images.shape[0], 1))], axis=0
    )
    # Add random noise to the labels - important trick!
    labels_domain += 0.05 * tf.random.uniform(labels_domain.shape)



    # Train the discriminator
    with tf.GradientTape(persistent=True) as tape:
        features = feature_extractor(combined_images)
        predictions_disc = discriminator(features)
        # predictions = discriminator(combined_images)
        d_loss = loss_fn(labels, predictions_disc)

        features = feature_extractor(real_images)
        predictions_clas = categorical_classifier(features)
        # predictions = classifier(real_images)
        c_loss = loss_fn_cls(real_label, predictions_clas)

        features = feature_extractor(combined_domain)
        predictions_domain = domain_classifier(features)
        domain_loss = loss_fn(labels_domain, predictions_domain)
        domain_loss = -1 * domain_loss

        # fe_loss = total_loss(predictions_disc, labels, predictions_clas, real_label)
        fe_loss = W_ADVERSARIAL * d_loss + W_CATEGORICAL * c_loss + W_DOMAIN * domain_loss

    grads_feature_extractor = tape.gradient(fe_loss, feature_extractor.trainable_weights)
    fe_optimizer.apply_gradients(zip(grads_feature_extractor, feature_extractor.trainable_weights))

    grads_discriminator = tape.gradient(d_loss, discriminator.trainable_weights)
    d_optimizer.apply_gradients(zip(grads_discriminator, discriminator.trainable_weights))

    grads_categorical = tape.gradient(c_loss, categorical_classifier.trainable_weights)
    c_optimizer.apply_gradients(zip(grads_categorical, categorical_classifier.trainable_weights))

    grads_domain = tape.gradient(domain_loss, domain_classifier.trainable_weights)
    domain_optimizer.apply_gradients(zip(grads_domain, domain_classifier.trainable_weights))



    #################################

    # Sample random points in the latent space
    random_latent_vectors = tf.random.normal(shape=(BATCH_SIZE, latent_dim))
    # Assemble labels that say "all real images"
    misleading_labels = tf.zeros((BATCH_SIZE, 1))

    # Train the generator (note that we should *not* update the weights
    # of the discriminator)!
    with tf.GradientTape() as tape:
        features = feature_extractor(generator(random_latent_vectors))
        predictions = discriminator(features)
        # predictions = discriminator(generator(random_latent_vectors))
        g_loss = loss_fn(misleading_labels, predictions)
    grads = tape.gradient(g_loss, generator.trainable_weights)
    g_optimizer.apply_gradients(zip(grads, generator.trainable_weights))


    c_acc_t = train_accuracy(tf.math.argmax(categorical_classifier(feature_extractor(real_images)), 1) , real_label)
    c_acc_v = val_accuracy(tf.math.argmax(categorical_classifier(feature_extractor(test_images)), 1), test_label)
    c_acc_target = target_accuracy(tf.math.argmax(categorical_classifier(feature_extractor(target_images)), 1), target_label)

    return c_acc_target, c_acc_v, c_acc_t, domain_loss, c_loss, d_loss, g_loss, fe_loss


# In[11]:


def plot_loss_values(gl_, al_, cl_, dl_, tl_):
  x = np.arange(len(gl_))

  plt.plot(x, gl_, label = "generative loss", linestyle="-.")
  plt.plot(x, al_, label = "adversarial loss", linestyle="-")
  plt.plot(x, cl_, label = "categorical loss", linestyle="--")
  plt.plot(x, dl_, label = "domain loss", linestyle=":")
  plt.plot(x, tl_, label = "total loss", linestyle=(0, (3, 1, 1, 1)))
  plt.legend()
  plt.show()


# In[12]:


def plot_acc_values(acc_source_, acc_target_, acc_source_val_):
  x = np.arange(len(acc_source_))
  plt.plot(x, acc_source_, label = "source acc", linestyle="-")
  plt.plot(x, acc_target_, label = "target acc", linestyle=":")
  plt.plot(x, acc_source_val_, label = "val acc", linestyle="-.")
  plt.legend()
  plt.show()


# # Training LOOP

# In[ ]:





# In[ ]:


# W_CATEGORICAL = 0.1
# W_ADVERSARIAL = 0
# W_DOMAIN      = 1

FinalResult = []

for W_CATEGORICAL in np.arange(0,1,.1):
    for W_ADVERSARIAL in np.arange(0,1,.1):
        for W_DOMAIN in np.arange(0,1,.1):
            
            feature_extractor = tf.keras.applications.InceptionV3(
                weights='imagenet',
                input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_DIM),
                include_top=False)

            feature_extractor.trainable = False

            feature_extractor = keras.Sequential(
                [
                    feature_extractor,
                    layers.GlobalAveragePooling2D(),
                    layers.Flatten(),
                    layers.Dropout(0.5),
                    layers.Dense(512, activation='relu', kernel_initializer='he_uniform'),

                ],
                name="feature_extractor",
            )

            for x, y in dataset_source.take(1):
              features = feature_extractor(x)

            categorical_classifier = keras.Sequential(
                [
                    layers.Dense(128, activation='relu', kernel_initializer='he_uniform'),
                    layers.Dropout(0.3),
                    layers.Dense(31, activation='softmax'),
                ],
                name="categorical_classifier",
            )
            cls = categorical_classifier(features)

            domain_classifier = keras.Sequential(
                [
                    layers.Dense(128, activation='relu', kernel_initializer='he_uniform'),
                    layers.Dropout(0.3),
                    GradReverse(),
                    layers.Dense(1),
                ],
                name="domain_classifier",
            )
            feature_maps = domain_classifier(features)

            discriminator = keras.Sequential(
                [
                    # layers.Dense(128, activation='relu', kernel_initializer='he_uniform'),
                    # layers.Dropout(0.3),
                    layers.Dense(1),
                ],
                name="discriminator",
            )
            disc = discriminator(features)

            #RGB



            generator = keras.Sequential(
                [
                    keras.Input(shape=(latent_dim,)),
                    # We want to generate 128 coefficients to reshape into a 7x7x128 map
                    layers.Dense(8 * 8 * 128),
                    layers.LeakyReLU(alpha=0.2),
                    layers.Reshape((8, 8, 128)),
                    layers.Conv2DTranspose(128, (4, 4), strides=(4, 4), padding="same"),
                    layers.LeakyReLU(alpha=0.2),
                    layers.Conv2DTranspose(128, (4, 4), strides=(4, 4), padding="same"),
                    layers.LeakyReLU(alpha=0.2),
                    # layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"),
                    # layers.LeakyReLU(alpha=0.2),
                    layers.Conv2D(3, (8, 8), padding="same", activation="sigmoid"),
                ],
                name="generator",
            )

            gen_data = generator(tf.random.normal(shape=(BATCH_SIZE, latent_dim)))


            # Instantiate one optimizer for the discriminator and another for the generator.
            d_optimizer = keras.optimizers.Adam()
            g_optimizer = keras.optimizers.Adam()
            c_optimizer = keras.optimizers.Adam()
            fe_optimizer = keras.optimizers.Adam()
            domain_optimizer = keras.optimizers.Adam()

            # Instantiate a loss function.
            loss_fn = keras.losses.BinaryCrossentropy(from_logits=True)
            loss_fn_cls = keras.losses.SparseCategoricalCrossentropy(from_logits=False)

            val_accuracy = tf.keras.metrics.Accuracy()
            train_accuracy = tf.keras.metrics.Accuracy()
            target_accuracy = tf.keras.metrics.Accuracy()


            ###################################################################

            gl_, al_, cl_, dl_, tl_, acc_source_, acc_target_, acc_source_val_ = [], [], [], [], [], [], [], []

            for epoch in range(EPOCHS):


                for step, ((real_images, real_label), (test_images, test_label), (target_images, target_label)) in enumerate(dataset):
                    # Train the discriminator & generator on one batch of real images.
                    acc_target, acc_source_val, acc_source, domain_loss, categorical_loss, adversarial_loss, generative_loss, fe_loss = train_step(real_images, real_label, test_images, test_label, target_images, target_label)

                    gl_ += [generative_loss]
                    al_ += [adversarial_loss]
                    cl_ += [categorical_loss]
                    dl_ += [domain_loss]
                    tl_ += [fe_loss]
                    acc_source_ += [acc_source]
                    acc_target_ += [acc_target]
                    acc_source_val_ +=[acc_source_val]

                    # Logging.
            #         if step % 50 == 0:
            #             # Print metrics
#                 clear_output(wait=True)
#                 print("Epoch: ", epoch)
#                 print("\ndomain loss at step %d: %.3f" % (step, domain_loss))
#                 print("discriminator loss at step %d: %.3f" % (step, adversarial_loss))
#                 print("adversarial loss at step %d: %.3f" % (step, generative_loss))
#                 print("categorical loss at step %d: %.3f" % (step, categorical_loss))
#                 print("\ncategorical train Accu at step %d: %.3f" % (step, acc_source))
#                 print("categorical val Accu at step %d: %.3f" % (step, acc_source_val))
#                 print("categorical target Accu at step %d: %.3f" % (step, acc_target))

#                 plot_loss_values(gl_, al_, cl_, dl_, tl_)
#                 plot_acc_values(acc_source_, acc_target_, acc_source_val_)
            tempResult = {'W_CATEGORICAL' : W_CATEGORICAL,
                          'W_ADVERSARIAL' : W_ADVERSARIAL,
                          'W_DOMAIN'      : W_DOMAIN,
                          'gl_':gl_, 
                          'al_':al_, 
                          'cl_':cl_, 
                          'dl_':dl_, 
                          'tl_':tl_, 
                          'acc_source_':acc_source_, 
                          'acc_target_':acc_target_, 
                          'acc_source_val_':acc_source_val_}
            FinalResult+=[tempResult]

            # save dictionary to person_data.pkl file
            print('W_CATEGORICAL: ', W_CATEGORICAL,
                  'W_ADVERSARIAL: ' , W_ADVERSARIAL,
                  'W_DOMAIN: ' , W_DOMAIN)
            

            
with open('result.pkl', 'wb') as fp:
    pickle.dump({'result':FinalResult}, fp)


# In[ ]:


import pickle

# Read dictionary pkl file
with open('data.pkl', 'rb') as fp:
    person = pickle.load(fp)
    print('Person dictionary')
    print(person)


# In[ ]:


person['result'][0].keys()


# In[ ]:


def plot_loss_values(gl_, al_, cl_, dl_, tl_):
  x = np.arange(len(gl_))

  plt.plot(x, gl_, label = "generative loss", linestyle="-.")
  plt.plot(x, al_, label = "adversarial loss", linestyle="-")
  plt.plot(x, cl_, label = "categorical loss", linestyle="--")
  plt.plot(x, dl_, label = "domain loss", linestyle=":")
  plt.plot(x, tl_, label = "total loss", linestyle=(0, (3, 1, 1, 1)))
  plt.legend()
  plt.show()




# In[ ]:


def plot_acc_values(acc_source_, acc_target_, acc_source_val_):
  x = np.arange(len(acc_source_))
  plt.plot(x, acc_source_, label = "source acc", linestyle="-")
  plt.plot(x, acc_target_, label = "target acc", linestyle=":")
  plt.plot(x, acc_source_val_, label = "val acc", linestyle="-.")
  plt.legend()
  plt.show()


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


for i in range(5):
    print("W_CATEGORICAL: ", person['result'][i]['W_CATEGORICAL'])
    print("W_ADVERSARIAL: ", person['result'][i]['W_ADVERSARIAL'])
    print("W_DOMAIN: ", person['result'][i]['W_DOMAIN'])
    
    plot_loss_values(person['result'][i]['gl_'], person['result'][i]['al_'], person['result'][i]['cl_'], person['result'][i]['dl_'], person['result'][i]['tl_'])


# In[ ]:


person['result'][0]['gl_'][0].numpy()


# In[ ]:


recap_acc = person['result']


# In[ ]:


recap_acc


# In[ ]:


get_ipython().system('pip install pandas')


# In[ ]:


import pandas as pd

df = pd.DataFrame.from_records(recap_acc)
df['gl_'] = df['gl_'].map(lambda x: x[-1].numpy())
df['al_'] = df['al_'].map(lambda x: x[-1].numpy())
df['cl_'] = df['cl_'].map(lambda x: x[-1].numpy())
df['dl_'] = df['dl_'].map(lambda x: x[-1].numpy())
df['tl_'] = df['tl_'].map(lambda x: x[-1].numpy())

df['acc_source_'] = df['acc_source_'].map(lambda x: x[-1].numpy())
df['acc_target_'] = df['acc_target_'].map(lambda x: x[-1].numpy())
df['acc_source_val_'] = df['acc_source_val_'].map(lambda x: x[-1].numpy())


# In[ ]:


df.head()


# In[ ]:


df.describe()


# In[ ]:


df.plot()


# In[ ]:




