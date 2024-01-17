



# #!/usr/bin/env python
# # coding: utf-8

import sys

# print(sys.argv[1:])

AMAZON = "amazon"
DSLR = "dslr"
WEBCAM = "webcam"

DIR_RESULT = "result/{}_{}/".format(sys.argv[4],sys.argv[5])

DIR_AMAZON = "office31/{}".format(AMAZON)
DIR_DSLR = "office31/{}".format(DSLR)
DIR_WEBCAM = "office31/{}".format(WEBCAM)

################################## config #################################

# 0, 1, 2, 3, 4
# M, N, O = sys.argv[1], sys.argv[2], sys.argv[3]

DIR_SOURCE = "office31/{}".format(sys.argv[4])
DIR_TARGET = "office31/{}".format(sys.argv[5])

###########################################################################

if DIR_SOURCE == DIR_AMAZON:
    SOURCE = AMAZON
elif DIR_SOURCE == DIR_DSLR:
    SOURCE = DSLR
elif DIR_SOURCE == DIR_WEBCAM:
    SOURCE = WEBCAM
    
if DIR_TARGET == DIR_AMAZON:
    SOURCE = AMAZON
elif DIR_TARGET == DIR_DSLR:
    SOURCE = DSLR
elif DIR_TARGET == DIR_WEBCAM:
    SOURCE = WEBCAM

BATCH_SIZE = 64
IMG_HEIGHT = 128
IMG_WIDTH = 128
IMG_DIM = 3

LATENT_DIM = 128

EPOCHS = 100

scales = [0, 0.1, 0.3, 0.6, 0.01]

W_CATEGORICAL = float(sys.argv[1])
W_ADVERSARIAL = float(sys.argv[2])
W_DOMAIN      = float(sys.argv[3])
    
TITLE = "{}_{} = {}-{}-{}".format(sys.argv[4],sys.argv[5], W_CATEGORICAL, W_ADVERSARIAL, W_DOMAIN)
    
print(DIR_RESULT + TITLE + '.pkl')
print(DIR_SOURCE, DIR_TARGET)

# In[2]:


# TITLE


# In[3]:


import tensorflow as tf

tf.random.set_seed(1202)

class DataLoader:

    def __init__(self, source, target, BATCH_SIZE=64, IMG_HEIGHT = 128, IMG_WIDTH = 128, IMG_DIM = 3):
        self.source = source
        self.target = target
        self.BATCH_SIZE = BATCH_SIZE
        self.IMG_HEIGHT = IMG_HEIGHT
        self.IMG_WIDTH = IMG_WIDTH
        self.IMG_DIM = IMG_DIM
        self.normalization_layer = tf.keras.layers.Rescaling(1./255)

    def load_source_train(self):
        dataset_source =  tf.keras.utils.image_dataset_from_directory(
                                self.source,
                                validation_split=0.2,
                                subset="training",
                                seed=123,
                                image_size=(self.IMG_HEIGHT, self.IMG_WIDTH),
                                batch_size=self.BATCH_SIZE)

        return dataset_source.map(lambda x, y: (self.normalization_layer(x), y))


    def load_source_validation(self):
        dataset_source_val =  tf.keras.utils.image_dataset_from_directory(
                                self.source,
                                validation_split=0.2,
                                subset="validation",
                                seed=123,
                                image_size=(self.IMG_HEIGHT, self.IMG_WIDTH),
                                batch_size=self.BATCH_SIZE)

        return dataset_source_val.map(lambda x, y: (self.normalization_layer(x), y))

    def load_target_test(self):
        dataset_target =  tf.keras.utils.image_dataset_from_directory(
                                self.target,
                                seed=123,
                                image_size=(self.IMG_HEIGHT, self.IMG_WIDTH),
                                batch_size=self.BATCH_SIZE)

        return dataset_target.map(lambda x, y: (self.normalization_layer(x), y))

    def load(self):
        return tf.data.Dataset.zip((self.load_source_train(), self.load_source_validation(), self.load_target_test()))


# # Model

# In[4]:


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


# In[5]:


import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers


class AML(tf.keras.Model):
    """Adeversarial Multitask Learning."""

    def __init__(self, weight_adversarial=0, weight_categorical=0, weight_domain=0):
        super(AML, self).__init__()

        self.InceptionV3 = tf.keras.applications.InceptionV3(
                            weights='imagenet',
                            input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_DIM),
                            include_top=False)

        self.InceptionV3.trainable = False

        self.feature_extractor = keras.Sequential(
                [
                    self.InceptionV3,
                    layers.GlobalAveragePooling2D(),
                    layers.Flatten(),
                    layers.Dropout(0.5),
                    layers.Dense(512, activation='relu', kernel_initializer='he_uniform'),

                ],
                name="feature_extractor",
            )

        self.categorical_classifier = keras.Sequential(
                [
                    layers.Dense(128, activation='relu', kernel_initializer='he_uniform'),
                    layers.Dropout(0.3),
                    layers.Dense(31, activation='softmax'),
                ],
                name="categorical_classifier",
            )

        self.domain_classifier = keras.Sequential(
            [
                layers.Dense(128, activation='relu', kernel_initializer='he_uniform'),
                layers.Dropout(0.3),
                GradReverse(),
                layers.Dense(1),
            ],
            name="domain_classifier",
        )

        self.generator = keras.Sequential(
            [
                keras.Input(shape=(LATENT_DIM,)),
                layers.Dense(8 * 8 * 128),
                layers.LeakyReLU(alpha=0.2),
                layers.Reshape((8, 8, 128)),
                layers.Conv2DTranspose(128, (4, 4), strides=(4, 4), padding="same"),
                layers.LeakyReLU(alpha=0.2),
                layers.Conv2DTranspose(128, (4, 4), strides=(4, 4), padding="same"),
                layers.LeakyReLU(alpha=0.2),
                layers.Conv2D(3, (8, 8), padding="same", activation="sigmoid"),
            ],
            name="generator",
        )

        self.discriminator = keras.Sequential(
            [
                layers.Dense(1),
            ],
            name="discriminator",
        )

        self.weight_categorical = weight_categorical
        self.weight_adversarial = weight_adversarial
        self.weight_domain      = weight_domain

        self.d_optimizer = keras.optimizers.Adam()
        self.g_optimizer = keras.optimizers.Adam()
        self.c_optimizer = keras.optimizers.Adam()
        self.fe_optimizer = keras.optimizers.Adam()
        self.domain_optimizer = keras.optimizers.Adam()

        self.val_accuracy = tf.keras.metrics.Accuracy()
        self.train_accuracy = tf.keras.metrics.Accuracy()
        self.target_accuracy = tf.keras.metrics.Accuracy()
        
        self.loss_fn = keras.losses.BinaryCrossentropy(from_logits=True)
        self.loss_fn_cls = keras.losses.SparseCategoricalCrossentropy(from_logits=False)



    def sample(self):
        random_latent_vectors = tf.random.normal(shape=(BATCH_SIZE, LATENT_DIM))
        data = self.generator(random_latent_vectors)
        features = self.feature_extractor(data)
        cat_cls = self.categorical_classifier(features)
        dom_cls = self.domain_classifier(features)
        dis_cls = self.discriminator(features)

        return cat_cls, dom_cls, dis_cls

    def predict_domain(self, x):
        features = self.feature_extractor(x)
        return self.domain_classifier(features)

    def predict_category(self, x):
        features = self.feature_extractor(x)
        return self.categorical_classifier(features)

    def predict_adversarial(self, x):
        features = self.feature_extractor(x)
        return self.discriminator(features)

    def generate_data(self):
        random_latent_vectors = tf.random.normal(shape=(BATCH_SIZE, LATENT_DIM))
        return self.generator(random_latent_vectors)


# In[6]:


@tf.function
def train_step(model, real_images, real_label, test_images, test_label, target_images, target_label):

    #adversarial data
    generated_images = model.generate_data()
    combined_images = tf.concat([generated_images, real_images], axis=0)
#     combined_labels = tf.concat([tf.ones((BATCH_SIZE, 1)), tf.zeros((BATCH_SIZE, 1))], axis=0)
    combined_labels = tf.concat([tf.ones((BATCH_SIZE, 1)), tf.zeros((real_images.shape[0], 1))], axis=0)
    combined_labels += 0.05 * tf.random.uniform(combined_labels.shape)

    #domain
    combined_domain = tf.concat([target_images, real_images], axis=0)
#     labels_domain = tf.concat([tf.ones((BATCH_SIZE, 1)), tf.zeros((BATCH_SIZE, 1))], axis=0)
    labels_domain = tf.concat([tf.ones((target_images.shape[0], 1)), tf.zeros((real_images.shape[0], 1))], axis=0)
    labels_domain += 0.05 * tf.random.uniform(labels_domain.shape)

    # Train the discriminator, cat_classifier, dom_classifier
    with tf.GradientTape(persistent=True) as tape:

        predictions_disc = model.predict_adversarial(combined_images)
        d_loss = model.loss_fn(combined_labels, predictions_disc)

        predictions_clas = model.predict_category(real_images)
        c_loss = model.loss_fn_cls(real_label, predictions_clas)

        predictions_domain = model.predict_domain(combined_domain)
        domain_loss = -1 * model.loss_fn(labels_domain, predictions_domain)
        # domain_loss = -1 * domain_loss

        fe_loss = model.weight_adversarial * d_loss + model.weight_categorical * c_loss + model.weight_domain * domain_loss

    grads_feature_extractor = tape.gradient(fe_loss, model.feature_extractor.trainable_weights)
    model.fe_optimizer.apply_gradients(zip(grads_feature_extractor, model.feature_extractor.trainable_weights))

    grads_discriminator = tape.gradient(d_loss, model.discriminator.trainable_weights)
    model.d_optimizer.apply_gradients(zip(grads_discriminator, model.discriminator.trainable_weights))

    grads_categorical = tape.gradient(c_loss, model.categorical_classifier.trainable_weights)
    model.c_optimizer.apply_gradients(zip(grads_categorical, model.categorical_classifier.trainable_weights))

    grads_domain = tape.gradient(domain_loss, model.domain_classifier.trainable_weights)
    model.domain_optimizer.apply_gradients(zip(grads_domain, model.domain_classifier.trainable_weights))

    # Train generator
    misleading_labels = tf.zeros((BATCH_SIZE, 1))
    with tf.GradientTape() as tape:
        generated_images = model.generate_data()
        predictions = model.predict_adversarial(generated_images)
        g_loss = model.loss_fn(misleading_labels, predictions)
    grads = tape.gradient(g_loss, model.generator.trainable_weights)
    model.g_optimizer.apply_gradients(zip(grads, model.generator.trainable_weights))

    c_acc_t = model.train_accuracy(tf.math.argmax(model.predict_category(real_images), 1) , real_label)
    c_acc_v = model.val_accuracy(tf.math.argmax(model.predict_category(test_images), 1), test_label)
    c_acc_target = model.target_accuracy(tf.math.argmax(model.predict_category(target_images), 1), target_label)

    return c_acc_target, c_acc_v, c_acc_t, domain_loss, c_loss, d_loss, g_loss, fe_loss


# In[7]:


dataset = DataLoader(DIR_SOURCE, DIR_TARGET).load()


# In[8]:


# @tf.function
def train(model, dataset, EPOCHS):

    gl_, al_, cl_, dl_, tl_, acc_source_, acc_target_, acc_source_val_ = [], [], [], [], [], [], [], []
    
    for epoch in range(EPOCHS):

        start = time.time()
        
        model.train_accuracy.reset_state()
        model.val_accuracy.reset_state()
        model.target_accuracy.reset_state()
        
        for step, ((real_images, real_label), (test_images, test_label), (target_images, target_label)) in enumerate(dataset):
            
            
            acc_target, acc_source_val, acc_source, domain_loss, categorical_loss, adversarial_loss, generative_loss, fe_loss = train_step(model, real_images, real_label, test_images, test_label, target_images, target_label)
            
            
            gl_ += [generative_loss]
            al_ += [adversarial_loss]
            cl_ += [categorical_loss]
            dl_ += [-1 * domain_loss]
            tl_ += [fe_loss]
            acc_source_ += [acc_source]
            acc_target_ += [acc_target]
            acc_source_val_ +=[acc_source_val]
            
        done = time.time()
        elapsed = done - start
        
        if(epoch % 10 == 0):
            print("Epoch {} - Time {}".format(epoch, elapsed))

    return gl_, al_, cl_, dl_, tl_, acc_source_, acc_target_, acc_source_val_


# In[9]:


import numpy as np
import matplotlib.pyplot as plt

def plot_loss_values(gl_, al_, cl_, dl_, tl_):
    x = np.arange(len(gl_))

    plt.plot(x, gl_, label = "generative loss", linestyle="-.")
    plt.plot(x, al_, label = "adversarial loss", linestyle="-")
    plt.plot(x, cl_, label = "categorical loss", linestyle="--")
    plt.plot(x, dl_, label = "domain loss", linestyle=":")
    plt.plot(x, tl_, label = "total loss", linestyle=(0, (3, 1, 1, 1)))
    plt.legend()
    plt.show()


# In[10]:


def plot_acc_values(acc_source_, acc_target_, acc_source_val_):
    x = np.arange(len(acc_source_))
    plt.plot(x, acc_source_, label = "source acc", linestyle="-")
    plt.plot(x, acc_target_, label = "target acc", linestyle=":")
    plt.plot(x, acc_source_val_, label = "val acc", linestyle="-.")
    plt.legend()
    plt.show()


# # Training Loop

# In[11]:

import time
import pickle

FinalResult = []
            
model = AML(weight_adversarial=W_ADVERSARIAL, weight_categorical=W_CATEGORICAL, weight_domain=W_DOMAIN)
model.sample()

print(TITLE)

gl_, al_, cl_, dl_, tl_, acc_source_, acc_target_, acc_source_val_ = train(model, dataset, EPOCHS)


# In[12]:


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

FinalResult += [tempResult]


# In[13]:


with open(DIR_RESULT + TITLE + '.pkl', 'wb') as fp:
    pickle.dump({'result':FinalResult}, fp)


# In[ ]:




