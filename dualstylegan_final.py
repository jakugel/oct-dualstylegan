#Imports
import numpy as np
import time
from functools import partial
import os
from keras.layers import Add, Layer
from PIL import Image
from matplotlib import colors

#Imports for layers and models
from keras.layers import Conv2D, Dense, AveragePooling2D, LeakyReLU, Activation
from keras.layers import Reshape, UpSampling2D, Dropout, Flatten, Input, add, Cropping2D
from keras.models import Model
from keras.optimizers import Adam
import keras.backend as K

from GAN.AdaIN import AdaInstanceNormalization

region_colours = ['#fde8ff', '#4285f4', '#db4437', '#f4b400', '#0f9d58', '#ff6d00', '#46bdc6', '#ab30c4', '#0e0d5e']
region_cmap = colors.ListedColormap(region_colours)

results_path = "dualstylegan_final_Results"
models_path = "dualstylegan_final_Models"

#Config Stuff
im_size = 128
num_layers = np.log2(im_size) - 2
latent_size = 128
BATCH_SIZE = 16
lrelucoef = 0.01
gplossweight = 50

# weighted sum output
class WeightedSum(Add):
    # init with default value
    def __init__(self, alpha=0.0, **kwargs):
        super(WeightedSum, self).__init__(**kwargs)
        self.alpha = K.variable(alpha, name='ws_alpha')

    # output a weighted sum of inputs
    def _merge_function(self, inputs):
        # only supports a weighted sum of two inputs
        assert (len(inputs) == 2)
        # ((1-a) * input1) + (a * input2)
        output = ((1.0 - self.alpha) * inputs[0]) + (self.alpha * inputs[1])
        return output


# update the alpha value on each instance of WeightedSum
def update_fadein(layers, step, n_steps):
    # calculate current alpha (linear from 0 to 1)
    alpha = step / float(n_steps - 1)
    for layer in layers:
        K.set_value(layer.alpha, alpha)


class MinibatchStdev(Layer):
    # initialize the layer
    def __init__(self, **kwargs):
        super(MinibatchStdev, self).__init__(**kwargs)

    # perform the operation
    def call(self, inputs):
        # calculate the mean value for each pixel across channels
        mean = K.mean(inputs, axis=0, keepdims=True)
        # calculate the squared differences between pixel values and mean
        squ_diffs = K.square(inputs - mean)
        # calculate the average of the squared differences (variance)
        mean_sq_diff = K.mean(squ_diffs, axis=0, keepdims=True)
        # add a small value to avoid a blow-up when we calculate stdev
        mean_sq_diff += 1e-8
        # square root of the variance (stdev)
        stdev = K.sqrt(mean_sq_diff)
        # calculate the mean standard deviation across each pixel coord
        mean_pix = K.mean(stdev, keepdims=True)
        # scale this up to be the size of one input feature map for each sample
        shape = K.shape(inputs)
        output = K.tile(mean_pix, (shape[0], shape[1], shape[2], 1))
        # concatenate with the output
        combined = K.concatenate([inputs, output], axis=-1)
        return combined

    # define the output shape of the layer
    def compute_output_shape(self, input_shape):
        # create a copy of the input shape as a list
        input_shape = list(input_shape)
        # add one to the channel dimension (assume channels-last)
        input_shape[-1] += 1
        # convert list to a tuple
        return tuple(input_shape)


#Style Z
def noise(n):
    return np.random.normal(0.0, 1.0, size = [n, latent_size])

#Noise Sample
def noiseImage(n):
    return np.random.uniform(0.0, 1.0, size = [n, im_size, im_size, 1])

#Get random samples from an array
def get_rand(array, amount):
    
    idx = np.random.randint(0, array.shape[0], amount)
    return array[idx]

#Import Images Function
def import_images():
    # load images and corresponding mask labels here
    # both should be of shape: (number of samples, im_size, im_size, 1)
    images = np.array([])
    labels = np.array([])

    trains = []
    trains_up = [None]

    for i in range(int(np.log2(128) - 1)):
        if i == 0:
            x_train = images
            y_train = labels

            train = np.concatenate([x_train, y_train], axis=-1)

            trains.append(train)
        else:
            new_shape = int(im_size / (2 ** i))
            train = []
            trainup = []

            for j in range(images.shape[0]):
                # resize with nearest neighbor interpolation
                new_image = np.array(Image.fromarray(np.squeeze(images[j])).resize((new_shape, new_shape), resample=0))
                new_label = np.array(Image.fromarray(np.squeeze(labels[j])).resize((new_shape, new_shape), resample=0))

                # resize back to original with nearest neighbor interpolation
                new_imageup = np.array(
                    Image.fromarray(np.squeeze(new_image)).resize((new_shape * 2, new_shape * 2), resample=0))
                new_labelup = np.array(
                    Image.fromarray(np.squeeze(new_label)).resize((new_shape * 2, new_shape * 2), resample=0))

                #
                # # store

                new_train = np.concatenate([np.expand_dims(new_image, axis=-1), np.expand_dims(new_label, axis=-1)], axis=-1)
                new_trainup = np.concatenate([np.expand_dims(new_imageup, axis=-1), np.expand_dims(new_labelup, axis=-1)], axis=-1)

                train.append(new_train)
                trainup.append(new_trainup)

            train = np.asarray(train)
            trains.append(train)
            trainup = np.asarray(trainup)
            trains_up.append(trainup)

    return trains, trains_up


#r1/r2 gradient penalty
def gradient_penalty_loss(y_true, y_pred, averaged_samples, weight):
    gradients = K.gradients(y_pred, averaged_samples)[0]
    gradients_sqr = K.square(gradients)
    gradient_penalty = K.sum(gradients_sqr,
                              axis=np.arange(1, len(gradients_sqr.shape)))
    
    # weight * ||grad||^2
    # Penalize the gradient norm
    return K.mean(gradient_penalty * weight)


#Upsample, Convolution, AdaIN, Noise, Activation, Convolution, AdaIN, Noise, Activation
def g_block(inp, style, noise, fil, u = True):
    
    b = Dense(fil, kernel_initializer = 'he_normal', bias_initializer = 'ones')(style)
    b = Reshape([1, 1, fil])(b)
    g = Dense(fil, kernel_initializer = 'he_normal', bias_initializer = 'zeros')(style)
    g = Reshape([1, 1, fil])(g)

    n = Conv2D(filters = fil, kernel_size = 1, padding = 'same', kernel_initializer = 'zeros', bias_initializer = 'zeros')(noise)
    
    if u:
        out = UpSampling2D(interpolation = 'bilinear')(inp)
        out = Conv2D(filters = fil, kernel_size = 3, padding = 'same', kernel_initializer = 'he_normal', bias_initializer = 'zeros')(out)
    else:
        out = Conv2D(filters = fil, kernel_size = 3, padding = 'same', kernel_initializer = 'he_normal', bias_initializer = 'zeros')(inp)

    out = add([out, n])
    out = AdaInstanceNormalization()([out, b, g])
    out = LeakyReLU(lrelucoef)(out)
    
    b = Dense(fil, kernel_initializer = 'he_normal', bias_initializer = 'ones')(style)
    b = Reshape([1, 1, fil])(b)
    g = Dense(fil, kernel_initializer = 'he_normal', bias_initializer = 'zeros')(style)
    g = Reshape([1, 1, fil])(g)

    n = Conv2D(filters = fil, kernel_size = 1, padding = 'same', kernel_initializer = 'zeros', bias_initializer = 'zeros')(noise)
    
    out = Conv2D(filters = fil, kernel_size = 3, padding = 'same', kernel_initializer = 'he_normal', bias_initializer = 'zeros')(out)

    out = add([out, n])
    out = AdaInstanceNormalization()([out, b, g])
    out = LeakyReLU(lrelucoef)(out)
    
    return out


#Convolution, Activation, Pooling, Convolution, Activation
def d_block(inp, fil, p = True):
    
    route2 = Conv2D(filters = fil, kernel_size = 3, padding = 'same', kernel_initializer = 'he_normal', bias_initializer = 'zeros')(inp)
    route2 = LeakyReLU(lrelucoef)(route2)
    if p:
        route2 = AveragePooling2D()(route2)
    route2 = Conv2D(filters = fil, kernel_size = 3, padding = 'same', kernel_initializer = 'he_normal', bias_initializer = 'zeros')(route2)
    out = LeakyReLU(lrelucoef)(route2)
    
    return out


def add_prog_gblock(old_model, inp_s, inp_n, inp, fil, i):
    block_end = old_model.layers[-2].output

    noise = Activation('linear')(inp_n)
    noise = Cropping2D(int((im_size - int(im_size / 2 ** (num_layers - i))) / 2))(noise)

    b = Dense(fil, kernel_initializer='he_normal', bias_initializer='ones')(inp_s[-1])
    b = Reshape([1, 1, fil])(b)
    g = Dense(fil, kernel_initializer='he_normal', bias_initializer='zeros')(inp_s[-1])
    g = Reshape([1, 1, fil])(g)

    n = Conv2D(filters=fil, kernel_size=1, padding='same', kernel_initializer='zeros', bias_initializer='zeros')(noise)

    upsampling = UpSampling2D(interpolation='bilinear')(block_end)
    out = Conv2D(filters=fil, kernel_size=3, padding='same', kernel_initializer='he_normal',
                 bias_initializer='zeros')(upsampling)

    out = add([out, n])
    out = AdaInstanceNormalization()([out, b, g])
    out = LeakyReLU(lrelucoef)(out)

    b = Dense(fil, kernel_initializer='he_normal', bias_initializer='ones')(inp_s[-1])
    b = Reshape([1, 1, fil])(b)
    g = Dense(fil, kernel_initializer='he_normal', bias_initializer='zeros')(inp_s[-1])
    g = Reshape([1, 1, fil])(g)

    n = Conv2D(filters=fil, kernel_size=1, padding='same', kernel_initializer='zeros', bias_initializer='zeros')(noise)

    out = Conv2D(filters=fil, kernel_size=3, padding='same', kernel_initializer='he_normal', bias_initializer='zeros')(
        out)
    out = add([out, n])
    out = AdaInstanceNormalization()([out, b, g])
    out = LeakyReLU(lrelucoef)(out)

    out_image = Conv2D(filters=2, kernel_size=1, padding='same', activation='sigmoid', bias_initializer='zeros')(out)

    model1 = Model(inp_s + [inp_n, inp], out_image)
    # get the output layer from old model
    out_old = old_model.layers[-1]
    # connect the upsampling to the old output layer
    out_image2 = out_old(upsampling)
    # define new output image as the weighted sum of the old and new models
    merged = WeightedSum()([out_image2, out_image])
    # define model
    model2 = Model(inp_s + [inp_n, inp], merged)
    return [model1, model2]


def add_prog_dblock(old_model, fil, filp1):
    # get shape of existing model
    in_shape = list(old_model.input.shape)
    # define new input shape as double the size
    input_shape = (in_shape[-2].value * 2, in_shape[-2].value * 2, in_shape[-1].value)
    in_image = Input(shape=input_shape)
    # define new input processing layer
    d = Conv2D(filp1, (1, 1), padding='same', kernel_initializer='he_normal', bias_initializer='zeros')(in_image)
    d = LeakyReLU(lrelucoef)(d)

    d = d_block(d, fil)

    block_new = d

    # skip the input, 1x1 and activation for the old model
    for i in range(3, len(old_model.layers)):
        d = old_model.layers[i](d)
    # define straight-through model
    model1 = Model(in_image, d)

    # downsample the new larger image
    downsample = AveragePooling2D()(in_image)   # TODO: check is this the correct/best way to downsample the image? Yes according to the paper this is correct
    # connect old input processing to downsampled new input
    block_old = old_model.layers[1](downsample)
    block_old = old_model.layers[2](block_old)

    # fade in output of old model input layer with new input
    d = WeightedSum()([block_old, block_new])
    # skip the input, 1x1 and activation for the old model
    for i in range(3, len(old_model.layers)):
        d = old_model.layers[i](d)

    # define straight-through model
    model2 = Model(in_image, d)

    return [model1, model2]


#This object holds the models
class GAN(object):
    
    def __init__(self, steps = 1, lr = 0.0001, decay = 0.00001):
        
        #Models
        self.dis_model_list = None
        self.gen_model_list = None
        self.S = None
        
        self.DM_models = None
        self.DMM_models = None
        self.AM_models = None
        self.MM_models = None
        
        #Config
        #Automatic Decay
        temp = (1 - decay) ** steps
        self.LR = lr * temp
        self.steps = steps
        
        #Init Models
        self.progressive_discriminator()
        self.progressive_generator()
        self.stylist()

    def progressive_discriminator(self):
        if self.dis_model_list:
            return self.dis_model_list

        self.dis_model_list = []

        n_blocks = int(np.log2(im_size) - 2)

        inp = Input(shape=[4, 4, 2])

        filters = [128, 128, 128, 128, 128, 128]

        # conv 1x1
        d = Conv2D(filters[1], (1, 1), padding='same', kernel_initializer='he_normal', bias_initializer='zeros')(inp)
        d = LeakyReLU(lrelucoef)(d)

        d = MinibatchStdev()(d)

        x = d_block(d, filters[0])

        x = Flatten()(x)

        x = Dense(128, kernel_initializer='he_normal', bias_initializer='zeros')(x)
        x = LeakyReLU(lrelucoef)(x)

        x = Dropout(0.2)(x)
        x = Dense(1, kernel_initializer='he_normal', bias_initializer='zeros')(x)

        model = Model(inputs=inp, outputs=x)

        self.dis_model_list.append([model, model])
        # create submodels
        for i in range(1, n_blocks + 1):
            # get prior model without the fade-on
            old_model = self.dis_model_list[i - 1][0]
            # create new model for next resolution
            if i == n_blocks:
                models = add_prog_dblock(old_model, filters[i], filters[i])
            else:
                models = add_prog_dblock(old_model, filters[i], filters[i + 1])
            # store model
            self.dis_model_list.append(models)

        return self.dis_model_list

    def progressive_generator(self):
        if self.gen_model_list:
            return self.gen_model_list
        
        self.gen_model_list = []

        inp_s = []
        inp_s.append(Input(shape=[latent_size]))

        # Get the noise image and crop for each size
        inp_n = Input(shape=[im_size, im_size, 1])
        noi = Activation('linear')(inp_n)
        noi = Cropping2D(int((im_size - int(im_size / 2 ** num_layers)) / 2))(noi)

        # Here do the actual generation stuff
        inp = Input(shape=[1])
        x = Dense(4 * 4 * 512, kernel_initializer='ones', bias_initializer='zeros')(inp)
        x = Reshape([4, 4, 512])(x)
        x = g_block(x, inp_s[0], noi, im_size, u=False)

        x = Conv2D(filters=2, kernel_size=1, padding='same', activation='sigmoid', bias_initializer='zeros')(x)

        model = Model(inputs=inp_s + [inp_n, inp], outputs=x)

        self.gen_model_list.append([model, model])

        nblocks = int(np.log2(im_size) - 2)

        filters = [128, 128, 128, 128, 128]

        for i in range(1, nblocks + 1):
            old_model = self.gen_model_list[i - 1][0]

            inp_s.append(Input(shape=[latent_size]))
            models = add_prog_gblock(old_model, inp_s, inp_n, inp, filters[i - 1], i)

            self.gen_model_list.append(models)

        return self.gen_model_list
    
    def stylist(self):
        
        if self.S:
            return self.S

        inp_s = Input(shape = [latent_size])
        sty = Dense(latent_size, kernel_initializer = 'he_normal', bias_initializer = 'zeros')(inp_s)
        sty = LeakyReLU(lrelucoef)(sty)
        sty = Dense(latent_size, kernel_initializer = 'he_normal', bias_initializer = 'zeros')(sty)
        sty = LeakyReLU(lrelucoef)(sty)
        sty = Dense(latent_size, kernel_initializer = 'he_normal', bias_initializer = 'zeros')(sty)
        sty = LeakyReLU(lrelucoef)(sty)
        sty = Dense(latent_size, kernel_initializer = 'he_normal', bias_initializer = 'zeros')(sty)
        sty = LeakyReLU(lrelucoef)(sty)
        
        self.S = Model(inputs = inp_s, outputs = sty)
        
        return self.S
    
    def AdModels(self):
        #S does update
        self.S.trainable = True
        for layer in self.S.layers:
            layer.trainable = True

        self.AM_models = []

        for i in range(len(self.dis_model_list)):
            self.AM_models.append([])
            for j in range(2):
                #D does not update
                self.dis_model_list[i][j].trainable = False
                for layer in self.dis_model_list[i][j].layers:
                    layer.trainable = False

                #G does update
                self.gen_model_list[i][j].trainable = True
                for layer in self.gen_model_list[i][j].layers:
                    layer.trainable = True

                #This model is simple sequential one with inputs and outputs
                gi = Input(shape = [latent_size])
                gs = self.S(gi)
                gi2 = Input(shape = [im_size, im_size, 1])
                gi3 = Input(shape = [1])

                style_layers = i + 1

                gf = self.gen_model_list[i][j](([gs] * style_layers) + [gi2, gi3])
                df = self.dis_model_list[i][j](gf)

                AM_model = Model(inputs = [gi, gi2, gi3], outputs = df)

                AM_model.compile(optimizer=Adam(self.LR, beta_1 = 0, beta_2 = 0.99, decay = 0.00001), loss='mse')

                self.AM_models[-1].append(AM_model)
        
        return self.AM_models
    
    def AdMixModels(self):
        #S does update
        self.S.trainable = True
        for layer in self.S.layers:
            layer.trainable = True

        self.MM_models = []

        for i in range(len(self.dis_model_list)):
            self.MM_models.append([])
            for j in range(2):
                # D does not update
                self.dis_model_list[i][j].trainable = False
                for layer in self.dis_model_list[i][j].layers:
                    layer.trainable = False

                # G does update
                self.gen_model_list[i][j].trainable = True
                for layer in self.gen_model_list[i][j].layers:
                    layer.trainable = True

                style_layers = i + 1

                #This model is simple sequential one with inputs and outputs
                inp_s = []
                ss = []
                for _ in range(style_layers):
                    inp_s.append(Input([latent_size]))
                    ss.append(self.S(inp_s[-1]))

                gi2 = Input(shape = [im_size, im_size, 1])
                gi3 = Input(shape = [1])

                gf = self.gen_model_list[i][j](ss + [gi2, gi3])
                df = self.dis_model_list[i][j](gf)

                MM_model = Model(inputs = inp_s + [gi2, gi3], outputs = df)

                MM_model.compile(optimizer = Adam(self.LR, beta_1 = 0, beta_2 = 0.99, decay = 0.00001), loss = 'mse')

                self.MM_models[-1].append(MM_model)

        return self.MM_models
    
    def DisModels(self):
        self.DM_models = []

        #S does not update
        self.S.trainable = False
        for layer in self.S.layers:
            layer.trainable = False

        for i in range(len(self.dis_model_list)):
            new_im_size = 2 ** (2 + i)
            self.DM_models.append([])
            for j in range(2):
                self.dis_model_list[i][j].trainable = True
                for layer in self.dis_model_list[i][j].layers:
                    layer.trainable = True

                self.gen_model_list[i][j].trainable = False
                for layer in self.gen_model_list[i][j].layers:
                    layer.trainable = False

                # Real Pipeline
                ri = Input(shape = [new_im_size, new_im_size, 2])
                dr = self.dis_model_list[i][j](ri)

                # Fake Pipeline
                gi = Input(shape = [latent_size])
                gs = self.S(gi)
                gi2 = Input(shape = [im_size, im_size, 1])
                gi3 = Input(shape = [1])

                style_layers = i + 1

                gf = self.gen_model_list[i][j](([gs] * style_layers) + [gi2, gi3])
                df = self.dis_model_list[i][j](gf)

                # Samples for gradient penalty
                # For r1 use real samples (ri)
                # For r2 use fake samples (gf)

                # Model With Inputs and Outputs
                dm_model = Model(inputs=[ri, gi, gi2, gi3], outputs=[dr, df, dr])

                # Create partial of gradient penalty loss
                # For r1, averaged_samples = ri
                # For r2, averaged_samples = gf
                partial_gp_loss = partial(gradient_penalty_loss, averaged_samples = ri, weight = gplossweight)

                #Compile With Corresponding Loss Functions
                dm_model.compile(optimizer=Adam(self.LR, beta_1 = 0, beta_2 = 0.99, decay = 0.00001), loss=['mse', 'mse', partial_gp_loss])
                self.DM_models[-1].append(dm_model)

        return self.DM_models
    
    def DisMixModels(self):
        #S does not update
        self.S.trainable = False
        for layer in self.S.layers:
            layer.trainable = False

        self.DMM_models = []

        for i in range(len(self.dis_model_list)):
            new_im_size = 2 ** (2 + i)
            self.DMM_models.append([])
            for j in range(2):
                # D does not update
                self.dis_model_list[i][j].trainable = True
                for layer in self.dis_model_list[i][j].layers:
                    layer.trainable = True

                # G does update
                self.gen_model_list[i][j].trainable = False
                for layer in self.gen_model_list[i][j].layers:
                    layer.trainable = False

                inp_s = []
                ss = []
                style_layers = i + 1
                for _ in range(style_layers):
                    inp_s.append(Input([latent_size]))
                    ss.append(self.S(inp_s[-1]))

                gi2 = Input(shape = [im_size, im_size, 1])
                gi3 = Input(shape = [1])

                gf = self.gen_model_list[i][j](ss + [gi2, gi3])
                df = self.dis_model_list[i][j](gf)

                ri = Input(shape = [new_im_size, new_im_size, 2])
                dr = self.dis_model_list[i][j](ri)

                DMM_model = Model(inputs = [ri] + inp_s + [gi2, gi3], outputs=[dr, df, dr])

                partial_gp_loss = partial(gradient_penalty_loss, averaged_samples = ri, weight = gplossweight)

                DMM_model.compile(optimizer=Adam(self.LR, beta_1 = 0, beta_2 = 0.99, decay = 0.00001), loss=['mse', 'mse', partial_gp_loss])

                self.DMM_models[-1].append(DMM_model)
        
        return self.DMM_models
    
    def predict(self, inputs, generator):

        for i in range(len(inputs) - 2):
            inputs[i] = self.S.predict(inputs[i])

        return generator.predict(inputs, batch_size = 4)

class WGAN(object):
    
    def __init__(self, steps = 1, lr = 0.0001, decay = 0.00001, silent = True):
        
        self.GAN = GAN(steps = steps, lr = lr, decay = decay)
        self.DisModels = self.GAN.DisModels()
        self.AdModels = self.GAN.AdModels()
        self.MixModels = self.GAN.AdMixModels()
        self.MixModelDs = self.GAN.DisMixModels()

        self.curDisModel = None
        self.curAdModel = None
        self.curAdMixModel = None
        self.curDisMixModel = None
        
        self.lastblip = time.clock()
        
        self.noise_level = 0

        self.im_list, self.im_listup = import_images()
        
        self.silent = silent

        self.ones = np.ones((BATCH_SIZE, 1), dtype=np.float32)
        self.zeros = np.zeros((BATCH_SIZE, 1), dtype=np.float32)
        self.nones = -self.ones
        
        self.enoise = noise(8)
        self.enoiseImage = noiseImage(8)
        
        self.t = [[], []]
    
    def train(self, steps_norm, steps_fadein):
        self.curDisModel = self.DisModels[0][0]
        self.curAdModel = self.AdModels[0][0]
        self.curAdMixModel = self.MixModels[0][0]
        self.curDisMixModel = self.MixModelDs[0][0]
        self.cur_gen = self.GAN.gen_model_list[0][0]
        self.cur_level = 0

        self.train_steps(steps_norm[0])

        for i in range(1, len(self.DisModels)):
            self.curDisModel = self.DisModels[i][1]
            self.curAdModel = self.AdModels[i][1]
            self.curAdMixModel = self.MixModels[i][1]
            self.curDisMixModel = self.MixModelDs[i][1]
            self.cur_gen = self.GAN.gen_model_list[i][1]

            weighted_sums = []
            for model in [self.curDisModel, self.curDisMixModel, self.curAdModel, self.curAdMixModel]:
                for layer in model.layers:
                    if isinstance(layer, Model):
                        for sublayer in layer.layers:
                            if isinstance(sublayer, WeightedSum):
                                # print("Here")
                                weighted_sums.append(sublayer)

            self.cur_level = i

            self.train_steps(steps_fadein[i - 1], fadein=True, weighted_sums=weighted_sums)

            self.curDisModel = self.DisModels[i][0]
            self.curAdModel = self.AdModels[i][0]
            self.curAdMixModel = self.MixModels[i][0]
            self.curDisMixModel = self.MixModelDs[i][0]
            self.cur_gen = self.GAN.gen_model_list[i][0]

            self.train_steps(steps_norm[i])

    def train_steps(self, n_steps, fadein=False, weighted_sums=None):
        for i in range(n_steps):
            if fadein:
                update_fadein(weighted_sums, i, n_steps)
                alpha = i / float(n_steps - 1)
            else:
                alpha = None

            # Train Alternating
            t1 = time.clock()
            if i % 10 <= 5:
                a = self.train_dis(alpha=alpha)
                t2 = time.clock()
                b = self.train_gen()
                t3 = time.clock()
            else:
                a = self.train_mix_d(alpha=alpha)
                t2 = time.clock()
                b = self.train_mix_g()
                t3 = time.clock()

            self.t[0].append(t2 - t1)
            self.t[1].append(t3 - t2)

            # Print info
            if i % 20 == 0 and not self.silent:
                print("\n\nRound " + str(i) + ":")
                print("D: " + str(a))
                print("G: " + str(b))
                s = round((time.clock() - self.lastblip) * 1000) / 1000
                print("T: " + str(s) + " sec")
                self.lastblip = time.clock()

                if self.GAN.steps % 100 == 0:
                    print("TD: " + str(np.sum(self.t[0])))
                    print("TG: " + str(np.sum(self.t[1])))

                    self.t = [[], []]

            # #Save Model
            if self.GAN.steps % 500 == 0:
                self.save(self.GAN.steps)
                self.evaluate(self.GAN.steps)

            self.GAN.steps = self.GAN.steps + 1
          
    def train_dis(self, alpha=None):
        idx = np.random.randint(0, self.im_list[-(self.cur_level + 1)].shape[0], BATCH_SIZE)

        if alpha is None:
            imgs_A = self.im_list[-(self.cur_level + 1)][idx]
            imgs_A = imgs_A.astype(np.float32) / 255
        else:
            imgs_A = self.im_list[-(self.cur_level + 1)][idx]
            imgs_A = imgs_A.astype(np.float32) / 255

            imgs_B = self.im_listup[-(self.cur_level)][idx]
            imgs_B = imgs_B.astype(np.float32) / 255

            imgs_A = np.add(((1.0 - alpha) * imgs_B), (alpha * imgs_A))

        train_data = [imgs_A, noise(BATCH_SIZE), noiseImage(BATCH_SIZE), self.ones]
        
        #Train
        d_loss = self.curDisModel.train_on_batch(train_data, [self.ones, self.nones, self.ones])
        
        return d_loss
    
    def train_mix_d(self, alpha=None):
        
        threshold = np.int32(np.random.uniform(0.0, self.cur_level + 1, size = [BATCH_SIZE]))
        n1 = noise(BATCH_SIZE)
        n2 = noise(BATCH_SIZE)
        
        n = []
        
        for i in range(self.cur_level + 1):
            n.append([])
            for j in range(BATCH_SIZE):
                if i < threshold[j]:
                    n[i].append(n1[j])
                else:
                    n[i].append(n2[j])
            n[i] = np.array(n[i])

        idx = np.random.randint(0, self.im_list[-(self.cur_level + 1)].shape[0], BATCH_SIZE)

        if alpha is None:
            imgs_A = self.im_list[-(self.cur_level + 1)][idx]
            imgs_A = imgs_A.astype(np.float32) / 255
        else:
            imgs_A = self.im_list[-(self.cur_level + 1)][idx]
            imgs_A = imgs_A.astype(np.float32) / 255

            imgs_B = self.im_listup[-(self.cur_level)][idx]
            imgs_B = imgs_B.astype(np.float32) / 255

            imgs_A = np.add(((1.0 - alpha) * imgs_B), (alpha * imgs_A))

        #Train
        d_loss = self.curDisMixModel.train_on_batch([imgs_A] + n + [noiseImage(BATCH_SIZE), self.ones], [self.ones, self.nones, self.ones])
        
        return d_loss
       
    def train_gen(self):
        
        #Train
        g_loss = self.curAdModel.train_on_batch([noise(BATCH_SIZE), noiseImage(BATCH_SIZE), self.ones], self.ones)
        
        return g_loss
    
    def train_mix_g(self):
        threshold = np.int32(np.random.uniform(0.0, self.cur_level + 1, size = [BATCH_SIZE]))
        n1 = noise(BATCH_SIZE)
        n2 = noise(BATCH_SIZE)
        
        n = []
        
        for i in range(self.cur_level + 1):
            n.append([])
            for j in range(BATCH_SIZE):
                if i < threshold[j]:
                    n[i].append(n1[j])
                else:
                    n[i].append(n2[j])
            n[i] = np.array(n[i])
        
        #Train
        g_loss = self.curAdMixModel.train_on_batch(n + [noiseImage(BATCH_SIZE), self.ones], self.ones)
        
        return g_loss
    
    def evaluate(self, num = 0):
        if not os.path.exists(results_path + "/"):
            os.makedirs(results_path + "/")
        n = noise(64)
        n2 = noiseImage(64)

        im = self.GAN.predict(([n] * (self.cur_level + 1)) + [n2, np.ones([64, 1])], self.cur_gen)

        r = []
        r.append(np.concatenate(im[:8,:,:,0], axis = 1))
        r.append(np.concatenate(im[8:16,:,:,0], axis = 1))
        r.append(np.concatenate(im[16:24,:,:,0], axis = 1))
        r.append(np.concatenate(im[24:32,:,:,0], axis = 1))
        r.append(np.concatenate(im[32:40,:,:,0], axis = 1))
        r.append(np.concatenate(im[40:48,:,:,0], axis = 1))
        r.append(np.concatenate(im[48:56,:,:,0], axis = 1))
        r.append(np.concatenate(im[56:64,:,:,0], axis = 1))

        c1 = np.concatenate(r, axis = 0)

        x = Image.fromarray(np.transpose(np.squeeze(np.uint8(c1*255)))).resize((im_size * 8, im_size * 8), resample=0)

        x.save(results_path + "/i"+str(num)+"ii.jpg")

        comb_mask = np.asarray(np.round(im[:, :, :, 1] * 3), dtype='uint8')

        r2 = []
        r2.append(np.concatenate(comb_mask[:8, :, :], axis=1))
        r2.append(np.concatenate(comb_mask[8:16, :, :], axis=1))
        r2.append(np.concatenate(comb_mask[16:24, :, :], axis=1))
        r2.append(np.concatenate(comb_mask[24:32, :, :], axis=1))
        r2.append(np.concatenate(comb_mask[32:40, :, :], axis=1))
        r2.append(np.concatenate(comb_mask[40:48, :, :], axis=1))
        r2.append(np.concatenate(comb_mask[48:56, :, :], axis=1))
        r2.append(np.concatenate(comb_mask[56:64, :, :], axis=1))

        c2 = np.concatenate(r2, axis=0)

        c2 = region_cmap(c2)
        c2 = c2[:, :, :3]

        x2 = Image.fromarray(np.transpose(np.uint8(c2 * 255), axes=(1, 0, 2))).resize((im_size * 8, im_size * 8), resample=0)

        x2.save(results_path + "/i" + str(num) + "ii_mask.jpg")
    
    def saveModel(self, num): #Save a Model
        if not os.path.exists(models_path + "/"):
            os.makedirs(models_path + "/")

        self.cur_gen.save(models_path + "/generator_%s.hdf5" % str(num))
        self.GAN.S.save(models_path + "/stylist_%s.hdf5" % str(num))
    
    def save(self, num): #Save JSON and Weights into /Models/
        self.saveModel(num)

        
if __name__ == "__main__":
    model = WGAN(lr = 0.00005, silent = False)

    model.train([25000, 25000, 25000, 25000, 50000, 100000], [10000, 10000, 10000, 25000, 50000])


