import os, sys, glob
sys.path.append(os.getcwd())

import time
import functools
import itertools

import numpy as np
import tensorflow as tf
import sklearn.datasets

import tflib as lib
import tflib.ops.linear
import tflib.ops.conv2d
import tflib.ops.batchnorm
import tflib.ops.deconv2d
import tflib.save_images
import tflib.small_imagenet
import tflib.ops.layernorm
import tflib.plot
from scipy.misc import imresize
from wgan_gp import resnet_generator, resnet_discriminator
from scipy.misc import imsave
from tensorflow.python.client import timeline


# DATA_DIR = ''

# Directory containing original MSCOCO images.
IMAGE_DIRS = ["/cs280/home/ubuntu/person", "/cs280/home/ubuntu/no_people"]

# Directory containing masks for associated MSCOCO images to use for training
MASK_DIRS = ["/cs280/home/ubuntu/person_mask", "/cs280/home/ubuntu/no_people_mask"]


if len(IMAGE_DIRS) == 0 or len(MASK_DIRS) == 0:
    raise Exception('Please specify paths to directories containing images and/or masks.')

MODE = 'wgan-gp' # dcgan, wgan, wgan-gp, lsgan
DIM = 64 # Model dimensionality
CRITIC_ITERS = 5 # How many iterations to train the critic for
N_GPUS = 1 # Number of GPUs
BATCH_SIZE = 64 # Batch size. Must be a multiple of N_GPUS
ITERS = 20000000 # How many iterations to train for
LAMBDA = 10 # Gradient penalty lambda hyperparameter
LAMBDA_REC = 0.95
LAMBDA_ADV = 0
OUTPUT_DIM = 64*64*3 # Number of pixels in each iamge

# Number of samples to put aside for validation.
# NUM_VAL_SAMPLES = 20
NUM_VAL_SAMPLES = 64

lib.print_model_settings(locals().copy())

DEVICES = ['/gpu:{}'.format(i) for i in range(N_GPUS)]

Generator, Discriminator = resnet_generator, resnet_discriminator

# Compute number of epochs needed.

num_examples = sum([len(list(glob.glob(image_dir + "/*.jpg"))) for image_dir in IMAGE_DIRS])
num_epochs = int(np.ceil(6*ITERS / num_examples))

def create_image_dataset(image_file_list, num_epochs, batch_size):
    def process_image(x):
        img = tf.image.resize_images(tf.image.decode_jpeg(tf.read_file(x)), size=(64,64))
        img_shape = img.get_shape()
        img = tf.cond(tf.equal(tf.shape(img)[-1], 1), lambda : tf.tile(img, (len(img_shape)-1)*[1] + [3]), lambda : img)
        img = tf.transpose(img, [2, 0, 1])

        return img

    image_dataset = tf.data.Dataset.from_tensor_slices(image_file_list)
    image_dataset = image_dataset.map(process_image) # Decoding function returns NHWC format.
    image_dataset = image_dataset.repeat(num_epochs)
    image_dataset = image_dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))


    return image_dataset

# Create image datasets (training and val).
image_files = list(itertools.chain.from_iterable([glob.glob(image_dir + "/*.jpg") for image_dir in IMAGE_DIRS]))
image_val_files, image_files = image_files[:NUM_VAL_SAMPLES], image_files[NUM_VAL_SAMPLES:]

image_dataset = create_image_dataset(image_files, num_epochs, BATCH_SIZE)
image_iterator = image_dataset.make_one_shot_iterator()

image_val_dataset = create_image_dataset(image_val_files, 1, NUM_VAL_SAMPLES)
image_val_iterator = image_val_dataset.make_one_shot_iterator()

# Create corresponding dataset of masks (https://stackoverflow.com/questions/48889482/feeding-npy-numpy-files-into-tensorflow-data-pipeline).

def create_mask_dataset(mask_file_list, num_epochs, batch_size):

    # Processing function for reading in a NumPy file.
    def read_npy_file(item):
        data = np.load(item.decode()).astype(np.float32)

        data = imresize(data, (64, 64))
        data = np.expand_dims(data, axis=0)
        # data = np.repeat(data, 3, axis=0)
        as32 = data.astype(np.float32)
        return as32/float(np.max(as32))

    mask_dataset = tf.data.Dataset.from_tensor_slices(mask_files)
    mask_dataset = mask_dataset.map(lambda item: tf.py_func(read_npy_file, [item], tf.float32))
    mask_dataset = mask_dataset.repeat(num_epochs)
    mask_dataset = mask_dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
    return mask_dataset

mask_files = list(itertools.chain.from_iterable([glob.glob(mask_dir + "/*.npy") for mask_dir in MASK_DIRS]))
mask_val_files, mask_files = mask_files[:NUM_VAL_SAMPLES], mask_files[NUM_VAL_SAMPLES:]

mask_dataset = create_mask_dataset(mask_files, num_epochs, BATCH_SIZE)

mask_iterator = mask_dataset.make_one_shot_iterator()

mask_val_dataset = create_mask_dataset(mask_val_files, 1, NUM_VAL_SAMPLES)
mask_val_iterator = mask_val_dataset.make_one_shot_iterator()


with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as session:

    # Load in validation set for evaluation.
    image_val_batch = session.run(image_val_iterator.get_next())    # Fixed image batch to use for validation.
    mask_val_batch = session.run(mask_val_iterator.get_next())

    # all_real_data_conv = tf.placeholder(tf.int32, shape=[BATCH_SIZE, 3, 64, 64])
    # # binary mask placeholder
    # all_real_data_mask = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 3, 64, 64])

    all_real_data_conv = image_iterator.get_next()
    all_real_data_mask = mask_iterator.get_next()
    all_real_data_mask.set_shape([BATCH_SIZE, 1, 64, 64])

    if tf.__version__.startswith('1.'):
        split_real_data_conv = tf.split(all_real_data_conv, len(DEVICES))
        split_real_data_mask = tf.split(all_real_data_mask, len(DEVICES))
    else:
        split_real_data_conv = tf.split(0, len(DEVICES), all_real_data_conv)
        split_real_data_mask = tf.split(0, len(DEVICES), all_real_data_mask)

    gen_costs, disc_costs = [],[]

    for device_index, (device, real_data_conv) in enumerate(zip(DEVICES, split_real_data_conv)):
        with tf.device(device):

            # real_data = tf.reshape(2*((tf.cast(real_data_conv, tf.float32)/255.)-.5), [int(BATCH_SIZE/len(DEVICES)), OUTPUT_DIM])
            real_data = 2*((tf.cast(real_data_conv, tf.float32)/255.)-.5)

            tiled_all_real_data_mask = tf.tile(all_real_data_mask, [1, 3, 1, 1])

            fake_data = Generator(tf.multiply(real_data, 1 - tiled_all_real_data_mask))
            # print(real_data.get_shape())

            # real_data.set_shape([64, 3, 64, 64])
            # fake_data = Generator(real_data)

            blended_fake_data = tf.multiply(fake_data, tiled_all_real_data_mask) + tf.multiply(real_data, 1-tiled_all_real_data_mask)
            # blended_fake_data = fake_data

            disc_real = Discriminator(real_data)
            disc_fake = Discriminator(blended_fake_data)

            rec_cost = tf.reduce_mean(tf.reduce_sum(tf.abs(blended_fake_data - real_data), axis=[1,2,3]))
            # rec_cost = tf.reduce_mean(tf.norm(blended_fake_data - real_data, axis=0, ord=1))

            if MODE == 'wgan':
                gen_cost = -tf.reduce_mean(disc_fake)
                disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

            elif MODE == 'wgan-gp':
                # gen_cost = -tf.reduce_mean(disc_fake)
                disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

                # gen_cost = LAMBDA_ADV * gen_cost + LAMBDA_REC * rec_cost
                gen_cost = LAMBDA_REC * rec_cost

                alpha = tf.random_uniform(
                    shape=[int(BATCH_SIZE/len(DEVICES)),1],
                    minval=0.,
                    maxval=1.
                )
                differences = blended_fake_data - real_data
                interpolates = real_data + (alpha*differences)
                gradients = tf.gradients(Discriminator(interpolates), [interpolates])[0]
                slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
                gradient_penalty = tf.reduce_mean((slopes-1.)**2)
                disc_cost += LAMBDA*gradient_penalty

            elif MODE == 'dcgan':
                try: # tf pre-1.0 (bottom) vs 1.0 (top)
                    gen_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake,
                                                                                      labels=tf.ones_like(disc_fake)))
                    disc_cost =  tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake,
                                                                                        labels=tf.zeros_like(disc_fake)))
                    disc_cost += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_real,
                                                                                        labels=tf.ones_like(disc_real)))
                except Exception as e:
                    gen_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(disc_fake, tf.ones_like(disc_fake)))
                    disc_cost =  tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(disc_fake, tf.zeros_like(disc_fake)))
                    disc_cost += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(disc_real, tf.ones_like(disc_real)))
                disc_cost /= 2.

            elif MODE == 'lsgan':
                gen_cost = tf.reduce_mean((disc_fake - 1)**2)
                disc_cost = (tf.reduce_mean((disc_real - 1)**2) + tf.reduce_mean((disc_fake - 0)**2))/2.

            else:
                raise Exception()

            gen_costs.append(gen_cost)
            disc_costs.append(disc_cost)

    gen_cost = tf.add_n(gen_costs) / len(DEVICES)
    disc_cost = tf.add_n(disc_costs) / len(DEVICES)

    if MODE == 'wgan':
        gen_train_op = tf.train.RMSPropOptimizer(learning_rate=5e-5).minimize(gen_cost,
                                             var_list=lib.params_with_name('Generator'), colocate_gradients_with_ops=True)
        disc_train_op = tf.train.RMSPropOptimizer(learning_rate=5e-5).minimize(disc_cost,
                                             var_list=lib.params_with_name('Discriminator.'), colocate_gradients_with_ops=True)

        clip_ops = []
        for var in lib.params_with_name('Discriminator'):
            clip_bounds = [-.01, .01]
            clip_ops.append(tf.assign(var, tf.clip_by_value(var, clip_bounds[0], clip_bounds[1])))
        clip_disc_weights = tf.group(*clip_ops)

    elif MODE == 'wgan-gp':
        gen_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0., beta2=0.9).minimize(gen_cost,
                                          var_list=lib.params_with_name('Generator'), colocate_gradients_with_ops=True)
        disc_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0., beta2=0.9).minimize(disc_cost,
                                           var_list=lib.params_with_name('Discriminator.'), colocate_gradients_with_ops=True)

    elif MODE == 'dcgan':
        gen_train_op = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5).minimize(gen_cost,
                                          var_list=lib.params_with_name('Generator'), colocate_gradients_with_ops=True)
        disc_train_op = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5).minimize(disc_cost,
                                           var_list=lib.params_with_name('Discriminator.'), colocate_gradients_with_ops=True)

    elif MODE == 'lsgan':
        gen_train_op = tf.train.RMSPropOptimizer(learning_rate=1e-4).minimize(gen_cost,
                                             var_list=lib.params_with_name('Generator'), colocate_gradients_with_ops=True)
        disc_train_op = tf.train.RMSPropOptimizer(learning_rate=1e-4).minimize(disc_cost,
                                              var_list=lib.params_with_name('Discriminator.'), colocate_gradients_with_ops=True)

    else:
        raise Exception()

    # For generating samples
    # fixed_noise = tf.constant(np.random.normal(size=(BATCH_SIZE, 128)).astype('float32'))
    all_fixed_noise_samples = []
    for device_index, device in enumerate(DEVICES):
        n_samples = BATCH_SIZE / len(DEVICES)
        image_val_batch = 2.0 * ((image_val_batch/255.0) - 0.5)
        all_fixed_noise_samples.append(Generator(tf.constant((1.0 - (mask_val_batch).repeat(3, axis=1)) * image_val_batch)))
        # all_fixed_noise_samples.append(Generator(tf.constant(image_val_batch)))


    if tf.__version__.startswith('1.'):
        all_fixed_noise_samples = tf.concat(all_fixed_noise_samples, axis=0)
    else:
        all_fixed_noise_samples = tf.concat(0, all_fixed_noise_samples)

    def generate_image(iteration):
        samples = session.run(all_fixed_noise_samples)



        samples = ((samples+1.)*(255./2)).astype('int32')
        lib.save_images.save_images(samples, 'samples_gen_{}.png'.format(iteration))
        # samples = mask_val_batch.repeat(3, axis=1)
        # print(mask_val_batch[0])
        # print(1-mask_val_batch[0])
        # print("=======")
        # print(np.max(mask_val_batch))
        # print(np.min(mask_val_batch))
        # print(samples.shape)
        # print(mask_val_batch.shape)
        # print(image_val_batch.shape)
        samples = samples * (mask_val_batch).repeat(3, axis=1) + (1.0 - (mask_val_batch).repeat(3, axis=1)) * image_val_batch

        print("sample min:", np.min(samples))
        print("sample max:", np.max(samples))

        samples = (1.0 - (mask_val_batch/255.).repeat(3, axis=1)) * image_val_batch
        # print(samples)
        lib.save_images.save_images(samples, 'samples_{}.png'.format(iteration))

    # # Dataset iterator
    # train_gen, dev_gen = lib.small_imagenet.load(BATCH_SIZE, data_dir=DATA_DIR)

    # def inf_train_gen():
    #     while True:
    #         for (images,) in train_gen():
    #             yield images

    # Save a batch of ground-truth samples


    lib.save_images.save_images(image_val_batch, 'samples_groundtruth.png')

    # Train loop
    session.run(tf.initialize_all_variables())

    saver = tf.train.Saver()
    # saver.restore(session, "models/l1_patch_only_model/model.ckpt")
    # generate_image("999999999")
    # print(1/0)


    # gen = inf_train_gen()
    for iteration in range(ITERS):
        print("iteration: ", iteration)
        if iteration % (1656) == 0:
            save_path = saver.save(session, "models/model.ckpt")
            print("Model saved in path: %s" % save_path)

        start_time = time.time()

        # Train generator
        if iteration > 0:
            # image_batch = session.run(all_real)
            # mask_batch = session.run(mask_iterator.get_next()[0])


            # TODO: Generator needs to take in masked images.
            # _ = session.run(gen_train_op, feed_dict={all_real_data_conv: image_batch,
            #                                          all_real_data_mask: mask_batch})

            # options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            # run_metadata = tf.RunMetadata()
            _gen_cost, _ = session.run([gen_cost, gen_train_op])
            print("gen loss:", _gen_cost)

            # _ = session.run(gen_train_op, options=options, run_metadata=run_metadata)
            # fetched_timeline = timeline.Timeline(run_metadata.step_stats)
            # chrome_trace = fetched_timeline.generate_chrome_trace_format()
            # with open('timeline_gen.json', 'w+') as f:
            #     f.write(chrome_trace)

        # Train critic
        if (MODE == 'dcgan') or (MODE == 'lsgan'):
            disc_iters = 1
        else:
            disc_iters = CRITIC_ITERS
        for i in range(disc_iters):
            print("in disc_iter", i)
            # _data = gen.next()

            # tmp = image_iterator.get_next()
            # tmp2 = mask_iterator.get_next()

            # image_batch = session.run(tf.transpose(image_iterator.get_next(), [0, 3, 1, 2]))


            # print(type(image_batch))
            # print(image_batch.shape)
            # print(1/0)

            # mask_batch = session.run(mask_iterator.get_next()[0])
            # print(type(mask_batch))
            # print(mask_batch.shape)
            # masked_images = image_batch * mask_batch

            # TODO: Need to run masked images through the generator and feed both the real images and reconstructed images to discriminator.
            # _disc_cost, _ = session.run([disc_cost, disc_train_op],
            #                             feed_dict={all_real_data_conv: image_batch,
            #                                        all_real_data_mask: mask_batch})

            # options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            # run_metadata = tf.RunMetadata()
            # _disc_cost, _ = session.run([disc_cost, disc_train_op])
            # print("disc loss:", _disc_cost)
            # _disc_cost, _ = session.run([disc_cost, disc_train_op], options=options, run_metadata=run_metadata)
            # fetched_timeline = timeline.Timeline(run_metadata.step_stats)
            # chrome_trace = fetched_timeline.generate_chrome_trace_format()
            # with open('timeline_discr.json', 'w+') as f:
            #     f.write(chrome_trace)

            if MODE == 'wgan':
                _ = session.run([clip_disc_weights])

        lib.plot.plot('train disc cost', _disc_cost)
        lib.plot.plot('time', time.time() - start_time)

        if iteration % 100 == 0:
            # t = time.time()
            # dev_disc_costs = []
            # for (images,) in dev_gen():
            #     # _dev_disc_cost = session.run(disc_cost, feed_dict={all_real_data_conv: images})
            #     _dev_disc_cost = session.run(disc_cost)
            #     dev_disc_costs.append(_dev_disc_cost)
            # lib.plot.plot('dev disc cost', np.mean(dev_disc_costs))

            generate_image(iteration)


        # if (iteration < 5) or (iteration % 200 == 199):
        #     lib.plot.flush()

        # lib.plot.tick()
