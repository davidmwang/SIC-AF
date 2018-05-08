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
from wgan_gp import resnet_generator, resnet_discriminator, resnet_discriminator_local
from scipy.misc import imsave
from tensorflow.python.client import timeline
from data.PythonAPI.utils import unison_shuffled_copies, get_mask_from_diagonal_coord


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
LAMBDA_REC = 0.80
LAMBDA_ADV = 0.20
IM_SIZE=128
OUTPUT_DIM = IM_SIZE*IM_SIZE*3 # Number of pixels in each iamge
DIRECTORY = "/cs280/home/ubuntu/l1_baseline"

os.mkdir(DIRECTORY)
os.mkdir("{}/models".format(DIRECTORY))
os.mkdir("{}/logs".format(DIRECTORY))
os.mkdir("{}/images".format(DIRECTORY))


# Number of samples to put aside for validation.
# NUM_VAL_SAMPLES = 20
NUM_VAL_SAMPLES = 64

lib.print_model_settings(locals().copy())

DEVICES = ['/gpu:{}'.format(i) for i in range(N_GPUS)]

Generator, Discriminator, Discriminator_local = resnet_generator, resnet_discriminator, resnet_discriminator_local

# Compute number of epochs needed.

num_examples = sum([len(list(glob.glob(image_dir + "/*.jpg"))) for image_dir in IMAGE_DIRS])
num_epochs = int(np.ceil(6*ITERS / num_examples))

def create_image_dataset(image_file_list, num_epochs, batch_size):
    def process_image(x):
        img = tf.image.resize_images(tf.image.decode_jpeg(tf.read_file(x)), size=(IM_SIZE, IM_SIZE))
        img_shape = img.get_shape()
        img = tf.cond(tf.equal(tf.shape(img)[-1], 1), lambda : tf.tile(img, (len(img_shape)-1)*[1] + [3]), lambda : img)
        img = tf.transpose(img, [2, 0, 1])

        return img

    image_dataset = tf.data.Dataset.from_tensor_slices(image_file_list)
    image_dataset = image_dataset.map(process_image) # Decoding function returns NHWC format.
    image_dataset = image_dataset.repeat(num_epochs)
    image_dataset = image_dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))


    return image_dataset



# Create corresponding dataset of masks (https://stackoverflow.com/questions/48889482/feeding-npy-numpy-files-into-tensorflow-data-pipeline).

def create_mask_dataset(mask_file_list, num_epochs, batch_size):

    # Processing function for reading in a NumPy file.
    def read_npy_file(item):
        data = np.load(item.decode()).astype(np.float32)

        data = imresize(data, (IM_SIZE, IM_SIZE))
        data = np.expand_dims(data, axis=0)
        # data = np.repeat(data, 3, axis=0)
        as32 = data.astype(np.float32)
        return as32/float(np.max(as32))

    mask_dataset = tf.data.Dataset.from_tensor_slices(mask_files)
    mask_dataset = mask_dataset.map(lambda item: tf.py_func(read_npy_file, [item], tf.float32))
    mask_dataset = mask_dataset.repeat(num_epochs)
    mask_dataset = mask_dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
    return mask_dataset


def create_local_patch_coordinate_dataset(mask_file_list, num_epochs, batch_size):
    # Processing function for reading in a NumPy file.
    def read_npy_file(item):
        data = np.load(item.decode()).astype(np.float32)

        data = imresize(data, (IM_SIZE, IM_SIZE))
        data = data.astype(np.float32)
        data = data/float(np.max(data))

        indices = np.where(data == 1.0)
        center_row = 0.5 * (np.max(indices[0]) + np.min(indices[0]))
        center_col = 0.5 * (np.max(indices[1]) + np.min(indices[1]))

        center_row = int(min(center_row, IM_SIZE-0.25*IM_SIZE))
        center_row = int(max(center_row, 0.25 * IM_SIZE))

        center_col = int(min(center_col, IM_SIZE-0.25*IM_SIZE))
        center_col = int(max(center_col, 0.25 * IM_SIZE))

        top_left = np.array([center_row - IM_SIZE//4, center_col - IM_SIZE//4])
        # data = np.expand_dims(top_left, axis=0)
        data = top_left.astype(np.int32)

        return data


    mask_dataset = tf.data.Dataset.from_tensor_slices(mask_files)
    mask_dataset = mask_dataset.map(lambda item: tf.py_func(read_npy_file, [item], tf.int32))
    mask_dataset = mask_dataset.repeat(num_epochs)
    mask_dataset = mask_dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
    return mask_dataset

mask_files = np.array(sorted(list(itertools.chain.from_iterable([glob.glob(mask_dir + "/*.npy") for mask_dir in MASK_DIRS]))))
image_files = np.array(sorted(list(itertools.chain.from_iterable([glob.glob(image_dir + "/*.jpg") for image_dir in IMAGE_DIRS]))))
mask_files, image_files = unison_shuffled_copies(mask_files, image_files)

image_val_files, image_files = image_files[:NUM_VAL_SAMPLES], image_files[NUM_VAL_SAMPLES:]
image_dataset = create_image_dataset(image_files, num_epochs, BATCH_SIZE)
image_iterator = image_dataset.make_one_shot_iterator()
image_val_dataset = create_image_dataset(image_val_files, 1, NUM_VAL_SAMPLES)
image_val_iterator = image_val_dataset.make_one_shot_iterator()

mask_val_files, mask_files = mask_files[:NUM_VAL_SAMPLES], mask_files[NUM_VAL_SAMPLES:]
print("mask val files: ", mask_val_files[:5])
print("image val files: ", image_val_files[:5])



mask_dataset = create_mask_dataset(mask_files, num_epochs, BATCH_SIZE)
mask_iterator = mask_dataset.make_one_shot_iterator()
mask_val_dataset = create_mask_dataset(mask_val_files, 1, NUM_VAL_SAMPLES)
mask_val_iterator = mask_val_dataset.make_one_shot_iterator()

# local_patch_dataset = create_local_patch_coordinate_dataset(mask_files, num_epochs, BATCH_SIZE)
# local_patch_iterator = local_patch_dataset.make_one_shot_iterator()
# local_patch_val_dataset = create_local_patch_coordinate_dataset(mask_val_files, 1, NUM_VAL_SAMPLES)
# local_patch_val_iterator = local_patch_val_dataset.make_one_shot_iterator()


def apply_batch_crop(img_batch, coord_batch):
    cropped = []

    for i in range(img_batch.get_shape()[0]):
        img = img_batch[i]
        # img = tf.gather(img_batch, [i])
        coord = coord_batch[i]

        # coord = tf.squeeze(tf.gather(coord_batch, [i]))
        # print("img batch: ", img_batch.get_shape())
        # print("coord batch: ", coord_batch.get_shape())
        # print("img: ", img.get_shape())
        # print("coord: ", coord.get_shape())
        #

        # print(1/0)
        # patch = tf.slice(img, [0, 0, coord[0], coord[1]],
        #                       [1, 3, IM_SIZE//2, IM_SIZE//2])
        patch = img[:, coord[0]:coord[0]+IM_SIZE//2, coord[1]:coord[1]+IM_SIZE//2]
        cropped.append(patch)

    asdf =  tf.stack(cropped)

    print("apply batch crop output shape", asdf.get_shape())
    return asdf

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as session:

    summary_writer = tf.summary.FileWriter("{}/logs".format(DIRECTORY), session.graph, flush_secs=10)


    # Load in validation set for evaluation.
    image_val_batch = session.run(image_val_iterator.get_next())    # Fixed image batch to use for validation.
    mask_val_batch = session.run(mask_val_iterator.get_next())
    local_patch_val_batch = session.run(local_patch_val_iterator.get_next())

    # all_real_data_conv = tf.placeholder(tf.int32, shape=[BATCH_SIZE, 3, 64, 64])
    # # binary mask placeholder
    # all_real_data_mask = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 3, 64, 64])

    all_real_data_conv = image_iterator.get_next()
    all_real_data_mask = mask_iterator.get_next()
    all_real_data_local_patch = tf.squeeze(local_patch_iterator.get_next())
    all_real_data_mask.set_shape([BATCH_SIZE, 1, IM_SIZE, IM_SIZE])
    all_real_data_local_patch.set_shape([BATCH_SIZE, 2])



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

            tiled_all_real_data_mask = tf.tile(all_real_data_mask, [1, 3, 1, 1])
            real_data_masked = tf.multiply(real_data_conv, 1 - tiled_all_real_data_mask)
            real_data_masked_and_scaled = 2*((tf.cast(real_data_masked, tf.float32)/255.)-.5)
            real_data = 2*((tf.cast(real_data_conv, tf.float32)/255.)-.5)


            real_data_masked_and_scaled_and_concat = tf.concat([real_data_masked_and_scaled, all_real_data_mask], axis=1)

            fake_data = Generator(real_data_masked_and_scaled_and_concat)
            # print(real_data.get_shape())

            # real_data.set_shape([64, 3, 64, 64])
            # fake_data = Generator(real_data)

            blended_fake_data = tf.multiply(fake_data, tiled_all_real_data_mask) + tf.multiply(real_data, 1-tiled_all_real_data_mask)
            # blended_fake_data = fake_data
            # real_data_local = apply_batch_crop(real_data, all_real_data_local_patch)
            # blended_fake_data_local = apply_batch_crop(blended_fake_data, all_real_data_local_patch)

            # disc_real = Discriminator(tf.concat(concat_dim=1,values=[real_data, all_real_data_mask]))

            # disc_real_local = Discriminator_local(real_data_local)
            # disc_fake = Discriminator(tf.concat(concat_dim=1,values=[blended_fake_data, all_real_data_mask]))
            # disc_fake_local = Discriminator_local(blended_fake_data_local)

            rec_cost = tf.reduce_mean(tf.reduce_sum(tf.abs(blended_fake_data - real_data), axis=[1,2,3]))
            # rec_cost = tf.reduce_mean(tf.norm(blended_fake_data - real_data, axis=0, ord=1))

            if MODE == 'wgan':
                gen_cost = -tf.reduce_mean(disc_fake)
                disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

            elif MODE == 'wgan-gp':
                # gen_cost = -tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_fake_local)

                disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)
                # disc_cost_local = tf.reduce_mean(disc_fake_local) - tf.reduce_mean(disc_real_local)

                # disc_cost = disc_cost_whole + disc_cost_local


                # gen_cost = LAMBDA_ADV * gen_cost + LAMBDA_REC * rec_cost
                gen_cost = LAMBDA_REC * rec_cost

                alpha = tf.random_uniform(
                    shape=[int(BATCH_SIZE/len(DEVICES)),1],
                    minval=0.,
                    maxval=1.
                )
                alpha = tf.expand_dims(alpha, axis=-1)
                alpha = tf.expand_dims(alpha, axis=-1)

                differences = blended_fake_data - real_data

                interpolates = real_data + (alpha*differences)
                gradients = tf.gradients(Discriminator(interpolates), [interpolates])[0]
                slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
                gradient_penalty = tf.reduce_mean((slopes-1.)**2)
                disc_cost += LAMBDA*gradient_penalty
                # locale
                # differences_local = blended_fake_data_local - real_data_local
                # interpolates_local = real_data_local + (alpha*differences_local)
                # gradients_local = tf.gradients(Discriminator(interpolates_local), [interpolates_local])[0]
                # slopes_local = tf.sqrt(tf.reduce_sum(tf.square(gradients_local), reduction_indices=[1]))
                # gradient_penalty_local = tf.reduce_mean((slopes_local-1.)**2)
                # disc_cost += LAMBDA*gradient_penalty_local

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
        image_val_batch_masked = image_val_batch * (1.0 - (mask_val_batch).repeat(3, axis=1))

        image_val_batch_normalized = 2.0 * ((image_val_batch_masked/255.0) - 0.5)
        all_fixed_noise_samples.append(Generator(tf.constant(np.concatenate((image_val_batch_normalized, mask_val_batch), axis=1))))
        # all_fixed_noise_samples.append(Generator(tf.constant(image_val_batch)))


    if tf.__version__.startswith('1.'):
        all_fixed_noise_samples = tf.concat(all_fixed_noise_samples, axis=0)
    else:
        all_fixed_noise_samples = tf.concat(0, all_fixed_noise_samples)

    def generate_image(iteration):
        samples = session.run(all_fixed_noise_samples)



        samples = ((samples+1.)*(255./2)).astype('int32')
        lib.save_images.save_images(samples, '{}/images/samples_gen_{}.png'.format(DIRECTORY, iteration))
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


        mask_only = samples * (mask_val_batch).repeat(3, axis=1)
        lib.save_images.save_images(mask_only, '{}/images/samples_mask_only_{}.png'.format(DIRECTORY, iteration))


        # samples = (1.0 - (mask_val_batch/255.).repeat(3, axis=1)) * image_val_batch
        # print(samples)
        lib.save_images.save_images(samples, '{}/images/samples_{}.png'.format(DIRECTORY, iteration))

    # # Dataset iterator
    # train_gen, dev_gen = lib.small_imagenet.load(BATCH_SIZE, data_dir=DATA_DIR)

    # def inf_train_gen():
    #     while True:
    #         for (images,) in train_gen():
    #             yield images

    # Save a batch of ground-truth samples

    # print("image_val_batch shape: ", tf.constant(image_val_batch).get_shape())
    # print("image val patch batch: ", tf.constant(local_patch_val_batch).get_shape())

    # print(local_patch_val_batch)

    # lib.save_images.save_images(session.run(apply_batch_crop(tf.constant(image_val_batch), tf.constant(local_patch_val_batch))), '{}/images/samples_local_patches.png'.format(DIRECTORY))

    lib.save_images.save_images(image_val_batch, '{}/images/samples_groundtruth.png'.format(DIRECTORY))

    # Train loop
    session.run(tf.initialize_all_variables())

    saver = tf.train.Saver()
    # saver.restore(session, "/cs280/home/ubuntu/l1_concat_downsample_128/models/model.ckpt")
    # print("Model restored. ")
    # generate_image("999999999")
    # print(1/0)

    # gen = inf_train_gen()
    for iteration in range(ITERS):
        print("==============iteration: ", iteration)

        if iteration % (828) == 0:
        # if iteration % (1656) == 0:
            save_path = saver.save(session, "{}/models/model.ckpt".format(DIRECTORY))
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
        # if (MODE == 'dcgan') or (MODE == 'lsgan'):
        #     disc_iters = 1
        # else:
        #     disc_iters = CRITIC_ITERS

        # # ================== UNCOMMENT LATER ================
        # for i in range(disc_iters):
        #     print("in disc_iter", i)

        #     _disc_cost, _, _disc_cost_whole, _disc_cost_local = session.run([disc_cost, disc_train_op, disc_cost_whole, disc_cost_local])
        #     print("disc loss:", _disc_cost)


        #     if MODE == 'wgan':
        #         _ = session.run([clip_disc_weights])
        # # ===================================================

        if iteration > 0:
            summary = tf.Summary()
            summary.value.add(tag='reconstruction cost', simple_value=_gen_cost)
            # summary.value.add(tag='discriminator cost', simple_value=_disc_cost)
            # summary.value.add(tag='discriminator cost (whole image)', simple_value=_disc_cost_whole)
            # summary.value.add(tag='discriminator cost (local crop)', simple_value=_disc_cost_local)
            summary_writer.add_summary(summary, iteration)

        if iteration % 200 == 0:
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
