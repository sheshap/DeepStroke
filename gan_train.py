from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np
from KDD import KDD
import gan_model as gan
import sys
import time

sys.path.append('../')
import image_utils as iu

results = {
    'output1a': './gen_img1a/',
    'output2a': './gen_img2a/',
    'output1b': './gen_img1b/',
    'output2b': './gen_img2b/',
    'checkpoint': './model/checkpoint',
    'model': './model/GAN-model.ckpt'
}

train_step = {
    'global_step': 10001,
    'logging_interval': 2000,
}

def main():
    start_time = time.time()  # Clocking start
    kdd=KDD()
    kdd.setData('Desktop','Phone', 'Tablet', 'Hold')

    # GPU configure
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as s:
        # GAN models
        model = gan.GAN(s)
        # Initializing
        s.run(tf.global_variables_initializer())

        sample_x, _ = kdd.getBatch(model.sample_num)
        sample_z1 = np.random.uniform(-1., 1., [model.sample_num, model.z_dim]).astype(np.float32)
        sample_z2 = np.random.uniform(-1., 1., [model.sample_num, model.z_dim]).astype(np.float32)

        d_overpowered = False
        for step in range(train_step['global_step']):
            batch_x, _ = kdd.getBatch(model.batch_size)#mnist.train.next_batch(model1.batch_size)

            batch_z1 = np.random.uniform(-1., 1., size=[model.batch_size, model.z_dim]).astype(np.float32)
            batch_z2 = np.random.uniform(-1., 1., size=[model.batch_size, model.z_dim]).astype(np.float32)
            # Update D1 and D2 networks
            if not d_overpowered:
                _, d1_loss, _, d2_loss, _, d3_loss1, _, d3_loss2 = s.run([model.d1_op, model.d1_loss, model.d2_op, model.d2_loss, model.d3_op1, model.d3_loss1, model.d3_op2, model.d3_loss2],
                                  feed_dict={
                                      model.x1: [batch_x[0,:900]],
                                      model.z1: batch_z1,
                                      model.x2: [batch_x[0, 900:1800]],
                                      model.z2: batch_z2,
                                      model.x3: [batch_x[0,1800:2700]],
                                  })

            # Update G1 and G2 networks
            _, g1_loss, _, g2_loss = s.run([model.g1_op, model.g1_loss, model.g2_op, model.g2_loss],
                              feed_dict={
                                  model.x1: [batch_x[0,:900]],
                                  model.z1: batch_z1,
                                  model.x2: [batch_x[0, 900:1800]],
                                  model.z2: batch_z2,
                                  model.x3: [batch_x[0, 1800:2700]],
                              })

            d_overpowered = ((d1_loss+d3_loss1) < (g1_loss / 2)) or ((d2_loss+d3_loss2) < (g2_loss / 2))

            if step % train_step['logging_interval'] == 0 and step != 0:
                batch_x, _ = kdd.getBatch(model.batch_size)#mnist.test.next_batch(model1.batch_size)
                batch_z1 = np.random.uniform(-1., 1., [model.batch_size, model.z_dim]).astype(np.float32)
                batch_z2 = np.random.uniform(-1., 1., [model.batch_size, model.z_dim]).astype(np.float32)
                d1_loss, g1_loss, summary1, d2_loss, g2_loss, summary2, d3_loss1, d3_loss2, summary3 = s.run([model.d1_loss, model.g1_loss, model.merged, model.d2_loss, model.g2_loss, model.merged, model.d3_loss1, model.d3_loss2, model.merged],
                                                feed_dict={
                                                    model.x1: [batch_x[0,:900]],
                                                    model.z1: batch_z1,
                                                    model.x2: [batch_x[0, 900:1800]],
                                                    model.z2: batch_z2,
                                                    model.x3: [batch_x[0, 1800:2700]],
                                                })

                d_overpowered = ((d1_loss + d3_loss1) < (g1_loss / 2)) or ((d2_loss + d3_loss2) < (g2_loss / 2))

                # Print loss
                print("Step %05d =>" % step,
                      "D1 loss: {:.2f}".format(d1_loss),
                      "G1 loss: {:.2f}".format(g1_loss),
                      "D2 loss: {:.2f}".format(d2_loss),
                      "G2 loss: {:.2f}".format(g2_loss),
                      "D3 loss (G1): {:.2f}".format((d3_loss1)),
                      "D3 loss (G2): {:.2f}".format((d3_loss2)))

                # Training G1 and G2 models with sample image and noise
                samples1, samples2 = s.run([model.g1, model.g2],
                                feed_dict={
                                    model.x1: [sample_x[0,:900]],
                                    model.z1: sample_z1,
                                    model.x2: [sample_x[0,900:1800]],
                                    model.z2: sample_z2,
                                })

                samples1a1 = np.reshape(samples1[0,:900], [-1, model.output_height, model.output_width, model.channel])
                samples2a1 = np.reshape(samples2[0,:900], [-1, model.output_height, model.output_width, model.channel])
                samples1b1 = np.reshape(samples1[0,900:1800], [-1, model.output_height, model.output_width, model.channel])
                samples2b1 = np.reshape(samples2[0,900:1800], [-1, model.output_height, model.output_width, model.channel])

                # Summary saver
                model.writer.add_summary(summary1, step)
                model.writer.add_summary(summary2, step)
                # Export image generated by model1 G
                sample_image_height = model.sample_size
                sample_image_width = model.sample_size
                #print(model1.output_height,model1.output_width)
                sample_dir1a1 = results['output1a'] + 'train1a1_{:08d}.png'.format(step)
                sample_dir2a1 = results['output2a'] + 'train2a1_{:08d}.png'.format(step)
                sample_dir1b1 = results['output1b'] + 'train1b1_{:08d}.png'.format(step)
                sample_dir2b1 = results['output2b'] + 'train2b1_{:08d}.png'.format(step)

                iu.save_images(samples1a1,
                               size=[sample_image_height, sample_image_width],
                               image_path=sample_dir1a1)
                iu.save_images(samples2a1,
                               size=[sample_image_height, sample_image_width],
                               image_path=sample_dir2a1)
                iu.save_images(samples1b1,
                               size=[sample_image_height, sample_image_width],
                               image_path=sample_dir1b1)
                iu.save_images(samples2b1,
                               size=[sample_image_height, sample_image_width],
                               image_path=sample_dir2b1)

    end_time = time.time() - start_time  # Clocking end

    # Elapsed time
    print("[+] Elapsed time {:.8f}s".format(end_time))  # took about 370s on my machine

    # Close tf.Session
    s.close()

if __name__ == '__main__':
    main()
