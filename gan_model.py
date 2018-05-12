import tensorflow as tf
tf.set_random_seed(777)  # reproducibility

class GAN:

    def __init__(self, s, batch_size=32, input_height=30, input_width=30, channel=1, n_classes=20,
                 sample_num=1 * 1, sample_size=1, output_height=30, output_width=30,
                 n_input=900, fc_unit=128, z_dim=128*2, g_lr=10e-4, d_lr=1e-4, epsilon=1e-9):

        self.s = s
        self.batch_size = batch_size

        self.input_height = input_height
        self.input_width = input_width
        self.channel = channel
        self.n_classes = n_classes

        self.sample_num = sample_num
        self.sample_size = sample_size
        self.output_height = output_height
        self.output_width = output_width

        self.n_input = n_input
        self.fc_unit = fc_unit

        self.z_dim = z_dim
        self.beta1 = 0.5
        self.d_lr, self.g_lr = d_lr, g_lr
        self.eps = epsilon

        # pre-defined
        self.d1_loss = 0.
        self.g1_loss = 0.
        self.d2_loss = 0.
        self.g2_loss = 0.
        self.d3_loss1 = 0.
        self.d3_loss2 = 0.
        self.d1_op = None
        self.g1_op = None
        self.d2_op = None
        self.g2_op = None
        self.d3_op1 = None
        self.d3_op2 = None

        self.merged = None
        self.writer = None
        self.saver = None

        # Placeholder
        self.x1 = tf.placeholder(tf.float32, shape=[None, self.n_input], name="x1-image")  # (-1, 900)
        self.z1 = tf.placeholder(tf.float32, shape=[None, self.z_dim], name='z1-noise')    # (-1, 128)

        self.x2 = tf.placeholder(tf.float32, shape=[None, self.n_input], name="x2-image")  # (-1, 900)
        self.z2 = tf.placeholder(tf.float32, shape=[None, self.z_dim], name='z2-noise')    # (-1, 128)

        self.x3 = tf.placeholder(tf.float32, shape=[None, self.n_input], name="x3-image")  # (-1, 900)

        self.build_gan()  # build GAN model

    def discriminator1(self, x1, reuse=None):
        with tf.variable_scope("discriminator1", reuse=reuse):
            x1 = tf.layers.flatten(x1)
            x1 = tf.layers.dense(x1, self.fc_unit, activation=tf.nn.leaky_relu, name='d1-fc-1')
            x1 = tf.layers.dense(x1, 1, activation=None, name='d1-fc-2')
        return x1

    def discriminator2(self, x2, reuse=None):
        with tf.variable_scope("discriminator2", reuse=reuse):
            x2 = tf.layers.flatten(x2)
            x2 = tf.layers.dense(x2, self.fc_unit, activation=tf.nn.leaky_relu, name='d2-fc-1')
            x2 = tf.layers.dense(x2, 1, activation=None, name='d2-fc-2')
        return x2

    def discriminator3(self, x3, reuse=None):
        with tf.variable_scope("discriminator3", reuse=reuse):
            x3 = tf.layers.flatten(x3)
            x3 = tf.layers.dense(x3, self.fc_unit, activation=tf.nn.leaky_relu, name='d3-fc-1')
            x3 = tf.layers.dense(x3, 1, activation=None, name='d3-fc-2')
        return x3

    def generator1(self, z1, reuse=None):
        with tf.variable_scope("generator1", reuse=reuse):
            x1 = tf.layers.dense(z1, self.fc_unit, activation=tf.nn.leaky_relu, name='g1-fc-1')
            x1 = tf.layers.dense(x1, self.n_input*2, activation=tf.nn.sigmoid, name='g1-fc-2')
        return x1

    def generator2(self, z2, reuse=None):
        with tf.variable_scope("generator2", reuse=reuse):
            x2 = tf.layers.dense(z2, self.fc_unit, activation=tf.nn.leaky_relu, name='g2-fc-1')
            x2 = tf.layers.dense(x2, self.n_input*2, activation=tf.nn.sigmoid, name='g2-fc-2')
        return x2

    def build_gan(self):
        # Generators g1 and g2
        self.g1 = self.generator1(self.z1)
        self.g2 = self.generator2(self.z2)
        self.gAa = self.g1[:,:900]
        d1_real = self.discriminator1(self.x1)
        d1_fake = self.discriminator1(self.gAa, reuse=True)

        self.gBa = self.g2[:,:900]
        d2_real = self.discriminator2(self.x2)
        d2_fake = self.discriminator2(self.gBa, reuse=True)

        d3_real = self.discriminator3(self.x3)
        self.gAb = self.g1[:, 900:1800]
        self.gBb = self.g2[:, 900:1800]
        d3_fake1 = self.discriminator3(self.gAb, reuse=True)
        d3_fake2 = self.discriminator3(self.gBb, reuse=True)
        # General GAN loss function referred in the paper

        # Softmax Loss
        z1_b = tf.reduce_sum(tf.exp(-d1_real)) + tf.reduce_sum(tf.exp(-d1_fake)) + self.eps
        z2_b = tf.reduce_sum(tf.exp(-d2_real)) + tf.reduce_sum(tf.exp(-d2_fake)) + self.eps
        z3_b1 =  tf.reduce_sum(tf.exp(-d3_real)) + tf.reduce_sum(tf.exp(-d3_fake1)) + self.eps
        z3_b2 =  tf.reduce_sum(tf.exp(-d3_real)) + tf.reduce_sum(tf.exp(-d3_fake2)) + self.eps
        b_plus = self.batch_size
        b_minus = self.batch_size

        g1_loss_due_to_d3 = tf.reduce_sum(d3_real / b_plus) + tf.reduce_sum(d3_fake1 / b_minus) + tf.log(z3_b1)
        g2_loss_due_to_d3 = tf.reduce_sum(d3_real / b_plus) + tf.reduce_sum(d3_fake2 / b_minus) + tf.log(z3_b2)

        self.g1_loss = tf.reduce_sum(d1_real / b_plus) + tf.reduce_sum(d1_fake / b_minus) + tf.log(z1_b) + g1_loss_due_to_d3
        self.g2_loss = tf.reduce_sum(d2_real / b_plus) + tf.reduce_sum(d2_fake / b_minus) + tf.log(z2_b) + g2_loss_due_to_d3

        self.d1_loss = tf.reduce_sum(d1_real / b_plus) + tf.log(z1_b)
        self.d2_loss = tf.reduce_sum(d2_real / b_plus) + tf.log(z2_b)

        self.d3_loss1 = tf.reduce_sum(d3_real / b_plus) + tf.log(z3_b1)
        self.d3_loss2 = tf.reduce_sum(d3_real / b_plus) + tf.log(z3_b2)

        tf.summary.histogram("z1-noise", self.z1)
        tf.summary.histogram("z2-noise", self.z2)

        tf.summary.histogram("d1_real", d1_real)
        tf.summary.histogram("d1_fake", d1_fake)
        tf.summary.scalar("d1_loss", self.d1_loss)
        tf.summary.scalar("g1_loss", self.g1_loss)

        tf.summary.histogram("d2_real", d2_real)
        tf.summary.histogram("d2_fake", d2_fake)
        tf.summary.scalar("d2_loss", self.d2_loss)
        tf.summary.scalar("g2_loss", self.g2_loss)

        tf.summary.histogram("d3_real", d3_real)
        tf.summary.histogram("d3_fake1", d3_fake1)
        tf.summary.histogram("d3_fake2", d3_fake2)

        tf.summary.scalar("d3_loss1", self.d3_loss1)
        tf.summary.scalar("d3_loss2", self.d3_loss2)
        # Optimizer
        t1_vars = tf.trainable_variables()
        t2_vars = tf.trainable_variables()
        t3_vars = tf.trainable_variables()
        d1_params = [v for v in t1_vars if v.name.startswith('discriminator1')]
        g1_params = [v for v in t1_vars if v.name.startswith('generator1')]

        d2_params = [v for v in t2_vars if v.name.startswith('discriminator2')]
        g2_params = [v for v in t2_vars if v.name.startswith('generator2')]

        d3_params = [v for v in t3_vars if v.name.startswith('discriminator3')]

        self.d1_op = tf.train.AdamOptimizer(learning_rate=self.d_lr,
                                           beta1=self.beta1).minimize(self.d1_loss, var_list=d1_params)
        self.g1_op = tf.train.AdamOptimizer(learning_rate=self.g_lr,
                                           beta1=self.beta1).minimize(self.g1_loss, var_list=g1_params)
        self.d2_op = tf.train.AdamOptimizer(learning_rate=self.d_lr,
                                           beta1=self.beta1).minimize(self.d2_loss, var_list=d2_params)
        self.g2_op = tf.train.AdamOptimizer(learning_rate=self.g_lr,
                                           beta1=self.beta1).minimize(self.g2_loss, var_list=g2_params)
        self.d3_op1 = tf.train.AdamOptimizer(learning_rate=self.d_lr,
                                           beta1=self.beta1).minimize(self.d3_loss1, var_list=d3_params)
        self.d3_op2 = tf.train.AdamOptimizer(learning_rate=self.d_lr,
                                           beta1=self.beta1).minimize(self.d3_loss2, var_list=d3_params)
        # Merge summary
        self.merged = tf.summary.merge_all()

        # Model saver
        self.saver = tf.train.Saver(max_to_keep=1)
        self.writer = tf.summary.FileWriter('./model/', self.s.graph)