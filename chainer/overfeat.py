import chainer
import chainer.functions as F
from chainer import initializers
import chainer.links as L


class overfeat(chainer.Chain):
    insize = 231

    def __init__(self, dtype):
        self.dtype = dtype
        W = initializers.HeNormal(1 / 2 ** 0.5, self.dtype)
        bias = initializers.Zero(self.dtype)
        kwargs = {'initialW': W, 'bias': bias}
        super(overfeat, self).__init__(
            conv1=L.Convolution2D(   3,   96, 11, stride=4, **kwargs),
            conv2=L.Convolution2D(  96,  256,  5, pad=0, **kwargs),
            conv3=L.Convolution2D( 256,  512,  3, pad=1, **kwargs),
            conv4=L.Convolution2D( 512, 1024,  3, pad=1, **kwargs),
            conv5=L.Convolution2D(1024, 1024,  3, pad=1, **kwargs),
            fc6=L.Linear(1024 * 6 * 6, 3072, **kwargs),
            fc7=L.Linear(3072, 4096, **kwargs),
            fc8=L.Linear(4096, 1000, **kwargs),
        )

    def forward(self, x_data, y_data, train=True):
        x = chainer.Variable(x_data.astype(self.dtype), volatile=False)
        t = chainer.Variable(y_data, volatile=False)

        h = F.max_pooling_2d(F.relu(self.conv1(x)), 2, stride=2)
        h = F.max_pooling_2d(F.relu(self.conv2(h)), 2, stride=2)
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.max_pooling_2d(F.relu(self.conv5(h)), 3, stride=2)
        h = F.relu(self.fc6(h))
        h = F.relu(self.fc7(h))
        h = self.fc8(h)
        return F.softmax_cross_entropy(h, t), F.accuracy(h, t)