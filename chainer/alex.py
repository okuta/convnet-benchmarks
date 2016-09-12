import chainer
import chainer.functions as F
from chainer import initializers
import chainer.links as L


class Alex(chainer.Chain):
    insize = 224

    def __init__(self, dtype):
        self.dtype = dtype
        W = initializers.HeNormal(1 / 2 ** 0.5, self.dtype)
        bias = initializers.Zero(self.dtype)
        kwargs = {'initialW': W, 'bias': bias}
        super(Alex, self).__init__(
            conv1=L.Convolution2D(3,  64, 11, stride=4, pad=2, **kwargs),
            conv2=L.Convolution2D(64, 192,  5, pad=2, **kwargs),
            conv3=L.Convolution2D(192, 384,  3, pad=1, **kwargs),
            conv4=L.Convolution2D(384, 256,  3, pad=1, **kwargs),
            conv5=L.Convolution2D(256, 256,  3, pad=1, **kwargs),
            fc6=L.Linear(256 * 6 * 6, 4096, **kwargs),
            fc7=L.Linear(4096, 4096, **kwargs),
            fc8=L.Linear(4096, 1000, **kwargs),
        )

    def forward(self, x_data, y_data, train=True):
        x = chainer.Variable(x_data.astype(self.dtype), volatile=False)
        t = chainer.Variable(y_data, volatile=False)

        h = F.max_pooling_2d(F.relu(self.conv1(x)), 3, stride=2)
        h = F.max_pooling_2d(F.relu(self.conv2(h)), 3, stride=2)
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.max_pooling_2d(F.relu(self.conv5(h)), 3, stride=2)
        h = F.relu(self.fc6(h))
        h = F.relu(self.fc7(h))
        h = self.fc8(h)
        return F.softmax_cross_entropy(h, t), F.accuracy(h, t)