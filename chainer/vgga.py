import chainer
import chainer.functions as F
from chainer import initializers
import chainer.links as L


class vgga(chainer.Chain):
    insize = 224

    def __init__(self, dtype):
        self.dtype = dtype
        W = initializers.HeNormal(1 / 2 ** 0.5, self.dtype)
        bias = initializers.Zero(self.dtype)
        kwargs = {'initialW': W, 'bias': bias}
        super(vgga, self).__init__(
            conv1=L.Convolution2D(  3,  64, 3, stride=1, pad=1, **kwargs),
            conv2=L.Convolution2D( 64, 128, 3, stride=1, pad=1, **kwargs),
            conv3=L.Convolution2D(128, 256, 3, stride=1, pad=1, **kwargs),
            conv4=L.Convolution2D(256, 256, 3, stride=1, pad=1, **kwargs),
            conv5=L.Convolution2D(256, 512, 3, stride=1, pad=1, **kwargs),
            conv6=L.Convolution2D(512, 512, 3, stride=1, pad=1, **kwargs),
            conv7=L.Convolution2D(512, 512, 3, stride=1, pad=1, **kwargs),
            conv8=L.Convolution2D(512, 512, 3, stride=1, pad=1, **kwargs),
            fc6=L.Linear(512 * 7 * 7, 4096, **kwargs),
            fc7=L.Linear(4096, 4096, **kwargs),
            fc8=L.Linear(4096, 1000, **kwargs),
        )

    def forward(self, x_data, y_data, train=True):
        x = chainer.Variable(x_data.astype(self.dtype), volatile=False)
        t = chainer.Variable(y_data, volatile=False)

        h = F.max_pooling_2d(F.relu(self.conv1(x)), 2, stride=2)
        h = F.max_pooling_2d(F.relu(self.conv2(h)), 2, stride=2)
        h = F.relu(self.conv3(h))
        h = F.max_pooling_2d(F.relu(self.conv4(h)), 2, stride=2)
        h = F.relu(self.conv5(h))
        h = F.max_pooling_2d(F.relu(self.conv6(h)), 2, stride=2)
        h = F.relu(self.conv7(h))
        h = F.max_pooling_2d(F.relu(self.conv8(h)), 2, stride=2)
        h = F.relu(self.fc6(h))
        h = F.relu(self.fc7(h))
        h = self.fc8(h)
        return F.softmax_cross_entropy(h, t), F.accuracy(h, t)