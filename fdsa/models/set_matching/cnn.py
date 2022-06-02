import torch
import torch.nn as nn
from fdsa.utils.hyperparameters import ACTIVATION_FN_FACTORY, POOLING_FN_FACTORY


class CNNSetMatching(nn.Module):
    """Generalisable CNN module to allow for flexibility in architecture."""

    def __init__(self, **params) -> None:
        """Constructor.

        Args:
            params (dict) containing the following keys:
                input_size (int): The number of input channels/ dimensions.
                output_channels (int): The desired number of output channels.
                conv_layers (int): Number of convolution layers to apply.
                kernel_size (List[(tuple or int)]): Size of the convolving kernel.
                stride (List[(tuple or int)]): Stride of the convolution.
                padding (List[(tuple or int)]): Zero-padding added to both sides of the
                    input.
                padding_mode (str): 'zeros', 'reflect', 'replicate' or 'circular'.
                dilation (List[(tuple or int)]): Spacing between kernel elements.
                conv_activation (str): Activation to apply after convolution.
                    See utils/hyperparameter.py for options.
                
                pooling (str): Type of pooling to apply.
                    See utils/hyperparameter.py for options.
                pooling_kernel_size (List[(tuple or int)]): The size of the window to
                    pool over.
                pooling_kernel_stride (List[(tuple or int)]): The stride of the window.
                pooling_kernel_padding (List[(tuple or int)]): Implicit zero padding to
                    be added on both sides.
                pooling_kernel_dilation (List[(tuple or int)]): Controls the stride of
                    elements in the window.
                
                fc_layers (int): Number of fully connected layers to add.
                fc_units (List[(int)]): List of hidden units for each
                    fully connected layer.
                fc_activation (str): Activation to apply after linear transform.
                    See utils/hyperparameter.py for options.
        """
        super(CNNSetMatching, self).__init__()

        self.img_height = params['img_height']
        self.img_width = params['img_width']
        self.input_channel = params['input_size']
        self.output_channels = params['output_channels']
        self.conv_layers = params['conv_layers']
        self.kernel_size = params['kernel_size']
        self.stride = params['stride']
        self.padding = params['padding']
        self.padding_mode = params['padding_mode']
        self.dilation = params['dilation']
        self.conv_activation = params['conv_activation']

        self.pooling = params['pooling']
        self.pooling_kernel_size = params['pooling_kernel_size']
        self.pooling_kernel_stride = params['pooling_kernel_stride']
        self.pooling_kernel_padding = params['pooling_kernel_padding']
        self.pooling_kernel_dilation = params['pooling_kernel_dilation']

        self.fc_layers = params['fc_layers']
        self.fc_units = params['fc_units']
        self.fc_activation = params['fc_activation']

        modules_conv = []
        out_channels = [self.input_channel] + self.output_channels

        w = self.img_width
        h = self.img_height

        for layer in range(self.conv_layers):
            conv = nn.Conv2d(
                out_channels[layer],
                out_channels[layer + 1],
                self.kernel_size[layer],
                self.stride[layer],
                self.padding[layer],
                self.dilation[layer],
                padding_mode=self.padding_mode
            )

            modules_conv.append(conv)
            w = self.compute_output_img_size(
                w, self.kernel_size[layer], self.padding[layer], self.stride[layer]
            )
            h = self.compute_output_img_size(
                h, self.kernel_size[layer], self.padding[layer], self.stride[layer]
            )
            activation = ACTIVATION_FN_FACTORY[self.conv_activation]

            modules_conv.append(activation)

            pooling = POOLING_FN_FACTORY[self.pooling](
                self.pooling_kernel_size[layer], self.pooling_kernel_stride[layer],
                self.pooling_kernel_padding, self.pooling_kernel_dilation
            )

            modules_conv.append(pooling)

            w = self.compute_output_img_size(
                w, self.pooling_kernel_size[layer], self.pooling_kernel_padding[layer],
                self.pooling_kernel_stride[layer]
            )
            h = self.compute_output_img_size(
                h, self.pooling_kernel_size[layer], self.pooling_kernel_padding[layer],
                self.pooling_kernel_stride[layer]
            )

        self.model_conv = nn.Sequential(*modules_conv)

        self.output_img_size = int(w * h * self.output_channels[-1])

        linear_units = [self.output_img_size] + self.fc_units
        modules_linear = []

        for layer in range(self.fc_layers):
            fc = nn.Linear(linear_units[layer], linear_units[layer + 1])
            modules_linear.append(fc)
            if self.fc_activation is not None:
                modules_linear.append(ACTIVATION_FN_FACTORY[self.fc_activation])

        self.model_fc = nn.Sequential(*modules_linear)

    def compute_output_img_size(
        self, input_size: int, filter_size: int, padding: int, stride: int
    ) -> int:
        """Computes the size of the output from a CNN in one dimension.

        Args:
            input_size (int): The number of input channels/ dimensions.
            filter_size (int): Size of the convolving kernel/pooling window.
            padding (int):  Zero-paddings on both sides for padding number of
                points.
            stride (int): Stride of the convolution.

        Returns:
            int: Output image size along one dimension.
        """
        return 1 + (input_size - filter_size + 2 * padding) / stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies convolutions and specified transformations on input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape
                [batch_size, in_channels, in_height, in_width].

        Returns:
            torch.Tensor: Transformed tensor of shape
                [batch_size, out_channels, out_height, out_width] if there is no
                fc layer. Otherwise, [batch_size, linear_units[-1]]
        """
        x = self.model_conv(x)
        x = x.view(-1, self.output_img_size)
        x = self.model_fc(x)

        return x
