import torch
import torch.nn as nn


def running_global_pool_1d(inputs, pooling_type="MAX"):
    """Same global pool, but only for the elements up to the current element.
  Useful for outputs where the state of future elements is not known.
  Takes no mask as all elements up to the current element are assumed to exist.
  Currently only supports maximum. Equivalent to using a lower triangle bias.
  Args:
    inputs: A tensor of shape [batch_size, sequence_length, input_dims]
      containing the sequences of input vectors.
    pooling_type: Pooling type to use. Currently only supports 'MAX'.
  Returns:
    A tensor of shape [batch_size, sequence_length, input_dims] containing the
    running 'totals'.
  """
    del pooling_type
    output = torch.cummax(inputs, dim=1)[0]
    return output


def global_pool_1d(inputs, pooling_type="MAX", mask=None):
    """Pool elements across the last dimension.
    Useful to convert a list of vectors into a single vector so as
    to get a representation of a set.
    Args:
      inputs: A tensor of shape [batch_size, sequence_length, input_dims]
        containing the sequences of input vectors.
      pooling_type: the pooling type to use, MAX or AVR
      mask: A tensor of shape [batch_size, sequence_length] containing a
        mask for the inputs with 1's for existing elements, and 0's elsewhere.
    Returns:
      A tensor of shape [batch_size, input_dims] containing the sequences of
      transformed vectors.
    """

    if mask is not None:
        mask = mask.unsqueeze_(2)
        inputs = torch.matmul(inputs, mask)

    if pooling_type == "MAX":
        output, indices = torch.max(inputs, 1, keepdim=False, out=None)

    elif pooling_type == "AVR":
        if mask is not None:

            output = torch.sum(inputs, 1, keepdim=False, dtype=None)

            num_elems = torch.sum(mask, 1, keepdim=True)

            output = torch.div(output, torch.max(num_elems, 1))
        else:
            output = torch.mean(inputs, axis=1)

    return output


def shape_list(x):
    """Return list of dims, statically where possible."""
    x = torch.as_tensor(x)


def conv_internal(conv_fn, inputs, filters, kernel_size, **kwargs):
    """Conditional conv_fn making kernel 1d or 2d depending on inputs shape."""
    static_shape = inputs.size()
    if not static_shape or len(static_shape) != 4:
        raise ValueError(
            "Inputs to conv must have statically known rank 4. "
            "Shape: " + str(static_shape)
        )

    if kwargs.get("padding") == "LEFT":
        dilation_rate = (1, 1)
        if "dilation_rate" in kwargs:
            dilation_rate = kwargs["dilation_rate"]
        assert kernel_size[0] % 2 == 1 and kernel_size[1] % 2 == 1
        height_padding = 2 * (kernel_size[0] // 2) * dilation_rate[0]
        if torch.equal(shape_list(inputs)[2], 1):
            cond_padding = torch.tensor(0)
        else:
            cond_padding = torch.tensor(2 * (kernel_size[1] // 2) * dilation_rate[1])
        width_padding = 0 if static_shape[2] == 1 else cond_padding
        padding = (0, 0, height_padding, 0, width_padding, 0, 0, 0)
        inputs = nn.functional.pad(inputs, padding)

        inputs = inputs.view(static_shape[0], None, None, static_shape[3])
        kwargs["padding"] = "VALID"

    def conv2d_kernel(kernel_size_arg):
        """Call conv2d but add suffix to name."""

        result = nn.Conv2d(inputs, filters, kernel_size_arg, groups=inputs)

        return result

    return conv2d_kernel(kernel_size)


def conv(inputs, filters, kernel_size, dilation_rate=(1, 1), **kwargs):

    def _conv2d(x, *args, **kwargs):
        return nn.Conv2d(*args, **kwargs)(x)

    return conv_internal(
        _conv2d, inputs, filters, kernel_size, dilation_rate=dilation_rate, **kwargs
    )


def conv1d(inputs, filters, kernel_size, dilation_rate=1, **kwargs):
    return torch.squeeze(
        conv(
            torch.expand_dims(inputs, 2),
            filters, (kernel_size, 1),
            dilation_rate=(dilation_rate, 1),
            **kwargs
        ), 2
    )


def linear_set_layer(
    layer_size, inputs, context=None, activation_fn=nn.ReLU(), dropout=0.0
):
    """Basic layer type for doing funky things with sets.
  Applies a linear transformation to each element in the input set.
  If a context is supplied, it is concatenated with the inputs.
    e.g. One can use global_pool_1d to get a representation of the set which
    can then be used as the context for the next layer.
  TODO: Add bias add (or control the biases used).
  Args:
    layer_size: Dimension to transform the input vectors to.
    inputs: A tensor of shape [batch_size, sequence_length, input_dims]
      containing the sequences of input vectors.
    context: A tensor of shape [batch_size, context_dims] containing a global
      statistic about the set.
    activation_fn: The activation function to use.
    dropout: Dropout probability.
    name: name.
  Returns:
    Tensor of shape [batch_size, sequence_length, output_dims] containing the
    sequences of transformed vectors.
  """

    batch_size, input_dims, sequence_length = inputs.size()

    linear_filter = nn.Conv1d(input_dims, layer_size, 1)
    outputs = linear_filter(inputs)

    if context is not None:

        if len(context.get_shape().as_list()) == 2:
            context = torch.expand(context, axis=1)
        cont_tfm = conv1d(context, layer_size, 1, activation=None)
        outputs += cont_tfm
    if activation_fn is not None:
        outputs = activation_fn(outputs)
    if dropout != 0.0:
        outputs = nn.functional.dropout(outputs, 1.0 - dropout)
    return outputs


def ravanbakhsh_set_layer(
    layer_size,
    inputs,
    mask=None,
    sequential=False,
    activation_fn=nn.Tanh(),
    dropout=0.0
):
    """Layer from Deep Sets paper: https://arxiv.org/abs/1611.04500 .
  More parameter-efficient version of a linear-set-layer with context.
  Args:
    layer_size: Dimension to transform the input vectors to.
    inputs: A tensor of shape [batch_size, sequence_length, vector]
      containing the sequences of input vectors.
    mask: A tensor of shape [batch_size, sequence_length] containing a
      mask for the inputs with 1's for existing elements, and 0's elsewhere.
    sequential: If true, will use a running global pool so each element will
      only depend on those before it. Set true if this layer is being used in
      an output sequence.
    activation_fn: The activation function to use.
    dropout: dropout.
    name: name.
  Returns:
    Tensor of shape [batch_size, sequence_length, vector] containing the
    sequences of transformed vectors.
  """
    del dropout

    if sequential:
        return linear_set_layer(
            layer_size,
            inputs - running_global_pool_1d(inputs),
            activation_fn=activation_fn
        )
    return linear_set_layer(
        layer_size,
        inputs - global_pool_1d(inputs, mask=mask).unsqueeze(1),
        activation_fn=activation_fn
    )
