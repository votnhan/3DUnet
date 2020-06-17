import keras.backend as K

def loss_VAE(input_shape, z_mean, z_var, weight_L2=1., weight_KL=1.):
    """
    loss(input_shape, inp, out_VAE, z_mean, z_var, e=1e-8, weight_L2=0.1, weight_KL=0.1)
    ------------------------------------------------------
    Since keras does not allow custom loss functions to have arguments
    other than the true and predicted labels, this function acts as a wrapper
    that allows us to implement the custom loss used in the paper, involving
    outputs from multiple layers.
    L = - L<dice> + weight_L2 ∗ L<L2> + weight_KL ∗ L<KL>
    - L<dice> is the dice loss between input and segmentation output.
    - L<L2> is the L2 loss between the output of VAE part and the input.
    - L<KL> is the standard KL divergence loss term for the VAE.
    Parameters
    ----------
    `input_shape`: A 4-tuple, required
        The shape of an image as the tuple (c, H, W, D), where c is
        the no. of channels; H, W and D is the height, width and depth of the
        input image, respectively.
    `inp`: An keras.layers.Layer instance, required
        The input layer of the model. Used internally.
    `out_VAE`: An keras.layers.Layer instance, required
        The output of VAE part of the decoder. Used internally.
    `z_mean`: An keras.layers.Layer instance, required
        The vector representing values of mean for the learned distribution
        in the VAE part. Used internally.
    `z_var`: An keras.layers.Layer instance, required
        The vector representing values of variance for the learned distribution
        in the VAE part. Used internally.
    `e`: Float, optional
        A small epsilon term to add in the denominator to avoid dividing by
        zero and possible gradient explosion.
    `weight_L2`: A real number, optional
        The weight to be given to the L2 loss term in the loss function. Adjust to get best
        results for your task. Defaults to 0.1.
    `weight_KL`: A real number, optional
        The weight to be given to the KL loss term in the loss function. Adjust to get best
        results for your task. Defaults to 0.1.
    Returns
    -------
    loss_(y_true, y_pred): A custom keras loss function
        This function takes as input the predicted and ground labels, uses them
        to calculate the dice loss. Combined with the L<KL> and L<L2 computed
        earlier, it returns the total loss.
    """
    c, H, W, D = input_shape
    n = c * H * W * D

    #loss_L2 = mse(inp, out_VAE)

    loss_KL = (1 / n) * K.sum(
        K.exp(z_var) + K.square(z_mean) - 1. - z_var,
        axis=-1
    )

    def loss_(y_true, y_pred):
        loss_L2 = K.mean(K.square(y_true - y_pred), axis=(1, 2, 3, 4))
        return weight_L2 * loss_L2 + weight_KL * loss_KL

    return loss_