from torch.autograd import Function

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # 反向传播时,需要对梯度取负
        output = grad_output.neg() * ctx.alpha

        return output, None