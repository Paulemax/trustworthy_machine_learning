from torch.autograd import Function

class GradientReversal(Function):
    '''
    Gradient Reversal Layer as used in Domain Adversarial Training
    Note: only backward is multiplied by alpha as in the original paper
    ctx is the context in which the function is called. In this case we
    just use it to save the alpha parameter for the backwards pass
    '''
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        '''
        The idea of this "layer" is maximize the gradient. 
        '''
        output = -ctx.alpha * grad_output
        return output, None
        

def grad_reverse(x,alpha=1):
    return GradientReversal.apply(x,alpha)
