import torch
from torch.autograd import Function
from ..build.lib import GANet
from torch.autograd import Variable
#import GANet
		

class SgaFunction(Function):
    def __init__(self):
        self.wsize = 5
#        self.radius = radius
    def forward(self, input, g0, g1, g2, g3):
        self.input = input
        self.g0 = g0
        self.g1 = g1
        self.g2 = g2
        self.g3 = g3
        assert(input.is_contiguous() == True and g0.is_contiguous() == True and g1.is_contiguous() == True and g2.is_contiguous() == True and g3.is_contiguous() == True)
        with torch.cuda.device_of(input):
            num, channels, depth, height, width = input.size()
            output = input.new().resize_(num, channels, depth, height, width).zero_()
            temp_out = input.new().resize_(num, channels, depth, height, width).zero_()
            mask = input.new().resize_(num, channels, depth, height, width).zero_()
            GANet.sga_cuda_forward(input, g0, g1, g2, g3, temp_out, output, mask)
 #           GANet.sga_cuda_forward(input, filters, output, self.radius)
            
            output = output.contiguous()
        self.save_for_backward(temp_out, mask)
        return output
    def backward(self, gradOutput):
        temp_out, mask = self.saved_variables
#        print temp_out.size()
#        print mask.size()
        assert(gradOutput.is_contiguous() == True)
        with torch.cuda.device_of(gradOutput):
            num, channels, depth, height, width = self.input.size()
            _, _, fsize, _, _ = self.g0.size()
#            print fsize            
            gradInput = gradOutput.new().resize_(num, channels, depth, height, width).zero_()
            grad0 = gradOutput.new().resize_(num, channels, fsize, height, width).zero_()
            grad1 = gradOutput.new().resize_(num, channels, fsize, height, width).zero_()
            grad2 = gradOutput.new().resize_(num, channels, fsize, height, width).zero_()
            grad3 = gradOutput.new().resize_(num, channels, fsize, height, width).zero_()
            temp_grad = gradOutput.new().resize_(num, channels, depth, height, width).zero_()     
            max_idx = gradOutput.new().resize_(num, channels, height, width).zero_()    

            GANet.sga_cuda_backward(self.input, self.g0, self.g1, self.g2, self.g3, temp_out, mask, max_idx, gradOutput, temp_grad, gradInput, grad0, grad1, grad2, grad3)
#            GANet.lga_cuda_backward(self.input, self.filters, gradOutput, gradInput, gradFilters, self.radius)
            gradInput = gradInput.contiguous()
            grad0 = grad0.contiguous()
            grad1 = grad1.contiguous()
            grad2 = grad2.contiguous()
            grad3 = grad3.contiguous()
        return gradInput, grad0, grad1, grad2, grad3
		
		
class Lga3d3Function(Function):
    def __init__(self, radius=1):
        self.radius = radius       
    def forward(self, input, filters):
        self.input = input
        self.filters = filters
        assert(input.is_contiguous() == True and filters.is_contiguous() == True)
        with torch.cuda.device_of(input):
            num, channels, depth, height, width = input.size()
            temp_out1 = input.new().resize_(num, channels, depth, height, width).zero_()
            temp_out2 = input.new().resize_(num, channels, depth, height, width).zero_()
            output = input.new().resize_(num, channels, depth, height, width).zero_()
            GANet.lga3d_cuda_forward(input, filters, temp_out1, self.radius)
            GANet.lga3d_cuda_forward(temp_out1, filters, temp_out2, self.radius)
            GANet.lga3d_cuda_forward(temp_out2, filters, output, self.radius)
            output = output.contiguous()
        self.save_for_backward(temp_out1, temp_out2)
        return output
    def backward(self, gradOutput):
        temp_out1, temp_out2 = self.saved_variables
        assert(gradOutput.is_contiguous() == True)
        with torch.cuda.device_of(gradOutput):
            num, channels, depth, height, width = self.input.size()
            _, _, fsize, _, _ = self.filters.size()
#            gradInput = gradOutput.new().resize_(num, channels, height, width).zero_()
            gradFilters = gradOutput.new().resize_(num, channels, fsize, height, width).zero_()
            GANet.lga3d_cuda_backward(temp_out2, self.filters, gradOutput, temp_out2, gradFilters, self.radius)
            GANet.lga3d_cuda_backward(temp_out1, self.filters, temp_out2, temp_out1, gradFilters, self.radius)
#            temp_out[...] = 0
            GANet.lga3d_cuda_backward(self.input, self.filters, temp_out1, temp_out2, gradFilters, self.radius)
#            temp_out[...] = gradOutput[...]
            temp_out2 = temp_out2.contiguous()
            gradFilters = gradFilters.contiguous()
        return temp_out2, gradFilters
class Lga3d2Function(Function):
    def __init__(self, radius=1):
        self.radius = radius       
    def forward(self, input, filters):
        self.input = input
        self.filters = filters
        assert(input.is_contiguous() == True and filters.is_contiguous() == True)
        with torch.cuda.device_of(input):
            num, channels, depth, height, width = input.size()
            temp_out = input.new().resize_(num, channels, depth, height, width).zero_()
            output = input.new().resize_(num, channels, depth, height, width).zero_()
            GANet.lga3d_cuda_forward(input, filters, temp_out, self.radius)
            GANet.lga3d_cuda_forward(temp_out, filters, output, self.radius)
            output = output.contiguous()
        self.save_for_backward(temp_out)
        return output
    def backward(self, gradOutput):
        temp_out, = self.saved_variables
        assert(gradOutput.is_contiguous() == True)
        with torch.cuda.device_of(gradOutput):
            num, channels, depth, height, width = self.input.size()
            _, _, fsize, _, _ = self.filters.size()
#            gradInput = gradOutput.new().resize_(num, channels, height, width).zero_()
            gradFilters = gradOutput.new().resize_(num, channels, fsize, height, width).zero_()
            GANet.lga3d_cuda_backward(temp_out, self.filters, gradOutput, temp_out, gradFilters, self.radius)
#            temp_out[...] = 0
            GANet.lga3d_cuda_backward(self.input, self.filters, temp_out, gradOutput, gradFilters, self.radius)
            temp_out[...] = gradOutput[...]
            temp_out = temp_out.contiguous()
            gradFilters = gradFilters.contiguous()
        return temp_out, gradFilters

class Lga3dFunction(Function):
    def __init__(self, radius=2):
        self.radius = radius
       
    def forward(self, input, filters):
        self.input = input
        self.filters = filters
        assert(input.is_contiguous() == True and filters.is_contiguous() == True)
        with torch.cuda.device_of(input):
            num, channels, depth, height, width = input.size()
            output = input.new().resize_(num, channels, depth, height, width).zero_()
            GANet.lga3d_cuda_forward(input, filters, output, self.radius)
            output = output.contiguous()
        return output
    def backward(self, gradOutput):
        assert(gradOutput.is_contiguous() == True)
        with torch.cuda.device_of(gradOutput):
            num, channels, depth, height, width = self.input.size()
            _, _, fsize, _, _ = self.filters.size()
            gradInput = gradOutput.new().resize_(num, channels, depth, height, width).zero_()
            gradFilters = gradOutput.new().resize_(num, channels, fsize, height, width).zero_()
            GANet.lga3d_cuda_backward(self.input, self.filters, gradOutput, gradInput, gradFilters, self.radius)
            gradInput = gradInput.contiguous()
            gradFilters = gradFilters.contiguous()
        return gradInput, gradFilters

class Lga3Function(Function):
    def __init__(self, radius=1):
        self.radius = radius       
    def forward(self, input, filters):
        self.input = input
        self.filters = filters
        assert(input.is_contiguous() == True and filters.is_contiguous() == True)
        with torch.cuda.device_of(input):
            num, channels, height, width = input.size()
            temp_out1 = input.new().resize_(num, channels, height, width).zero_()
            temp_out2 = input.new().resize_(num, channels, height, width).zero_()
            output = input.new().resize_(num, channels, height, width).zero_()
            GANet.lga_cuda_forward(input, filters, temp_out1, self.radius)
            GANet.lga_cuda_forward(temp_out1, filters, temp_out2, self.radius)
            GANet.lga_cuda_forward(temp_out2, filters, output, self.radius)
            output = output.contiguous()
        self.save_for_backward(temp_out1, temp_out2)
        return output
    def backward(self, gradOutput):
        temp_out1, temp_out2 = self.saved_variables
        assert(gradOutput.is_contiguous() == True)
        with torch.cuda.device_of(gradOutput):
            num, channels, height, width = self.input.size()
            _, fsize, _, _ = self.filters.size()
#            gradInput = gradOutput.new().resize_(num, channels, height, width).zero_()
            gradFilters = gradOutput.new().resize_(num, fsize, height, width).zero_()
            GANet.lga_cuda_backward(temp_out2, self.filters, gradOutput, temp_out2, gradFilters, self.radius)
            GANet.lga_cuda_backward(temp_out1, self.filters, temp_out2, temp_out1, gradFilters, self.radius)
#            temp_out[...] = 0
            GANet.lga_cuda_backward(self.input, self.filters, temp_out1, temp_out2, gradFilters, self.radius)
#            temp_out[...] = gradOutput[...]
            temp_out2 = temp_out2.contiguous()
            gradFilters = gradFilters.contiguous()
        return temp_out2, gradFilters
class Lga2Function(Function):
    def __init__(self, radius=1):
        self.radius = radius       
    def forward(self, input, filters):
        self.input = input
        self.filters = filters
        assert(input.is_contiguous() == True and filters.is_contiguous() == True)
        with torch.cuda.device_of(input):
            num, channels, height, width = input.size()
            temp_out = input.new().resize_(num, channels, height, width).zero_()
            output = input.new().resize_(num, channels, height, width).zero_()
            GANet.lga_cuda_forward(input, filters, temp_out, self.radius)
            GANet.lga_cuda_forward(temp_out, filters, output, self.radius)
            output = output.contiguous()
        self.save_for_backward(temp_out)
        return output
    def backward(self, gradOutput):
        temp_out, = self.saved_variables
        assert(gradOutput.is_contiguous() == True)
        with torch.cuda.device_of(gradOutput):
            num, channels, height, width = self.input.size()
            _, fsize, _, _ = self.filters.size()
#            gradInput = gradOutput.new().resize_(num, channels, height, width).zero_()
            gradFilters = gradOutput.new().resize_(num, fsize, height, width).zero_()
            GANet.lga_cuda_backward(temp_out, self.filters, gradOutput, temp_out, gradFilters, self.radius)
#            temp_out[...] = 0
            GANet.lga_cuda_backward(self.input, self.filters, temp_out, gradOutput, gradFilters, self.radius)
            temp_out[...] = gradOutput[...]
            temp_out = temp_out.contiguous()
            gradFilters = gradFilters.contiguous()
        return temp_out, gradFilters

class LgaFunction(Function):
    def __init__(self, radius=2):
        self.radius = radius
       
    def forward(self, input, filters):
        self.input = input
        self.filters = filters
        assert(input.is_contiguous() == True and filters.is_contiguous() == True)
        with torch.cuda.device_of(input):
            num, channels, height, width = input.size()
            output = input.new().resize_(num, channels, height, width).zero_()
            GANet.lga_cuda_forward(input, filters, output, self.radius)
            output = output.contiguous()
        return output
    def backward(self, gradOutput):
        assert(gradOutput.is_contiguous() == True)
        with torch.cuda.device_of(gradOutput):
            num, channels, height, width = self.input.size()
            _, fsize, _, _ = self.filters.size()
            gradInput = gradOutput.new().resize_(num, channels, height, width).zero_()
            gradFilters = gradOutput.new().resize_(num, fsize, height, width).zero_()
            GANet.lga_cuda_backward(self.input, self.filters, gradOutput, gradInput, gradFilters, self.radius)
            gradInput = gradInput.contiguous()
            gradFilters = gradFilters.contiguous()
        return gradInput, gradFilters
class MyLoss2Function(Function):
    def __init__(self, thresh=1, alpha=2):
        self.thresh = thresh
        self.alpha = alpha
    def forward(self, input1, input2):
        self.diff = input1 - input2
        temp=torch.abs(self.diff)
        temp[temp < self.thresh] = temp[temp < self.thresh] ** 2 / self.thresh
        tag = (temp <= self.thresh + self.alpha) & (temp >= self.thresh)
        temp[tag]=temp[tag] * 2 - (temp[tag] - self.thresh) ** 2 /(2.0 * self.alpha) - self.thresh
        temp[temp > self.thresh + self.alpha] += (self.alpha / 2.0)
        
        return torch.mean(temp)
    def backward(self, gradOutput):
        scale = torch.abs(self.diff)
        scale[scale > self.thresh + self.alpha] = 1
        tag = (scale <= self.thresh+self.alpha) & (scale >= self.thresh)
        scale[tag] = 2 - (scale[tag] - self.thresh) / self.alpha
        tag = scale < self.thresh
        scale[tag] = 2*scale[tag] / self.thresh
        self.diff[self.diff > 0] = 1.0
        self.diff[self.diff < 0] = -1.0
        self.diff = self.diff * scale * gradOutput / scale.numel()
        return self.diff, Variable(torch.Tensor([0]))

class MyLossFunction(Function):
    def __init__(self, upper_thresh=5, lower_thresh=1):
        self.upper_thresh = upper_thresh
        self.lower_thresh = lower_thresh
    def forward(self, input1, input2):
        self.diff = input1 - input2
        return torch.mean(torch.abs(self.diff))
    def backward(self, gradOutput):
        scale = torch.abs(self.diff)
        scale[scale > self.upper_thresh] = 1
        tag = (scale <= self.upper_thresh) & (scale >= self.lower_thresh)
        scale[tag] = 2 - torch.abs(scale[tag]-(self.upper_thresh + self.lower_thresh)/2.)/2.
        self.diff[self.diff > 0] = 1
        self.diff[self.diff < 0] = -1
        self.diff = self.diff * scale * gradOutput
        return self.diff, Variable(torch.Tensor([0]))

