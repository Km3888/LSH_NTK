import torch
import torch.functional as F

def compute_approximation(model,data):
    y_0=model(data)
    w_0=get_param_vector(model)
    grads=[]
    for datapoint in data:
        grad_vector=get_grad_vector(model,datapoint)
        grads.append(grad_vector)
    G=torch.stack(grads)
    return y_0,w_0,G


def compute_gradient_magnitude(data,targets,model):
    magnitudes=[]
    for i,x in enumerate(data):
        model.zero_grad()
        output=model(x)
        loss=F.mse_loss(targets[i],output)
        loss.backward()

        grads = []
        for param in model.parameters():
            grads.append(param.grad.view(-1))
        magnitude=torch.norm(torch.cat(grads))
        magnitudes.append(magnitude)
    magnitudes=torch.tensor(magnitudes).to(data.device)
    return magnitudes

def estimate_gradient_magnitude(y_0,w_0,G,w,targets):
    abs_a=torch.norm(G,dim=1)
    increment=(w-w_0)
    effect= torch.matmul(G,increment)
    estimated_output=(effect+y_0.view([-1]))
    err=targets-estimated_output

    grad_estimate=torch.abs(err*abs_a)
    return grad_estimate

def get_param_vector(model):
    params=[]
    for param in model.parameters():
        params.append(param.view(-1))
    return torch.cat(params)

def get_grad_vector(model,input):
    output=model(input)
    model.zero_grad()
    output.backward()
    grads=[]
    for param in model.parameters():
        grads.append(param.grad.view(-1))
    return torch.cat(grads)

def get_kl_div(curr_model,d):
    w=get_param_vector(curr_model)
    actual_gradient_magnitude = compute_gradient_magnitude(d['data'], d['targets'], curr_model)
    actual_gradient_magnitude/=actual_gradient_magnitude.sum()

    estimated_gradient_magnitude = estimate_gradient_magnitude(d['y_0'], d['w_0'], d['G'], w, d['targets'])
    estimated_gradient_magnitude /= estimated_gradient_magnitude.sum()

    uniform_dist = torch.ones_like(estimated_gradient_magnitude)
    uniform_dist /=uniform_dist.sum()

    output=kl_div(actual_gradient_magnitude,estimated_gradient_magnitude)
    baseline=kl_div(actual_gradient_magnitude,uniform_dist)
    return output,baseline

def kl_div(a,b):
    bit_diffs=torch.log(a/b)
    weighted_by_prob=a*bit_diffs
    return weighted_by_prob.sum()
