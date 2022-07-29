from torch import nn
import torch

class VI(nn.Module):
    def __init__(self):
        super().__init__()
        self.out = nn.Sequential(
            nn.Linear(in_features=(1), out_features=20),
            nn.ReLU(),
            nn.Linear(in_features=20, out_features=10),
            nn.ReLU(),
            nn.Linear(in_features=10, out_features=1),
        )
        self.out_mu = nn.Linear(in_features=1, out_features=1)
        self.out_log_var = nn.Linear(in_features=1, out_features=1)
        
    # we can not backprob with sampling x so we repar... it so we have a distribution to another variable
    def reparametrize(self, mu, log_var):
        sigma = torch.exp(0.5 * log_var)
        eps = torch.randn_like(sigma)
        return mu + sigma * eps

    def forward(self, x):
        out = self.out(x)
        mu = self.out_mu(out)
        log_var = self.out_log_var(out)
        return self.reparametrize(mu, log_var), mu, log_var
    

def ll_gaussian(y, mu, log_var):
    sigma = torch.exp(0.5 * log_var)
    return -0.5 * torch.log(2 * torch.pi * sigma**2) - (1 / (2 * sigma**2))* (y-mu)**2
    

def neg_elbo(y_pred, y, mu, log_var):
    # likelihood of observing y given variational distribution mu and sigma/log_var
    likelihood = ll_gaussian(y, mu, log_var) 
    
    # prior probability of y_pred (given unit normal prior) mean = 0 and std = 1
    log_prior = ll_gaussian(y_pred, torch.tensor(0.), torch.tensor(1.))
    
    # variational probability of y_pred given variational distribution with mu and sigma/log_var
    log_p_q = ll_gaussian(y_pred, mu, log_var)
    
    # return -ELBO averaged over all samples
    return - (likelihood + log_prior - log_p_q).mean()