import os
import pandas as pd
import numpy as np
import collections
import sys
import torch.nn.functional as F
from .modelBase import modelBase
from utils import CHARAS_LIST
import pickle as pkl
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, TensorDataset

from tqdm import tqdm
MAX_EPOCH = 200


class CA_base_wyh(nn.Module,modelBase):
    def __init__(self,name,omit_char=[],device='cuda'):
        nn.Module.__init__(self)
        modelBase.__init__(self,name)
        self.ca_beta=None
        self.ca_factor=None
        self.train_loader=None
        self.valid_loader=None
        self.test_loader=None
        self.optimizer=None
        self.criterion=None
        self.factor_nn_pred=[]
        self.name=name
        self.device=device
    
    def forward(self,beta_input,factor_input):
        beta_output=self.ca_beta(beta_input)
        factor_output=self.ca_factor(factor_input)
        output=torch.mm(beta_output,factor_output.T)
        return output
    
    
    
    def _get_dataset(self,period):
        with open('./data/mon_list.pkl','rb') as file:
            month_list=pkl.load(file).to_list()
        if period=='train':
            month=[i for i in month_list if i>=self.train_period[0] and i<=self.train_period[1]]
        if period=='valid':
            month=[i for i in month_list if i>=self.valid_period[0] and i<=self.valid_period[1]]
        if period=='test':
            month=[i for i in month_list if i>=self.test_period[0] and i<=self.test_period[1]]    
        
        self.p_charas=pd.read_pickle('./data/p_charas.pkl')
        self.portfolio_ret=pd.read_pickle('./data/portfolio_rets.pkl')
        
        betas=[]
        factors=[]
        labels=[]
        for mon in month:
            beta=self.p_charas.loc[self.p_charas['DATE']==mon][CHARAS_LIST].T.values    #  94*94
            label=self.portfolio_ret.loc[self.portfolio_ret['DATE']==mon][CHARAS_LIST].T.values
            factor=self.portfolio_ret.loc[self.portfolio_ret['DATE']==mon][CHARAS_LIST].T.values   #94*94
            
            betas.append(beta)
            factors.append(factor)
            labels.append(label)
            
        betas_tensor=torch.tensor(betas,dtype=torch.float32).to(self.device)
        factors_tensor=torch.tensor(factors,dtype=torch.float32).to(self.device)    
        labels_tensor=torch.tensor(labels,dtype=torch.float32).to(self.device)
        dataset=TensorDataset(betas_tensor,factors_tensor,labels_tensor)
        
        return DataLoader(dataset,batch_size=1,shuffle=True)
        
        
    def train_one_epoch(self):
        
        self.train_loader=self._get_dataset(period='train')
        epoch_loss=0
        
        for i,(beta_nn_input,factor_nn_input,label) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            beta_nn_input=beta_nn_input.squeeze(0).T
            factor_nn_input=factor_nn_input.squeeze(0).T
            label=label.squeeze(0)
            output = self.forward(beta_nn_input, factor_nn_input)
            loss = self.criterion(output, label)
            
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()

            if i % 100 == 0:
                # print(f'Batches: {i}, loss: {loss.item()}')
                pass

        return epoch_loss / len(self.train_loader)
    
    def _valid_one_epoch(self):
        epoch_loss = 0.0
        for i, (beta_nn_input, factor_nn_input, labels) in enumerate(self.valid_loader):
            # beta_nn_input reshape: (1, 94, 94) -> (94, 94) (1*P*N => N*P)
            # factor_nn_input reshape: (1, 94, 1) -> (1, 94) (1*P*1 => 1*P)
            # labels reshape: (1, 94) -> (94, ) (1*N => N,)
            beta_nn_input = beta_nn_input.squeeze(0).T
            factor_nn_input = factor_nn_input.squeeze(0).T
            labels = labels.squeeze(0)

            output = self.forward(beta_nn_input, factor_nn_input)
            loss = self.criterion(output, labels)
            epoch_loss += loss.item()

        return epoch_loss / len(self.valid_loader)
    
    def train_model(self):
        if 'saved_models' not in os.listdir('./'):
            os.mkdir('saved_models')
        
        self.train_loader=self._get_dataset(period='train')
        self.valid_loader=self._get_dataset(period='valid')
        
        
        min_error=np.inf
        train_loss=[]
        valid_loss=[]
        for epoch in tqdm(range(MAX_EPOCH)):
            self.train()
            trainloss=self.train_one_epoch()
            train_loss.append(trainloss)
            
            self.eval()
            with torch.no_grad():
                validloss=self._valid_one_epoch()    
                valid_loss.append(validloss)
            
            if validloss < min_error:
                min_error = validloss
                no_update_steps = 0
                # save model
                torch.save(self.state_dict(), f'./saved_models/{self.name}.pt')
            else:
                no_update_steps += 1
                
            if no_update_steps > 2: # early stop, if consecutive 3 epoches no improvement on validation set
                print(f'Early stop at epoch {epoch}')
                break
            # load from (best) saved model
            self.load_state_dict(torch.load(f'./saved_models/{self.name}.pt'))
        return train_loss, valid_loss
    
    def test_model(self):
        # beta, factor, label = self.test_dataset
        # i = np.random.randint(len(beta))
        # beta_nn_input = beta[i]
        # factor_nn_input = factor[i]
        # labels = label[i]
        self.test_loader=self._get_dataset(period='test')
        output = None
        label = None
        for i, beta_nn_input, factor_nn_input, labels in enumerate(self.test_loader):
            # convert to tensor
            # beta_nn_input = torch.tensor(beta_nn_input, dtype=torch.float32).T.to(self.device)
            # factor_nn_input = torch.tensor(factor_nn_input, dtype=torch.float32).T.to(self.device)
            # labels = torch.tensor(labels, dtype=torch.float32).T.to(self.device)
            output = self.forward(beta_nn_input, factor_nn_input)
            break

        loss = self.criterion(output, labels)
        print(f'Test loss: {loss.item()}')
        print(f'Predicted: {output}')
        print(f'Ground truth: {labels}')
        return output, labels
    
    def calBeta(self,month):
        
        _,beta,_,_=self._get_item(month)
        beta_tensor=torch.tensor(beta,dtype=torch.float32).T.to(self.device)
        if self.name[0]=='V':
            beta_encode=self.beta_encoder(beta_tensor)
            mu=self.mu(beta_encode)
            logvar=self.logvar(beta_encode)
            beta_reparam=self.reparameter(mu,logvar)
            beta=self.beta_decoder(beta_reparam)
        else:
            beta=self.ca_beta(beta_tensor)
        return beta
    
    def calFactor(self,month):
        _,_,factor,_=self._get_item(month)    
        factor_tensor=torch.tensor(factor,dtype=torch.float32).T.to(self.device)
        factor=self.ca_factor(factor_tensor).T
        self.factor_nn_pred.append(factor)
        return factor
    
    
    def _get_item(self,month):
        betas=pd.read_pickle('./data/p_charas.pkl')
        factors=pd.read_pickle('./data/portfolio_rets.pkl')
        labels=pd.read_pickle('./data/portfolio_rets.pkl')
        beta_mon=betas.loc[betas['DATE']==month][CHARAS_LIST]
        factor_mon=factors.loc[factors['DATE']==month][CHARAS_LIST].T.values
        label_mon=labels.loc[labels['DATE']==month][CHARAS_LIST].T.values
        return beta_mon.index,beta_mon.T.values,factor_mon,label_mon
    
    def cal_delayed_Factor(self,month):
        if self.refit_cnt==0:
            delayed_factor=self.factor_nn_pred[0]
        else:
            delayed_factor=torch.mean(torch.stack(self.factor_nn_pred[:self.refit_cnt]),dim=0)
        return delayed_factor
    
    def reset_parameters(self):
        def _reset(m):
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()

        if self.name.startswith("VAE_wzy"):
            # feature extractor
            self.feature_extractor.apply(_reset)

            # posterior network
            for layer in (self.post_mu, self.post_sigma):
                if hasattr(layer, "reset_parameters"):
                    layer.reset_parameters()

            self.decoder.alpha.apply(_reset)
            self.decoder.beta.reset_parameters()

            # prior network
            self.prior.alpha.apply(_reset)

        elif self.name[0]=='V':
            for layer in self.beta_encoder:
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
            for layer in self.beta_decoder:
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
            for layer in self.beta_decoder:
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
            for layer in self.ca_factor:
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
        else:
            for layer in self.ca_beta:
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
            for layer in self.ca_factor:
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
        self.optimizer.state=collections.defaultdict(dict)
    def release_gpu(self):
        if self.train_loader is not None:
            del self.train_loader
        if self.valid_loader is not None:
            del self.valid_loader
        if self.test_loader is not None:
            del self.test_loader
        torch.cuda.empty_cache()

class CA0_wyh(CA_base_wyh):
    def __init__(self,hidden_size,dropout_rate=0.1,lr=0.001,omit_char=[],device='cuda'):
        CA_base_wyh.__init__(self,name=f'CA0_wyh_{hidden_size}',omit_char=omit_char,device=device)
        self.ca_beta=nn.Sequential(
            nn.Linear(94,hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        self.ca_factor=nn.Sequential(
            nn.Linear(94,hidden_size)
        )
        
        self.optimizer=torch.optim.Adam(self.parameters(),lr=lr)
        self.criterion=nn.MSELoss().to(device)
        
class CA1_wyh(CA_base_wyh):
    def __init__(self,hidden_size,dropout_rate=0.1,lr=0.001,omit_char=[],device='cuda'):
        CA_base_wyh.__init__(self,name=f'CA1_wyh_{hidden_size}',omit_char=omit_char,device=device)
        self.ca_beta=nn.Sequential(
            nn.Linear(94,32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32,hidden_size),
        )
        self.ca_factor=nn.Sequential(
            nn.Linear(94,hidden_size)
        )
        self.optimizer=torch.optim.Adam(self.parameters(),lr=lr)
        self.criterion=nn.MSELoss().to(device)
        
class CA2_wyh(CA_base_wyh):
    def __init__(self,hidden_size,dropout_rate=0.1,lr=0.001,omit_char=[],device='cuda'):
        CA_base_wyh.__init__(self,name=f'CA2_wyh_{hidden_size}',omit_char=omit_char,device=device)
        self.ca_beta=nn.Sequential(
            nn.Linear(94,32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32,16),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(16,hidden_size),
        )
        self.ca_factor=nn.Sequential(
            nn.Linear(94,hidden_size)
        )
        self.optimizer=torch.optim.Adam(self.parameters(),lr=lr)
        self.criterion=nn.MSELoss().to(device)
        
class CA3_wyh(CA_base_wyh):
    def __init__(self,hidden_size,dropout_rate=0.1,lr=0.001,omit_char=[],device='cuda'):
        CA_base_wyh.__init__(self,name=f'CA3_wyh_{hidden_size}',omit_char=omit_char,device=device)
        self.ca_beta=nn.Sequential(
            nn.Linear(94,32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32,16),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(16,8),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(8,hidden_size),
        )
        self.ca_factor=nn.Sequential(
            nn.Linear(94,hidden_size)
        )
        self.optimizer=torch.optim.Adam(self.parameters(),lr=lr)
        self.criterion=nn.MSELoss().to(device)
        
        
# class VAE_base(CA_base_wyh):
#     def __init__(self,hidden_size,dropout_rate=0.1,lr=0.001,omit_char=[],device='cuda',hidden_dims=[32,16,8]):
#         nn.Module.__init__(self)
#         CA_base_wyh.__init__(self,name=f'VAE_{hidden_size}',omit_char=omit_char,device=device)
#         # self.vae=VAE_wyh(hidden_size,dropout_rate,lr,device,hidden_dims)
#         self.beta_decoder=None
#         self.beta_encoder=None
#         self.mu=None
#         self.logvar=None
#         self.ca_factor=None
        
#     def forward(self,beta_input,factor_input):
#         beta_reparam,beta_output,factor_output,mu,logvar=self.vae(beta_input,factor_input)
#         output=torch.mm(beta_output,factor_output.T)
#         return output,mu,logvar
    

    
#     def train_one_epoch(self):
#         self.train_loader=self._get_dataset(period='train')
#         epoch_loss=0
#         for i,(beta_nn_input,factor_nn_input,label) in enumerate(self.train_loader):
#             self.optimizer.zero_grad()
#             beta_nn_input=beta_nn_input.squeeze(0).T
#             factor_nn_input=factor_nn_input.squeeze(0).T
#             label=label.squeeze(0)
#             output,mu,logvar= self.forward(beta_nn_input, factor_nn_input)
#             print(self.criterion)
#             loss = self.criterion(factor_nn_input, output,mu,logvar)
            
#             loss.backward()
#             self.optimizer.step()
#             epoch_loss += loss.item()

#             if i % 100 == 0:
#                 # print(f'Batches: {i}, loss: {loss.item()}')
#                 pass

#         return epoch_loss / len(self.train_loader)
    
#     def _valid_one_epoch(self):
#         epoch_loss = 0.0
#         for i, (beta_nn_input, factor_nn_input, labels) in enumerate(self.valid_loader):
#             # beta_nn_input reshape: (1, 94, 94) -> (94, 94) (1*P*N => N*P)
#             # factor_nn_input reshape: (1, 94, 1) -> (1, 94) (1*P*1 => 1*P)
#             # labels reshape: (1, 94) -> (94, ) (1*N => N,)
#             beta_nn_input = beta_nn_input.squeeze(0).T
#             factor_nn_input = factor_nn_input.squeeze(0).T
#             labels = labels.squeeze(0)

#             output,mu,logvar = self.forward(beta_nn_input, factor_nn_input)
#             loss = self.criterion(factor_nn_input, output,mu,logvar)
#             epoch_loss += loss.item()

#         return epoch_loss / len(self.valid_loader)

class VAE_wyh(CA_base_wyh):
    def __init__(self,hidden_size,dropout_rate=0.1,lr=0.001,device='cuda',omit_char=[],hidden_dims=[32,16,8]):
        CA_base_wyh.__init__(self,name=f'VAE_{hidden_size}',omit_char=omit_char,device=device)
        self.beta_encoder=nn.Sequential(
            nn.Linear(94,hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32,hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(16,hidden_dims[2]),
        )
        self.mu=nn.Linear(hidden_dims[2],hidden_dims[2])
        self.logvar=nn.Linear(hidden_dims[2],hidden_dims[2])
        self.beta_decoder=nn.Sequential(
            nn.Linear(hidden_dims[2],hidden_size),
        )
        
        self.ca_factor=nn.Sequential(
            nn.Linear(94,hidden_size)
        )
        self.optimizer=torch.optim.Adam(self.parameters(),lr=lr)
        self.criterion=self.criterion_
        self.reparameter=self.reparameter_
    def reparameter_(self,mu,logvar):
        std=torch.exp(0.5*logvar)
        eps=torch.randn_like(std)
        return mu+eps*std
    
    def forward(self,beta_input,factor_input):
        beta_latent=self.beta_encoder(beta_input)
        mu=self.mu(beta_latent)
        logvar=self.logvar(beta_latent)
        beta_reparam=self.reparameter(mu,logvar)
        beta_output=self.beta_decoder(beta_reparam)
        factor_output=self.ca_factor(factor_input)
        output=torch.mm(beta_output,factor_output.T)
        return output,mu,logvar,beta_output
        # return beta_reparam,beta_output,factor_output,mu,logvar
    
    def criterion_(self,factor_input,output,mu,logvar):
        recon_loss=F.mse_loss(factor_input,output,reduction='sum')
        kl_loss=-0.5*torch.sum(1+logvar-mu.pow(2)-logvar.exp())
        return recon_loss+kl_loss
    
    def train_one_epoch(self):
        self.train_loader=self._get_dataset(period='train')
        epoch_loss=0
        for i,(beta_nn_input,factor_nn_input,label) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            beta_nn_input=beta_nn_input.squeeze(0).T
            factor_nn_input=factor_nn_input.squeeze(0).T
            label=label.squeeze(0)
            output,mu,logvar,beta_output= self.forward(beta_nn_input, factor_nn_input)
            # print(self.criterion)
            loss = self.criterion(factor_nn_input, output,mu,logvar)
            
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()

            if i % 100 == 0:
                # print(f'Batches: {i}, loss: {loss.item()}')
                pass

        return epoch_loss / len(self.train_loader)
    
    def _valid_one_epoch(self):
        epoch_loss = 0.0
        for i, (beta_nn_input, factor_nn_input, labels) in enumerate(self.valid_loader):
            # beta_nn_input reshape: (1, 94, 94) -> (94, 94) (1*P*N => N*P)
            # factor_nn_input reshape: (1, 94, 1) -> (1, 94) (1*P*1 => 1*P)
            # labels reshape: (1, 94) -> (94, ) (1*N => N,)
            beta_nn_input = beta_nn_input.squeeze(0).T
            factor_nn_input = factor_nn_input.squeeze(0).T
            labels = labels.squeeze(0)

            output,mu,logvar,beta_output = self.forward(beta_nn_input, factor_nn_input)
            loss = self.criterion(factor_nn_input, output,mu,logvar)
            epoch_loss += loss.item()

        return epoch_loss / len(self.valid_loader)

class Alpha(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, hidden_dim)
        self.mu     = nn.Linear(hidden_dim, out_dim)
        self.sigma  = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        h      = F.leaky_relu(self.linear(x), 0.2)
        mu     = self.mu(h)
        sigma  = F.softplus(self.sigma(h))
        return mu, sigma

class FeatureExtractor(nn.Module):
    def __init__(self, in_dim=94, d_model=32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.LeakyReLU(0.2),
            nn.Linear(d_model, d_model),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.mlp(x)
    
class FeatureDecoder(nn.Module):
    def __init__(self, d_model: int, z_dim: int):
        super().__init__()
        self.alpha = Alpha(d_model, z_dim, 1)
        self.beta  = nn.Linear(d_model, z_dim)

    def forward(self, e, z_mu, z_sigma):
        a_mu, a_sigma = self.alpha(e)
        beta_load     = self.beta(e)

        if z_mu.dim() == 1:
            z_mu     = z_mu.unsqueeze(0)
            z_sigma  = z_sigma.unsqueeze(0)

        y_mu = a_mu.squeeze(-1) + (beta_load * z_mu).sum(dim=1)
        y_var = a_sigma.squeeze(-1).pow(2) + (beta_load.pow(2) * z_sigma.pow(2)).sum(dim=1)
        y_sigma = torch.sqrt(y_var + 1e-8)

        return y_mu, y_sigma
    
class LatentPrior(nn.Module):
    def __init__(self, d_model: int, z_dim: int):
        super().__init__()
        self.alpha = Alpha(d_model, z_dim, z_dim)

    def forward(self, e):
        h = e.mean(dim=0)
        return self.alpha(h)

class VAE_wzy(CA_base_wyh):

    def __init__(self,hidden_size,dropout_rate=0.1,lr=0.001,device='cuda',omit_char=[],
                 beta_in_dim=94, d_model=32):
        self.gamma = 1.0  

        CA_base_wyh.__init__(self,name=f'VAE_wzy_{hidden_size}',omit_char=omit_char,device=device)
        self.feature_extractor   = FeatureExtractor(beta_in_dim, d_model)

        self.post_mu    = nn.Linear(1, hidden_size)
        self.post_sigma = nn.Linear(1, hidden_size)

        self.decoder = FeatureDecoder(d_model, hidden_size)

        self.prior   = LatentPrior(d_model, hidden_size)
        self.optimizer=torch.optim.Adam(self.parameters(),lr=lr)

    def forward(self, beta_input, factor_input, test: bool = False):
        e = self.feature_extractor(beta_input)

        y_flat     = factor_input.view(-1, 1)               # (N,1)
        mu_post = self.post_mu(y_flat).squeeze(-1)   # (N,K)
        sigma_post = F.softplus(self.post_sigma(y_flat).squeeze(-1))

        y_mu, y_sigma = self.decoder(e, mu_post, sigma_post)

        z_prior_mu, z_prior_sigma = self.prior(e)
        y_pred_mu, y_pred_sigma   = self.decoder(e, z_prior_mu, z_prior_sigma)

        if test:
            return y_pred_mu, y_pred_sigma

        return (y_mu, y_sigma,
                mu_post, sigma_post,
                z_prior_mu, z_prior_sigma)
    
    def gaussian_nll(self, y, mu, sigma, eps=1e-8):
        sigma2 = sigma.pow(2) + eps
        nll = 0.5 * (torch.log(2 * torch.pi * sigma2) + (y - mu).pow(2) / sigma2)
        return nll.mean()
    
    def kl_div_diag_gaussians(self, mu_q, sigma_q, mu_p, sigma_p, eps=1e-8):
        var_q, var_p = sigma_q.pow(2) + eps, sigma_p.pow(2) + eps
        kl = 0.5 * ((var_q + (mu_q - mu_p).pow(2)) / var_p + torch.log(var_p) - torch.log(var_q) - 1)
        return kl.mean()

    def vae_loss(self, y_true, y_mu, y_sigma, mu_post, sigma_post, mu_prior, sigma_prior):
        nll = self.gaussian_nll(y_true, y_mu, y_sigma)
        kl  = self.kl_div_diag_gaussians(mu_post, sigma_post,
                                    mu_prior, sigma_prior)
        return nll + self.gamma * kl, nll.detach(), kl.detach()
    
    def train_one_epoch(self):
        self.train_loader = self._get_dataset(period='train')
        self.train()
        epoch_loss, epoch_nll, epoch_kl = 0, 0, 0

        for beta_in, factor_in, _ in self.train_loader:
            beta_in   = beta_in.squeeze(0).T.to(self.device)       # (N, 94)
            factor_in = factor_in.squeeze(0).T.to(self.device)     # (N,)

            (y_mu, y_sigma,
            mu_post, sigma_post,
            mu_prior, sigma_prior) = self.forward(beta_in, factor_in, test=False)

            loss, nll, kl = self.vae_loss(factor_in,
                                        y_mu, y_sigma,
                                        mu_post, sigma_post,
                                        mu_prior, sigma_prior)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()
            epoch_nll  += nll.item()
            epoch_kl   += kl.item()

        n = len(self.train_loader)
        return epoch_loss / n, epoch_nll / n, epoch_kl / n

    def _valid_one_epoch(self):
        self.eval()
        epoch_loss, epoch_nll, epoch_kl = 0, 0, 0
        with torch.no_grad():
            for beta_in, factor_in, _ in self.valid_loader:
                beta_in   = beta_in.squeeze(0).T.to(self.device)
                factor_in = factor_in.squeeze(0).T.to(self.device)

                (y_mu, y_sigma,
                mu_post, sigma_post,
                mu_prior, sigma_prior) = self.forward(beta_in, factor_in, test=False)

                loss, nll, kl = self.vae_loss(factor_in,
                                            y_mu, y_sigma,
                                            mu_post, sigma_post,
                                            mu_prior, sigma_prior)

                epoch_loss += loss.item()
                epoch_nll  += nll.item()
                epoch_kl   += kl.item()

        n = len(self.valid_loader)
        return epoch_loss / n, epoch_nll / n, epoch_kl / n
    
    def train_model(self, max_epoch=200, patience=3):
        if 'saved_models' not in os.listdir('./'):
            os.mkdir('saved_models')
        
        self.train_loader=self._get_dataset(period='train')
        self.valid_loader=self._get_dataset(period='valid')
        
        train_loss_list=[]
        valid_loss_list=[]

        no_update_steps, best_val = 0, float('inf')
        for epoch in range(1, max_epoch+1):
            train_loss, train_nll, train_kl = self.train_one_epoch()
            val_loss,   val_nll,   val_kl   = self._valid_one_epoch()

            train_loss_list.append(train_loss)
            valid_loss_list.append(val_loss)

            # print(f'Epoch {epoch:03d} | '
            #     f'Train L={train_loss:.4f} (NLL={train_nll:.4f}, KL={train_kl:.4f}) | '
            #     f'Val L={val_loss:.4f}')

            if val_loss < best_val:
                best_val = val_loss
                no_update_steps = 0
                torch.save(self.state_dict(), f'./saved_models/{self.name}.pt')
            else:
                no_update_steps += 1
                if no_update_steps >= patience:
                    print(f'Early stop at epoch {epoch}')
                    break

            self.load_state_dict(torch.load(f'./saved_models/{self.name}.pt'))
        return train_loss_list, valid_loss_list

    @torch.no_grad()
    def test_model(self):
        self.test_loader = self._get_dataset(period='test')
        self.eval()
        for beta_in, factor_in, _ in self.test_loader:
            beta_in   = beta_in.squeeze(0).T.to(self.device)
            factor_in = factor_in.squeeze(0).T.to(self.device)
            y_mu, y_sigma = self.forward(beta_in, factor_in, test=True)
            nll = self.gaussian_nll(factor_in, y_mu, y_sigma)
            print(f'Test NLL: {nll.item():.4f}')
            break

    def inference(self, month):       
        assert month >= self.test_period[0], f"Month error, {month} is not in test period {self.test_period}"
        
        _,beta,factor,_=self._get_item(month)
        beta_tensor=torch.tensor(beta,dtype=torch.float32).T.to(self.device)
        factor_tensor=torch.tensor(factor,dtype=torch.float32).T.to(self.device)

        (y_mu, y_sigma, mu_post, _, _, _) = self.forward(beta_tensor, factor_tensor)
        ret = y_mu

        self.factor_nn_pred.append(mu_post.detach())

        return ret.unsqueeze(1)
        
    
    def predict(self, month):
        assert month >= self.test_period[0] and month <= self.test_period[1], f"Month error, {month} is not in test period {self.test_period}"
        
        _,beta,_,_=self._get_item(month)
        beta_tensor=torch.tensor(beta,dtype=torch.float32).T.to(self.device)

        e         = self.feature_extractor(beta_tensor)                   # (N,d_model)
        alpha_mu, alpha_sigma = self.decoder.alpha(e)                # (N,1),(N,1)
        beta_load = self.decoder.beta(e)                             # (N,K)

        if hasattr(self, "factor_nn_pred") and len(self.factor_nn_pred) > 0:
            past_mu = torch.stack(self.factor_nn_pred, dim=0)  # (T,K)
            f_delay = past_mu.mean(dim=0)                                   # (K,)
            z_sigma = torch.zeros_like(f_delay)
        else:
            f_delay, z_sigma = self.prior(e)

        y_mu = alpha_mu.squeeze(-1) + (beta_load * f_delay).sum(dim=1)       # (N,)
        ret = y_mu
        
        return ret.unsqueeze(1)
        
if __name__=='__main__':
    model=CA_base_wyh()
    model.train_one_epoch()
    