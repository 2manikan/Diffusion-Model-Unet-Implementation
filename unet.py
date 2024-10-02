#
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import torch
from torch import nn
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.optim as optim
import gc
import torch.optim as optim
from torch.optim import Adam
import math

def ts_embed(timesteps, dim):
    assert len(timesteps.size())==1, "wrong dimension"
    half_dim=dim//2
    exponent=-math.log(10000) * torch.arange(start=0, end=half_dim, dtype=torch.float32).to("cuda")
    exponent=exponent/(half_dim - 0)
    emb=torch.exp(exponent)
    emb=timesteps[:, None].float() * emb[None, :]
    emb=emb*1
    emb=torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    

    #padding with zeros
    if dim%2==1:
       emb=torch.nn.functional.pad(emb, (0,1,0,0))

    return emb		  


class RBlock(nn.Module):
  def __init__(self, in_channels, out_channels, up=False, down=False, time_dim=None, num_groups=4):
    super().__init__()
    self.first_norm=nn.GroupNorm(num_groups=num_groups, num_channels=in_channels)
    self.nonlinear=nn.SiLU()

    if down:
      self.conv_block=nn.AvgPool2d(kernel_size=2,stride=2)
    self.down=down
    self.up=up

    self.first_conv_layer=nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
    self.second_conv_layer=nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

    self.time_layer=nn.Linear(time_dim, out_channels)
    self.second_norm=nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)

    self.use_shortcut=in_channels != out_channels #enough for this impl.
    self.channel_change=None
    if self.use_shortcut==True:
       self.channel_change=nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=1,padding=0,bias=True) 

  def forward(self, image, ts):
    feature_map=self.nonlinear(self.first_norm(image)) #make sure image object is intact 

    feature_map=self.first_conv_layer(feature_map)
    ts_new=self.time_layer(ts); ts_new=ts_new[:,:,None,None]
    feature_map+=ts_new
    feature_map=self.second_norm(feature_map)


    feature_map=self.nonlinear(feature_map)
    feature_map=self.second_conv_layer(feature_map)

    #added rn by me!!!!!!!!!!!!!
    if self.channel_change!=None:
       image=self.channel_change(image)

    return (image+feature_map)


class UnetModel(nn.Module):
  def __init__(self):
    super().__init__()
    num_groups=4
    t_dim=16*4

    #timestep stuff
    self.ts_layers=nn.Sequential(
        nn.Linear(16,16*4,True), #bc 16 is the "out_layer" as shown below. sample_proj_bias is true.
        nn.SiLU(),
        nn.Linear(16*4,16*4,True)
    )

    self.start=nn.Conv2d(3,16,kernel_size=3, stride=1, padding=1)

    self.down_blocks=nn.ModuleList([])
    
    self.channel_numbers=[16, 32, 64, 128, 256, 512] #can change this
    out_channels=self.channel_numbers[0]
  
    for i in range(0,6): #down_layers
      in_channels=out_channels
      out_channels=self.channel_numbers[i]
      final_block = i ==len(self.channel_numbers)-1
      for j in range(2): #each layer
        in_channels2=in_channels if j==0 else out_channels
        self.down_blocks.append(
            RBlock(in_channels2,out_channels,False,False,t_dim,num_groups)
        )
        
      if not final_block:
        self.down_blocks.append(
          #nn.AvgPool2d(kernel_size=2,stride=2) #if conv=False
	  nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=2,padding=1,bias=True) #as conv=True
        )

    #MIDDLE BLOCK
    out_channels=self.channel_numbers[-1]
    self.middle_blocks=nn.ModuleList([
            RBlock(out_channels,out_channels,False,False,t_dim,num_groups),
            RBlock(out_channels,out_channels,False,False,t_dim,num_groups)
                                    ]
                                     )

    reverse_channels=list(reversed(self.channel_numbers))
    out_channels=reverse_channels[0]
    self.up_blocks=nn.ModuleList([])
    for i in range(0,6):
      prev_out_channels=out_channels
      out_channels=reverse_channels[i]
      in_channels=reverse_channels[min(i+1,len(reverse_channels)-1)]
      final_block = i ==len(reverse_channels)-1

          
      
      for j in range(3):
        skip_channels=in_channels if (j==3-1) else out_channels
        res_in_channels = prev_out_channels if j==0 else out_channels
        
        self.up_blocks.append(
              RBlock(res_in_channels+skip_channels,out_channels,False,False,t_dim,num_groups)
          )
      if not final_block:
         self.up_blocks.append(nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1,bias=True))
        
      prev_out_channels=out_channels
      #upsample block will be in forward function.
      
      

    self.end_process=nn.Sequential(
        nn.GroupNorm(num_groups=num_groups, num_channels=16, eps=1e-5), #make norm_eps=1e-5 
        nn.SiLU(),
        nn.Conv2d(16,3,kernel_size=3,padding=1)
    )

  def forward(self, x, t):
   
    #ts_embeds
    t=ts_embed(t, 16) #check dimension used
    t=self.ts_layers(t)

    #starting
    x=self.start(x)

    
    #downsample

    sk_c=[x.view(x.size())]
    for i in range(0,len(self.down_blocks)):
      if i%3!=2:
        x=self.down_blocks[i](x, t)
        #sk_c.append(x)
      else:
        x=self.down_blocks[i](x)
      sk_c.append(x)
    
    #middle
    #for i in range(len(self.middle_blocks)):
    x=self.middle_blocks[0](x,t)
    x=self.middle_blocks[1](x,t)
    
    #upsample
    #count=0
    #i=0
    #while(count<len(self.up_blocks)): #+1 bc the block just ends at the last RESNET upsampling. We also want to upsample to get to the correct image size.    
    #  if i%4!=3:
    #    x=self.up_blocks[count](torch.cat([x,sk_c.pop()],dim=1), t)
    #    count+=1
    #  else:
    #    #if count!=len(self.up_blocks)-1:
    #    x=nn.functional.interpolate(x,scale_factor=2,mode='nearest')   
    #  i+=1

    for i in range(0,len(self.up_blocks)):
        if i%4!=3:
           x=self.up_blocks[i](torch.cat([x,sk_c.pop()],dim=1), t)
        else:
           x=nn.functional.interpolate(x,scale_factor=2,mode='nearest')
           x=self.up_blocks[i](x)
                   

    x=self.end_process(x)

    return x
