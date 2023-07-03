import modules.scripts as scripts
import gradio as gr
import os
import torch
import numpy as np

from modules import images, devices, script_callbacks
import modules.processing as processing
from modules.shared import opts, cmd_opts, state
import modules.shared as shared

enabled = False
normalize = False
normalize_weights = [0,0]

color = [255,255,255]
strength = 1.0

def normalize_noise(noise, weight):
        print("Normalize: " + str(normalize))
        #per channel normalization
        for ix, channel in enumerate(noise):
                #we noticed a big shift in the i2i dynamic range, so we are normalizing each channel separately to fix this
                #all channels should be normally distributed, so we can use the mean and std to normalize
                print("Channel: " + str(ix))
                print("Mean: " + str(torch.mean(channel)))
                print("Std: " + str(torch.std(channel)))
                channel_norm = (channel - torch.mean(channel)) / torch.std(channel)
                print("Final Mean: " + str(torch.mean(channel_norm)))
                print("Final Std: " + str(torch.std(channel_norm)))
                #mix in the normalized channel with the original channel
                noise[ix] = (channel_norm * weight) + (channel * (1-weight))
        return noise

def create_random_tensors(shape, seeds, subseeds=None, subseed_strength=0.0, seed_resize_from_h=0, seed_resize_from_w=0, p=None):
        eta_noise_seed_delta = opts.eta_noise_seed_delta or 0
        xs = []
        if p is not None and p.sampler is not None and (len(seeds) > 1 and opts.enable_batch_seeds or eta_noise_seed_delta > 0):
                sampler_noises = [[] for _ in range(p.sampler.number_of_needed_noises(p))]
        else:
                sampler_noises = None

        for i, seed in enumerate(seeds):
                noise_shape = shape if seed_resize_from_h <= 0 or seed_resize_from_w <= 0 else (shape[0], seed_resize_from_h//8, seed_resize_from_w//8)

        subnoise = None
        if subseeds is not None:
                subseed = 0 if i >= len(subseeds) else subseeds[i]

                subnoise = devices.randn(subseed, noise_shape)

        noise = devices.randn(seed, noise_shape)
        global enabled
        global normalize
        global normalize_weights
        global color

        #Normalize before:
        if normalize == True:
                noise = normalize_noise(noise, normalize_weights[0])
        print("Enabled: " + str(enabled))
        if enabled == True:
                
                #R - [+0.9,                -0.9,             -0.9],  #L2
                #G - [-0.9,                +0.9,             -0.9],  #L2 
                #B - [-0.9,                -0.9,             +0.9],  #L2 
                
                #C - [-0.9,                +0.9,             +0.9],  #L2  
                #M - [+0.9,                -0.9,             +0.9],  #L2
                #Y - [+0.9,                +0.9,             -0.9],  #L2
                
                #Half-Strength Colors
                #Purple - [0,                -0.9,             +0.9],  #L2 (128,0,255)
                #Lime   - [0,                +0.9,             -0.9],  #L2 (0,255,128)

                #Red-Yellow - [+0.9,                0,             -0.9],  #L2 (255,128,0) (ORANGE)
                #Blue-Green - [-0.9,                0,             +0.9],  #L2 (0,128,255)

                #Green-Blue - [-0.9,                +0.9,             0],  #L2 (0,255,128)
                #Red-Purple - [+0.9,                -0.9,             0],  #L2 (255,0,128)


                


                #Orange is composed of Red and Yellow
                #Orange - [-0.9,                0,             +0.9],  #L2
                
                #Rl, Gl, and Bl will be functions
                #so the incoming L2 row would be picked off and go thru the function and then the coeffs could be einsummed, I guess
                #Rl = (-0.007 * R + 0.9226)
                #Gl = (-0.007 * G + 0.9226)
                #Bl = (-0.007 * B + 0.9226)
                l = 0.18215; c = 0.28; s = 0.17
                R = color[0]
                G = color[1]
                B = color[2]
                vR = R / 255
                vG = G / 255
                vB = B / 255
                Rl = 0.9 * (R/255) - 0.9 * (1-(R/255))
                Gl = 0.9 * (G/255) - 0.9 * (1-(G/255))
                Bl = 0.9 * (B/255) - 0.9 * (1-(B/255))

                #calculate how much  

                #calculate the luminance
                #lum = (0.2126 *vR) + (0.7152 * vG) + (0.0722 * vB)
                lum = (R + G + B) / (3 * 255)
                #scale to between -1 and 1
                lum = (lum * 2) - 1
                lum = lum * 0.6
                print("Lum: " + str(lum))
                #scale lum to between 0 and 1
                #lum = lum / 255

                #calculate the Rl, Gl, and Bl values including the luminance

                


                print("Rl: " + str(Rl))
                print("Gl: " + str(Gl))
                print("Bl: " + str(Bl))
                # R  G  B
                #[ +, -, -], #L0 - Cyan <-> Red
                #[ +, +, -], #L1 - Blue <-> Yellow
                #[ -, +, +], #L2 - Red <-> Cyan
                #[ -, +, -], #L3 - Magenta <-> Green
                
                #l0 = (G + B) to R
                #l1 = B to (R + G)
                #l2 = R to (G + B)
                #l3 = G to (R + B)
                coefs = torch.tensor([
                # +/-     R                    G                 B
                        [  +l*Rl+lum,                  -l*Gl+lum,               -l*Bl+lum],  #L0 
                        [  +c*Rl+lum,                  +c*Gl+lum,               -c*Bl+lum],  #L1
                        [  -Rl,                    +Gl,                   +Bl],  #L2
                        [  -s*Rl,                  +s*Gl,               -s*Bl],  #L3
                        #[  s,                  -s,               s],  #L3
                        #[  -Rl,                  +Gl,               -Bl],  #L3
                ]).to(noise.device)
                """ coefs = torch.tensor([
                # +/-     R                    G                 B
                        [  -l,                  +l,               +l],  #L0 
                        [  -c,                  -c,               +c],  #L1 
                        [  +Rl,                -Gl,             -Bl],  #L2
                        [  +s,                  -s,               +s]   #L3 
                ]).to(noise.device) """
                offsets = [0, 0, 0, 0]
                #coefs = coefs * strength
                offsets[0] = sum(coefs[0]*strength)
                offsets[1] = sum(coefs[1]*strength)
                offsets[2] = sum(coefs[2]*strength)
                offsets[3] = sum(coefs[3]*strength)
                print(coefs)
                print(offsets)
                offsets = torch.tensor(offsets).to(noise.device)
                #apply each layer of offsets to each of the respective layers of noise
                for i in range(0,4):
                        noise[i] = noise[i] + offsets[i]
                
                
        #Normalize after:
        if normalize == True:
                noise = normalize_noise(noise, normalize_weights[1])

        if subnoise is not None:
            noise = processing.slerp(subseed_strength, noise, subnoise)

        if noise_shape != shape:
            x = devices.randn(seed, shape)
            dx = (shape[2] - noise_shape[2]) // 2
            dy = (shape[1] - noise_shape[1]) // 2
            w = noise_shape[2] if dx >= 0 else noise_shape[2] + 2 * dx
            h = noise_shape[1] if dy >= 0 else noise_shape[1] + 2 * dy
            tx = 0 if dx < 0 else dx
            ty = 0 if dy < 0 else dy
            dx = max(-dx, 0)
            dy = max(-dy, 0)

            x[:, ty:ty+h, tx:tx+w] = noise[:, dy:dy+h, dx:dx+w]
            noise = x

        if sampler_noises is not None:
            cnt = p.sampler.number_of_needed_noises(p)

            if eta_noise_seed_delta > 0:
                torch.manual_seed(seed + eta_noise_seed_delta)

            for j in range(cnt):
                sampler_noises[j].append(devices.randn_without_seed(tuple(noise_shape)))

        xs.append(noise)

        if sampler_noises is not None:
                p.sampler.sampler_noises = [torch.stack(n).to(shared.device) for n in sampler_noises]

        x = torch.stack(xs).to(shared.device)
        return x

class ColorfulNoiseScript(scripts.Script):
        def title(self):
                return "Colorful Noise"
        def show(self, is_img2img):
                return scripts.AlwaysVisible
        def ui(self, is_img2img):
                with gr.Accordion('Colorful Noise', open=False):
                        #enable or disable the extension, checkbox
                        with gr.Row():
                                enabled_button = gr.Checkbox(label="Enable CN", value=False)
                                normalize_button = gr.Checkbox(label="Enable Normalization", value=False)
                                color_button = gr.ColorPicker(label="ColorSelector")
                        with gr.Row():
                                strength_slider = gr.Slider(-1, 2, step=0.1, label="Strength Slider",value=1)
                        with gr.Row():
                                normalize_before_slider = gr.Slider(0, 1, step=0.01, label="Normalize Before", value=1)
                                normalize_after_slider = gr.Slider(0, 1, step=0.01, label="Normalize After", value=0.9)
                return [enabled_button, normalize_button, color_button, strength_slider, normalize_before_slider, normalize_after_slider]
        
        def process(self, p, enabled_button, normalize_button, color_button, strength_slider, normalize_before_slider, normalize_after_slider):
                global color
                global enabled
                global strength
                global normalize
                global normalize_weights



                if normalize_button == True:
                        normalize = True
                        normalize_weights = [normalize_before_slider, normalize_after_slider]
                else:
                        normalize = False
                        normalize_weights = [0,0]


                if enabled_button == True:
                        enabled = True

                        strength = strength_slider
                        h = color_button.lstrip('#')
                        color = list(int(h[i:i+2], 16) for i in (0, 2, 4))
                        print(color)
                        
                        #need to override the create_random_tensors function in processing.py, but it's not a class, it's just a function
                        #doing this just causes a recursion depth error
                        #fix the recursion error and override the function
                        
                else:
                       enabled = False
                       strength = 0
                
                processing.create_random_tensors = create_random_tensors
                p.create_random_tensors = create_random_tensors
                #proc = process_images(p)
                #return proc


