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

color = [0.0,0.0,0.0]
strength = 1.0
luminance = 0.0

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

        # if we have multiple seeds, this means we are working with batch size>1; this then
        # enables the generation of additional tensors with noise that the sampler will use during its processing.
        # Using those pre-generated tensors instead of simple torch.randn allows a batch with seeds [100, 101] to
        # produce the same images as with two batches [100], [101].
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

        # randn results depend on device; gpu and cpu get different results for same seed;
        # the way I see it, it's better to do this on CPU, so that everyone gets same result;
        # but the original script had it like this, so I do not dare change it for now because
        # it will break everyone's seeds.
        noise = devices.randn(seed, noise_shape)
        global enabled
        global normalize
        global normalize_weights
        global color
        global luminance

        #Normalize before:
        if normalize == True:
                noise = normalize_noise(noise, normalize_weights[0])
        print("Enabled: " + str(enabled))
        if enabled == True:
                #the 0.18215 is the hard-coded magic number scale factor.
                
                #layer 0 is brightness
                #layer 1 is brightness
                #layer 2 is red-cyan
                #layer 3 is magenta-green

                #layer 0, 1, 2, 3
                #offsetlist = [0,0,0.2,0.2]
                #offsetlist = [0,0,0.18215,0.18215]
                #offsetlist = [0,0,0,0]
                #mult = 0.18215
                #mult = 0.3
                

                #implement inverse colour system, convert from a rgb list to format for offsets
                #color = [1,0,0] #red
                #color = [0,1,0] #green
                #color = [0,0,1] #blue

                #offsetlist[0] is brightness, so we don't want to change that much
                #offsetlist[1] is brightness with more contrast/black
                #offsetlist[2] is red-cyan
                #offsetlist[3] is magenta-green

                #offsetlist[2] - is red
                #offsetlist[2] + is cyan
                #offsetlist[3] - is magenta
                #offsetlist[3] + is green
                # R  G  B
                #[ +, -, -], #L0 - Red <-> Cyan (G + B)
                #[ +, +, -], #L1 - Yellow <-> Blue (R + G)
                #[ -, +, +], #L2 -
                #[ -, +, -], #L3 -  Green <-> Magenta (R + B)
                coefs = torch.tensor([
                        #     R         G          B
                        [ +0.26455683, +0.1218086,  +0.12589741], #L0 -
                        [ +0.11823522, +0.4185906,  +0.10553435], #L1 -   
                        
                        [ -0.2771753,  +0.2331822,  +0.18334152], #L2 - Red   <-> Cyan (G + B)        
                        [ -0.13701592, -0.1350247,  -0.72429520]  #L3 - Green <-> Magenta (R + B)
                ]).to(noise.device)
                #these conversion values are acting on the 4 channel latents themselves
                #providing a 'translation' from 4-channel latent space to 3-channel RGB space.
                #If we know the approximate conversion, there are a few things we can do. We can decode for live previews extremely fast and with little compute.
                #We could do the inverse and apply a custom color profile to all of the latents, or only specific latents.
                #Its another way of doing the offset, but with fine grain control, if desired.
                
                #multiply rgb color by the coefficient matrix's columns, this gives us the offsets for each layer
                #Multiply each column in coefs by each index in color
                #mult = 0.18215

                #luminance hack:
                coefs[0,:] = coefs[0,:] * luminance
                coefs[1,:] = coefs[1,:] * luminance

                coefs[:,0] = coefs[:,0] * color[0]
                coefs[:,1] = coefs[:,1] * color[1]
                coefs[:,2] = coefs[:,2] * color[2]
                #sum the columns to get the offset for each layer
                offsets = torch.sum(coefs, dim=1)
                #multiply the offsets by the strength
                offsets = offsets * strength

                #Qualitative notes:
                #======================================================
                #red at -1.0 luminance is DARK MAGENTA
                #red at -0.5 luminance is MAGENTA
                #red at 0 luminance is MAGENTA
                #red at 0.5 luminance is BRIGHT PINK
                #red at 1.0 luminance is WHITE

                #conclusion: red is too bright, needs less luminance
                #conclusion: red is too blue, needs less blue
                
                #======================================================
                #orange at -1.0 luminance is DARK BLUE
                #orange at -0.5 luminance is PURPLE
                #orange at 0 luminance is PINK
                #orange at 0.5 luminance is WHITE
                #orange at 1.0 luminance is BRIGHT WHITE

                #conclusion: orange is too bright, needs less luminance
                #conclusion: orange is too blue, needs less blue

                #======================================================
                #yellow at -1.0 luminance is DARK BLUE
                #yellow at -0.5 luminance is BLUE
                #yellow at 0 luminance is WHITE/PINK
                #yellow at 0.5 luminance is WHITE
                #yellow at 1.0 luminance is BRIGHT WHITE
                
                #conclusion: yellow is too bright, needs less luminance
                #conclusion: yellow is too blue, needs less blue
                
                #======================================================
                #green at -1.0 luminance is DARK BLUE
                #green at -0.5 luminance is BLUE
                #green at 0 luminance is CYAN
                #green at 0.5 luminance is WHITE/BRIGHT BLUE
                #green at 1.0 luminance is BRIGHT WHITE

                #conclusion: green is too bright, needs less luminance
                #conclusion: green is too blue, needs less blue

                #======================================================
                #blue at -1.0 luminance is LIGHT BLUE/CYAN
                #blue at -0.5 luminance is CYAN
                #blue at 0 luminance is LIGHT CYAN
                #blue at 0.5 luminance is VERY LIGHT CYAN
                #blue at 1.0 luminance is VERY LIGHT CYAN/WHITE

                #conclusion: blue is too bright, needs much less luminance
                #conclusion: blue is too green, needs less green

                #======================================================
                #indigo at -1.0 luminance is BLUE
                #indigo at -0.5 luminance is LIGHT BLUE/PURPLE
                #indigo at 0 luminance is WHITE
                #indigo at 0.5 luminance is WHITE
                #indigo at 1.0 luminance is WHITE

                #conclusion: indigo is too bright, needs less luminance
                #conclusion: indigo is too blue, needs less blue or more red

                #======================================================
                #violet at -1.0 luminance is BLUE/LIGHT BLUE
                #violet at -0.5 luminance is LIGHT BLUE
                #violet at 0 luminance is WHITE/BLUE
                #violet at 0.5 luminance is WHITE
                #violet at 1.0 luminance is WHITE

                #conclusion: violet is too bright, needs less luminance
                #conclusion: violet is too blue, needs less blue or more red

                #======================================================









                #apply the offsets to the noise tensor
                offsets_tensor =torch.Tensor(offsets).reshape([4,1,1])
                #move the offset tensor to the same device as the noise tensor
                offsets_tensor = offsets_tensor.to(noise.device)
                noise = torch.add(noise, offsets_tensor)
                #red strength = 
                #offsetlist[2] = offsetlist[2] + strength.value*mult*(-color[0]) #red
                #green strength =
                #offsetlist[3] = offsetlist[3] + strength.value*mult*color[1] #green
                #blue strength =
                #offsetlist[3] = offsetlist[3] + strength.value*mult*(-color[2]) #magenta
                #offsetlist[2] = offsetlist[2] + strength.value*mult*(color[2])  #cyan


                #offsettensor =torch.Tensor(offsetlist).reshape([4,1,1])
                #move the offset tensor to the same device as the noise tensor
                #offsettensor = offsettensor.to(noise.device)
                #noise = torch.add(noise, offsettensor)

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
                                normalize_button = gr.Checkbox(label="Normalize", value=False)
                                color_button = gr.ColorPicker(label="ColorSelector")
                        with gr.Row():
                                strength_slider = gr.Slider(-1, 2, step=0.1, default=0.5, label="StrengthSlider")
                        with gr.Row():
                                normalize_before_slider = gr.Slider(0, 1, step=0.05, default=0, label="Normalize Before")
                                normalize_after_slider = gr.Slider(0, 1, step=0.05, default=0, label="Normalize After")
                        with gr.Row():
                                #luminance control
                                luminance_slider = gr.Slider(-1, 1, step=0.05, default=0, label="Luminance (L0/L1)")
                return [enabled_button, normalize_button, color_button, strength_slider, normalize_before_slider, normalize_after_slider, luminance_slider]
        
        def process(self, p, enabled_button, normalize_button, color_button, strength_slider, normalize_before_slider, normalize_after_slider, luminance_slider):
                global color
                global enabled
                global strength
                global normalize
                global normalize_weights
                global luminance
                luminance = luminance_slider

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
                        color = [x/255 for x in color]
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


