import modules.scripts as scripts
import gradio as gr
import os
import torch
import numpy as np

from modules import images, devices, script_callbacks
import modules.processing as processing
from modules.shared import opts, cmd_opts, state
import modules.shared as shared

enabled = None
color = [0.0,0.0,0.0]
strength = 1.0

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
        print("Enabled: " + str(enabled.value))
        if enabled.value == True:
                #the 0.18215 is the hard-coded magic number scale factor.
                
                #layer 0 is brightness
                #layer 1 is brightness
                #layer 2 is red-cyan
                #layer 3 is magenta-green

                #layer 0, 1, 2, 3
                #offsetlist = [0,0,0.2,0.2]
                #offsetlist = [0,0,0.18215,0.18215]
                offsetlist = [0,0,0,0]
                mult = 0.18215
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
                
                #red strength = 
                offsetlist[2] = offsetlist[2] + strength.value*mult*(-color[0]) #red



                #green strength =
                offsetlist[3] = offsetlist[3] + strength.value*mult*color[1] #green
                

                #blue strength =
                offsetlist[3] = offsetlist[3] + strength.value*mult*(-color[2]) #magenta
                offsetlist[2] = offsetlist[2] + strength.value*mult*(color[2])  #cyan


                offsettensor =torch.Tensor(offsetlist).reshape([4,1,1])
                #move the offset tensor to the same device as the noise tensor
                offsettensor = offsettensor.to(noise.device)
                noise = torch.add(noise, offsettensor)
        
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
                global enabled, color, strength
                with gr.Accordion('Colorful Noise', open=False):
                        #enable or disable the extension, checkbox
                        with gr.Row():
                                enabled = gr.Checkbox(label="Enabled", value=True)
                                color_button = gr.ColorPicker(label="Color", value="#000000")
                                #gradio.ColorPicker.change,
                                #This listener is triggered when the component's value changes either because of user input (e.g. a user types in a textbox) OR because of a function update
                                #update the color value
                        with gr.Row():
                                strength = gr.Slider(0, 2, step=0.1, value=1)
                return [enabled, strength, color_button]

        def process(self, p, enabled, strength, color_button):
                
                #color is a list of 3 values, r,g,b, each between 0 and 1
                #convert to a list of 3 values, each between 0 and 1
                #color is a string, so we need to convert it to a list of 3 values
                #COLOR FORMAT:
                # #ff0080
                #convert hex to rgb
                global color

                h = color_button.lstrip('#')
                color = list(int(h[i:i+2], 16) for i in (0, 2, 4))
                color = [x/255 for x in color]
                print(color)
                
                #need to override the create_random_tensors function in processing.py, but it's not a class, it's just a function
                #doing this just causes a recursion depth error
                #fix the recursion error and override the function
                processing.create_random_tensors = create_random_tensors
                #reset p to use the new create_random_tensors function
                p.create_random_tensors = create_random_tensors
                

                #proc = process_images(p)
                #return proc


