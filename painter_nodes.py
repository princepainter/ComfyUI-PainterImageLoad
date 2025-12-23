import torch
import numpy as np
from PIL import Image, ImageOps, ImageSequence
import folder_paths
import node_helpers
import os
import hashlib

class PainterImageLoad:
    def __init__(self):
        self.output_dir = folder_paths.get_input_directory()

    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {
            "required": {
                "image": ("IMAGE",),
                "image_name": (sorted(files), {"image_upload": True})
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "process_image"
    CATEGORY = "image"
    
    # This flag enables the "Execute Selected Nodes" option in the context menu
    # and treats the node as a terminal/output node.
    OUTPUT_NODE = True

    def process_image(self, image, image_name):
        # Convert incoming tensor to PIL Image for saving
        # Assuming input shape is [B, H, W, C], taking the first image in batch
        i = 255. * image[0].cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        
        # Define sync filename for frontend compatibility
        filename = "painter_sync.png"
        save_path = os.path.join(self.output_dir, filename)
        
        # Save to input directory to ensure Mask Editor can access it
        img.save(save_path, pnginfo=None)

        # Handle MASK processing using the saved file
        image_path = folder_paths.get_annotated_filepath(filename)
        img_for_mask = Image.open(image_path)
        
        output_masks = []
        for i in ImageSequence.Iterator(img_for_mask):
            i = ImageOps.exif_transpose(i)
            if "A" in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            else:
                # Default mask if no alpha channel exists
                mask = torch.zeros((image.shape[1], image.shape[2]), dtype=torch.float32)
            output_masks.append(mask.unsqueeze(0))

        output_mask = torch.cat(output_masks, dim=0) if len(output_masks) > 1 else output_masks[0]

        # Return UI signal to force refresh the preview and filename on the node
        return {
            "ui": {
                "images": [{"filename": filename, "type": "input"}],
            },
            "result": (image, output_mask)
        }

    @classmethod
    def IS_CHANGED(s, image, image_name):
        # Ensure node triggers whenever the input image tensor changes
        return float(torch.mean(image))

    @classmethod
    def VALIDATE_INPUTS(s, image, image_name):
        return True
