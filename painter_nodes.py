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
                "image_name": (sorted(files), {"image_upload": True})
            },
            "optional": {
                "image": ("IMAGE",)
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "process_image"
    CATEGORY = "image"
    
    OUTPUT_NODE = True

    def process_image(self, image_name, image=None):
        is_stream_mode = image is not None
        
        # Use the selected filename as the display name in both modes
        display_name = image_name
        
        # Path for the file that will be displayed in UI
        display_path = os.path.join(self.output_dir, display_name)
        
        # Source image for processing and mask generation
        if is_stream_mode:
            # Convert tensor to PIL
            i = 255. * image[0].cpu().numpy()
            img_pil = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            # Save to the display path for UI preview
            img_pil.save(display_path, pnginfo=None)
        else:
            # Load from selected file, ensure it exists at display path
            source_path = folder_paths.get_annotated_filepath(image_name)
            img_pil = Image.open(source_path)
            
            # If source is not at display path, copy it there
            if source_path != display_path:
                img_pil.save(display_path, pnginfo=None)

        # Generate mask from the display file
        image_path = folder_paths.get_annotated_filepath(display_name)
        img_for_mask = Image.open(image_path)
        
        output_masks = []
        for i in ImageSequence.Iterator(img_for_mask):
            i = ImageOps.exif_transpose(i)
            if "A" in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            else:
                mask = torch.zeros((img_pil.height, img_pil.width), dtype=torch.float32)
            output_masks.append(mask.unsqueeze(0))

        output_mask = torch.cat(output_masks, dim=0) if len(output_masks) > 1 else output_masks[0]
        
        # Prepare output tensor
        if not is_stream_mode:
            source_img_np = np.array(img_pil.convert('RGB')).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(source_img_np).unsqueeze(0)
        else:
            image_tensor = image

        return {
            "ui": {
                # Always return the selected filename for UI consistency
                "images": [{"filename": display_name, "type": "input"}],
            },
            "result": (image_tensor, output_mask)
        }

    @classmethod
    def IS_CHANGED(s, image_name, image=None):
        # This is critical: return a unique value when upstream image is connected
        # This forces the frontend to refresh the preview
        if image is not None:
            # Create a unique signature based on the actual image content
            # This ensures the UI updates on every new upstream image
            mean_val = float(torch.mean(image))
            # Add hash of the image tensor to force refresh
            image_hash = hashlib.md5(image[0].cpu().numpy().tobytes()).hexdigest()[:8]
            return f"stream_{image_hash}_{mean_val}"
        else:
            # For manual mode, use file modification time
            source_path = folder_paths.get_annotated_filepath(image_name)
            return os.path.getmtime(source_path)

    @classmethod
    def VALIDATE_INPUTS(s, image_name, image=None):
        return True
