from .painter_nodes import PainterImageLoad

NODE_CLASS_MAPPINGS = {
    "PainterImageLoad": PainterImageLoad
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PainterImageLoad": "Painter Image Load"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
