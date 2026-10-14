"""Typed projection errors — never silently coerce bad panoramas."""


class NotFullEquirectError(ValueError):
    """Image is not full equirectangular 2:1 (W == 2*H)."""


class PartialPanoError(ValueError):
    """GPano cropped-area metadata indicates a partial sphere."""
