"""Family-neutral checkpoint router. Architecture detectors remain isolated."""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from h3converter.h3_detect import detect as detect_h3
from h3converter.krea2_detect import detect as detect_krea2
from h3converter.safetensor_io import Header

class ModelFamily(str, Enum):
    H3 = "minimax_h3"
    KREA2 = "krea2"
    UNKNOWN = "unknown"

@dataclass
class ModelDetection:
    family: ModelFamily
    detail: object
    @property
    def convertible(self): return bool(getattr(self.detail, "convertible", False))
    @property
    def geometry(self): return getattr(self.detail, "geometry", None)
    @property
    def errors(self): return getattr(self.detail, "errors", [])
    @property
    def warnings(self): return getattr(self.detail, "warnings", [])
    @property
    def float_dtype(self): return getattr(self.detail, "float_dtype", None)
    @property
    def already_quantized(self): return getattr(self.detail, "already_quantized", False)
    @property
    def already_curve_pruned(self): return getattr(self.detail, "already_curve_pruned", False)
    @property
    def is_h3(self): return self.family is ModelFamily.H3
    @property
    def is_krea2(self): return self.family is ModelFamily.KREA2

def detect(header: Header) -> ModelDetection:
    h3 = detect_h3(header)
    if h3.is_h3:
        return ModelDetection(ModelFamily.H3, h3)
    krea = detect_krea2(header)
    if krea.is_krea2:
        return ModelDetection(ModelFamily.KREA2, krea)
    errors = list(dict.fromkeys(h3.errors + krea.errors))
    h3.errors = errors
    return ModelDetection(ModelFamily.UNKNOWN, h3)
