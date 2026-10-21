"""YOLO11n shelf detector (product, price_tag, empty_gap) running on ONNX Runtime."""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from ..schemas import YOLO_CLASSES, BBox, DetClass, Detection

# per-class score thresholds: tags are small and cheap to OCR, so keep more of them
CONF = {DetClass.PRODUCT: 0.35, DetClass.PRICE_TAG: 0.30, DetClass.EMPTY_GAP: 0.35}
NMS_IOU = 0.5


class Detector:
    def __init__(self, onnx_path: Path, threads: int = 4):
        import onnxruntime as ort

        from .runtime import session_options

        self.session = ort.InferenceSession(str(onnx_path), session_options(threads),
                                            providers=["CPUExecutionProvider"])
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.size = int(inp.shape[2])

    def _letterbox(self, img: np.ndarray) -> tuple[np.ndarray, float, int, int]:
        h, w = img.shape[:2]
        r = self.size / max(h, w)
        nh, nw = round(h * r), round(w * r)
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR)
        top, left = (self.size - nh) // 2, (self.size - nw) // 2
        canvas = np.full((self.size, self.size, 3), 114, np.uint8)
        canvas[top:top + nh, left:left + nw] = resized
        return canvas, r, left, top

    def __call__(self, image: Image.Image) -> list[Detection]:
        img = np.asarray(image.convert("RGB"))
        canvas, r, dx, dy = self._letterbox(img)
        x = canvas.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
        out = self.session.run(None, {self.input_name: x})[0][0].T     # (anchors, 4 + classes)

        boxes_cxcywh, scores = out[:, :4], out[:, 4:]
        cls = scores.argmax(1)
        conf = scores.max(1)
        thresholds = np.array([CONF[c] for c in YOLO_CLASSES])
        keep = conf >= thresholds[cls]
        boxes_cxcywh, cls, conf = boxes_cxcywh[keep], cls[keep], conf[keep]
        if not len(conf):
            return []

        xywh = np.column_stack([boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2,
                                boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2,
                                boxes_cxcywh[:, 2], boxes_cxcywh[:, 3]])
        idx = cv2.dnn.NMSBoxesBatched(xywh.tolist(), conf.tolist(), cls.tolist(), 0.0, NMS_IOU)
        h, w = img.shape[:2]
        dets = []
        for i in np.array(idx).flatten():
            x1 = (xywh[i, 0] - dx) / r
            y1 = (xywh[i, 1] - dy) / r
            x2 = x1 + xywh[i, 2] / r
            y2 = y1 + xywh[i, 3] / r
            dets.append(Detection(
                bbox=BBox(x1=float(np.clip(x1, 0, w)), y1=float(np.clip(y1, 0, h)),
                          x2=float(np.clip(x2, 0, w)), y2=float(np.clip(y2, 0, h))),
                cls=YOLO_CLASSES[int(cls[i])], conf=float(conf[i])))
        return dets
