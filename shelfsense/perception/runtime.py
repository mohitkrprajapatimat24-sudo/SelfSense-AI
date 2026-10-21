"""CPU thread settings that behave inside a CPU-limited container.

Under `docker run --cpus 4` the process still sees every host core, so
libraries start one thread per core and fight over a 4-CPU time budget.
ONNX Runtime threads also spin while waiting for work, which burns that budget.
Everything here is capped to the configured thread count. Spinning helps on a
machine with idle cores and hurts under a CPU quota, so the container turns it
off with SHELFSENSE_SPIN=0.
"""

import os

import cv2
import onnxruntime as ort

SPIN = os.environ.get("SHELFSENSE_SPIN", "1")


def session_options(threads: int) -> ort.SessionOptions:
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.add_session_config_entry("session.intra_op.allow_spinning", SPIN)
    so.add_session_config_entry("session.inter_op.allow_spinning", SPIN)
    return so


def limit_threads(threads: int) -> None:
    cv2.setNumThreads(threads)
    try:
        import faiss
        faiss.omp_set_num_threads(threads)
    except ImportError:
        pass
    _patch_rapidocr()


_patched = False


def _patch_rapidocr() -> None:
    """RapidOCR builds its own ONNX sessions; switch spinning off there too."""
    global _patched
    if _patched:
        return
    from rapidocr.inference_engine.onnxruntime.main import OrtInferSession

    original = OrtInferSession._init_sess_opts

    def init_sess_opts(cfg):
        so = original(cfg)
        so.inter_op_num_threads = 1
        so.add_session_config_entry("session.intra_op.allow_spinning", SPIN)
        so.add_session_config_entry("session.inter_op.allow_spinning", SPIN)
        return so

    OrtInferSession._init_sess_opts = staticmethod(init_sess_opts)
    _patched = True
