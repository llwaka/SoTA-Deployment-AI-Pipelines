import bentoml
import numpy as np
import cv2
import onnxruntime as ort
from PIL import Image as PILImage
import asyncio
import concurrent.futures
import random
import traceback
from ultralytics import YOLO


# ---------------------------
# YOLO MODEL and PREPROCESSING
# ---------------------------
model_org = YOLO("yolo11s.pt")
model_org.export(format="onnx", nms=True, task="detect")  # Export to ONNX with NMS

def letterbox(img, new_size=640):
    h, w = img.shape[:2]
    scale = new_size / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    img_resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pad_w, pad_h = new_size - nw, new_size - nh
    top, bottom = pad_h // 2, pad_h - pad_h // 2
    left, right = pad_w // 2, pad_w - pad_w // 2
    img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right,
                                    cv2.BORDER_CONSTANT, value=(114,114,114))
    return img_padded, scale, (left, top)

def preprocess(image: PILImage.Image, size=640):
    """Convert PIL → padded tensor [1,3,H,W]"""
    img = np.array(image.convert("RGB"))
    img_lb, scale, pad = letterbox(img, size)
    img_chw = img_lb[:, :, ::-1].transpose(2,0,1).astype(np.float32)/255.0
    return img_chw[np.newaxis, ...], img.shape, scale, pad

def scale_box(box, scale, pad):
    x1, y1, x2, y2 = box
    x1, y1 = (x1 - pad[0])/scale, (y1 - pad[1])/scale
    x2, y2 = (x2 - pad[0])/scale, (y2 - pad[1])/scale
    return [max(x1,0), max(y1,0), x2, y2]

# ---------------------------
# Robust scalar conversion for Ultralytics ONNX output
# ---------------------------
def to_scalar(val):
    """Convert any float/array/list to a single Python float safely."""
    if isinstance(val, np.ndarray):
        if val.size == 1:
            return float(val.item())
        else:
            return float(val.flatten()[0])
    elif isinstance(val, list):
        return to_scalar(np.array(val))
    return float(val)

# ---------------------------
# YOLO11 SERVICE
# ---------------------------

@bentoml.service(name="YOLO11_Service", traffic={"timeout":300})
class YOLO11Service:

    def __init__(self):
        model_path = "yolo11s.onnx"

        # CPU for testing; switch to GPU if available. 
        # providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        # Also, remember to use onnxruntime-gpu package.
        providers = ["CPUExecutionProvider"]

        # Create ONNX Runtime session options
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 4
        sess_options.inter_op_num_threads = 4
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Create a pool of sessions for concurrency. Also, create ONNX runtime sessions.
        self.num_sessions = 4
        self.sessions = [
            ort.InferenceSession(model_path, sess_options, providers=providers)
            for _ in range(self.num_sessions)
        ]
        self.input_name = self.sessions[0].get_inputs()[0].name

        # ThreadPoolExecutor for async inference
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.num_sessions*2)

        # Warmup
        dummy = np.zeros((1,3,640,640),dtype=np.float32)
        for s in self.sessions:
            s.run(None, {self.input_name: dummy})
        print("[YOLO11] Warmup complete")

    # Run inference in a thread pool
    async def _run_inference(self, tensor):
        session = random.choice(self.sessions)
        loop = asyncio.get_event_loop()
        outputs = await loop.run_in_executor(self.executor, lambda: session.run(None, {self.input_name: tensor}))
        return outputs[0]

    # ---------------------------
    # API: raw image input
    # ---------------------------
    @bentoml.api
    async def detect(self, image: PILImage.Image):
        try:
            tensor, orig_shape, scale, pad = preprocess(image)
            raw_out = await self._run_inference(tensor)

            results = []
            for det in raw_out:
                # Safely convert any numpy array or nested list to scalar
                x1, y1, x2, y2, conf, cls_id = [to_scalar(v) for v in det[:6]] # first 6 elements: x1, y1, x2, y2, conf, cls_id
                box = scale_box([x1, y1, x2, y2], scale, pad)
                results.append({"box": box, "confidence": conf, "class_id": int(cls_id)})
            return {"detections": results}

        except Exception as e:
            print("[ERROR] Exception in detect():", e)
            traceback.print_exc()
            return {"error": str(e)}
# Create a BentoML service instance
yolo11_service = YOLO11Service()
