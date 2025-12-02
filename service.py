import bentoml
import cv2
from ultralytics import YOLO
import numpy as np
import onnxruntime as ort
from PIL import Image as PILImage

# YOLO model
model_org = YOLO("yolo11s.pt")
model_org.export(format="onnx", nms=True, task="detect")  # Export to ONNX with NMS

# Preprocessing (YOLO)
def letterbox(img, size=640):
    """Resize + pad image while keeping aspect ratio (Ultralytics style)."""
    h, w = img.shape[:2]
    scale = size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_w, pad_h = size - new_w, size - new_h
    top, bottom = pad_h // 2, pad_h - (pad_h // 2)
    left, right = pad_w // 2, pad_w - (pad_w // 2)

    img_padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT)
    return img_padded, scale, (left, top)

def preprocess(image: PILImage.Image, size=640):
    """Convert PIL → padded tensor of shape [1,3,H,W]."""
    img = np.array(image.convert("RGB"))
    img_lb, scale, pad = letterbox(img, size=size)
    img_chw = img_lb[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    return img_chw[np.newaxis, ...], img.shape, scale, pad

def scale_box(box, scale, pad):
    """Scale YOLO padded detection box back to original image size."""
    x1, y1, x2, y2 = box
    x1, y1 = (x1 - pad[0]) / scale, (y1 - pad[1]) / scale
    x2, y2 = (x2 - pad[0]) / scale, (y2 - pad[1]) / scale
    return [x1, y1, x2, y2]

# Bentoml Service
@bentoml.service(name= "Object_Detection_Service")
class YOLO11Service:
    
    def __init__(self):
        # Load the model and convert to ONNX format with NMS
        super().__init__()
        self.model_path= "yolo11s.onnx"
        
        # Detect provider. Notice on GPU enabled machines, CUDAExecutionProvider will be used. Also, remember to use onnxruntime-gpu package.
        available_providers = ort.get_available_providers() # Available ONNX Runtime Execution Providers
        
        providers = [
            p for p in available_providers 
            if p in ('CPUExecutionProvider')] # Others:'CUDAExecutionProvider', 'CoreMLExecutionProvider',

        # Create ONNX Runtime Inference Session
        self.session = ort.InferenceSession(self.model_path, providers=providers) # Session is tha actual object used to run inference. The model is loaded once at service startup.
        self.input_name = self.session.get_inputs()[0].name # the name of model's input tensor

    @bentoml.api
    async def detect(self, image: PILImage.Image) -> dict:
        
        """ YOLO11s ONNX detection API using built-in NMS. Ultralytics ONNX (nms=True) outputs: [num_dets, 6] → [x1, y1, x2, y2, conf, class_id] """
        # Preprocess the input image
        input_tensor, original_shape, scale, pad = preprocess(image) #        
        # Run inference
        outputs = self.session.run(None, {self.input_name: input_tensor})
        detections = outputs[0]  # Model outputs with built-in NMS. shape: [num_dets, 6]
        
        # Postprocess the outputs
        results = []
        for det in detections:
          
            # Convert any nested structure to flat list of floats
            flat_det = []

            for d in det[:6]:  # first 6 elements: x1, y1, x2, y2, conf, cls_id
                # If it's a numpy array, flatten and take first element
                if isinstance(d, np.ndarray):
                    flat_det.extend(d.flatten().tolist())
                # If it's a list, flatten recursively
                elif isinstance(d, list):
                    flat_det.extend(np.array(d).flatten().tolist())
                else:  # already a scalar
                    flat_det.append(float(d))

            # Take first 6 values safely
            x1, y1, x2, y2, conf, cls_id = flat_det[:6]

            box = scale_box([x1, y1, x2, y2], scale, pad)

            results.append({
                "box": box,
                "confidence": float(conf),
                "class_id": int(cls_id)
            })
            
        return {"detections": results}
        
# Create a BentoML service instance
yolo11_service = YOLO11Service()



