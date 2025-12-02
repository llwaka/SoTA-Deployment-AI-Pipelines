# SoTA-Deployment-AI-Pipelines
Production deployment AI pipelines automate the deployment of AI models in production. Amongst key considerations when serving AI models in production is to ensure that the AI soultion meets several application requirements, including latency and throughput. As such, in real-world AI deployments, model quality can be traded-off for low latency and high throughput from system architecture and operational perspectives.

Generally, the performance of AI application in production is affected by many factors often categorized in three classes:
- **Hardware**: the physical computing resources that run the AI system e.g., Compute power (GPUs/TPUs/CPUs), memory etc.
- **Software**: code, framework, and configurations that govern how the model is executed e.g., model architecture and size, model optimization techniques (quantization, prunning, distillation), Inference frameworks, batching strategy etc.
- **Service**: the overall system environment where the AI runs e.g., API architecture (synchronous vs. asynchronous) that affects latency, scalability, and user experience.

In this workshop, you will perform simple AI inference evaluations (w.r.t latency and throughput) of an AI service that uses a pre-trained YOLO model for the specified scenario of single-stream and offline benchmark.
- **Single-stream**: one query stream with a sample size of one (often useful for apps with critical latency response). Metric: stream's 90th-percentile latency
- **Offline**: a query that includes all sample-data IDs in batches (often for batch processing apps, latency is unconstrained). Metric: Throughput (samples/second)
For the evaluations, a coco-dataset will be used.

## Procedure
- Run the YOLO11 model deployed as service using bentoml.
   - Note: Ensure correct setting of the onnxruntime provider. Commands: ``import onnxruntime as ort``and ``print("Available ONNX Runtime Execution Providers:", available_providers)``
- Print system performance (latency and throughput for the scenarios) on CPU.
- Interprete the results.
- Optinal: make changes and run on GPU


## Running the code
1. Using Python/conda virtual environment
   - Clone repository
   - Create the virtual environment, activate it and install dependencies specified in requirements.txt
   - Open two terminals and
      - on one terminal, run bento server with command ``bentoml serve``
      - on the other terminal, run benchmark file e.g., ``python benchmark.py --folder "data/coco-dataset/val2017" --sizes 20 100 300 500 ``
   
2. Using Docker
   - Clone repository
   - Build docker image: ``docker build -t yolo11-bento .``
   - Run BentoML service: ``docker run -p 3000:3000 yolo11-bento``
   - Run benchmark file inside the container: `` docker run --rm yolo11-bento \
    python benchmark.py --folder data/coco-dataset/val2017 --url http://host.docker.internal:3000/detect``

  
