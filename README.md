# SoTA-Deployment-AI-Pipelines
Production deployment AI pipelines automate the deployment of AI models in production. Amongst key considerations when serving AI models in production is to ensure that the AI soultion meets several application requirements, including latency and throughput. As such, in real-world AI deployments, model quality can be traded-off for low latency and high throughput from system architecture and operational perspectives.

Generally, the performance of AI application in production is affected by many factors often categorized in three classes:
- **Hardware**: the physical computing resources that run the AI system e.g., Compute power (GPUs/TPUs/CPUs), memory etc.
- **Software**: code, framework, and configurations that govern how the model is executed e.g., model architecture and size, model optimization techniques (quantization, prunning, distillation), Inference frameworks, batching strategy etc.
- **Service**: the overall system environment where the AI runs e.g., API architecture (synchronous vs. asynchronous) that affects latency, scalability, and user experience.

In this workshop, you will perform simple AI inference evaluations (w.r.t latency and throughput) of an AI service that uses a pre-trained [YOLO11 model in onnx format](https://docs.ultralytics.com/integrations/onnx/#cpu-deployment) for the specified scenario of single-stream server scenario.
- **Single-stream** Latency Test (baseline)*: sends one request at a time and evaluates the best possible per-image latency. Metric: per-request latency
- **Concurrent session scaling test**: for each concurency level (e.g., 1, 2, 4, 8 workers) assumed as multiclients send requests simulatenously and measure whether the system is cabable of handling server workloads. Metric: Throughput (images/second)

For the evaluations, a [coco-dataset](https://cocodataset.org/#download) will be used and is already downloaded in this repository.

## Procedure
- Run the YOLO11 model deployed as service with an API endpoint using bentoml.
   - Note: Ensure correct setting of the onnxruntime provider. Commands: ``import onnxruntime as ort``and ``print("Available ONNX Runtime Execution Providers:", available_providers)``
- Print performance metrics (latency and throughput for the given scenarios) on CPU.
- Interprete the results.
- Optinal: make different changes (e.g., to the provider and run on GPU) and observe performance


## Running the code
Using Python/conda virtual environment
   - Clone repository
   - Create the virtual environment, activate it and install dependencies specified in requirements.txt
   - Open two terminals and
      - on one terminal, run bento server with command ``bentoml serve``
      - on the other terminal, run benchmark file e.g., ``python benchmark.py --folder "data/coco-dataset/val2017" ``


  
