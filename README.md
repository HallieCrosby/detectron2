This is not official AWS repository. Code provided "as is".

# Goals
This repository implements port of latest [Detectron2](https://github.com/facebookresearch/detectron2/) ("D2") to [Amazon Sagemaker Components for Kubeflow Pipelines](https://aws.amazon.com/blogs/machine-learning/introducing-amazon-sagemaker-components-for-kubeflow-pipelines/). Scope includes:
- [x] finetune D2 model on custom dataset using Sagemaker distributed training and hosting via Kubeflow Pipelines.

## Containers
Amazon Sagemaker uses docker containers both for training and inference:
- `Dockerfile` is training container, sources from `container_training` directory will be added at training time;
- `Dockerfile.serving` is serving container, `container_serving` directory will added at inference time.
- 'Dockerfile.dronetraining' is a custom training container for custom dataset.

**Note**: by default training container compiles Detectron2 for Volta architecture (Tesla V100 GPUs). If you'd like to run training on other GPU architectures, consider updating this [environment variable](https://github.com/vdabravolski/detectron2-sagemaker/blob/e6252211b819815962207d1a61d5675d213e0f25/Dockerfile#L21). Here is an example on how to compile Detectron2 for all supported architectures:

`ENV TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"`

**Custom Container**:
We extended one of our existing and predefined SageMaker deep learning framework containers for this demo. 

Amazon SageMaker provides prebuilt Docker images that include deep learning framework libraries and other dependencies needed for training and inference. With the Amazon SageMaker SageMaker Python SDK, you can train and deploy models using these popular deep learning frameworks. These prebuilt Docker images are stored in Amazon Elastic Container Registry (Amazon ECR). 

The AWS Deep Learning Containers for PyTorch include containers for training on CPU and GPU, optimized for performance and scale on AWS. These Docker images have been tested with Amazon SageMaker, EC2, ECS, and EKS, and provide stable versions of NVIDIA CUDA, cuDNN, Intel MKL, and other required software components to provide a seamless user experience for deep learning workloads. All software components in these images are scanned for security vulnerabilities and updated or patched in accordance with AWS Security best practices.

You can customize these prebuilt containers or extend them to handle any additional functional requirements for your algorithm or model that the prebuilt Amazon SageMaker Docker image doesn't support.

## Distributed training on COCO2017 dataset
See `d2_byoc_coco2017_training.ipynb` for end-to-end example of how to train your Detectron2 model on Sagemaker. Current implementation supports both multi-node and multi-GPU training on Sagemaker distributed cluster.

### Training cluster config
- To define parameters of your distributed training cluster, use Sagemaker Estimator configuration:
```python
d2 = sagemaker.estimator.Estimator(...
                                   train_instance_count=2, 
                                   train_instance_type='ml.p3.16xlarge',
                                   train_volume_size=100,
                                   ...
                                   )
```
###  Detecrton2 config
Detectron2 config is defined in Sagemaker Hyperparameters dict:
```python
hyperparameters = {"config-file":"COCO-InstanceSegmentation/mask_rcnn_R_101_C4_3x.yaml", 
                   #"local-config-file" : "config.yaml", # if you'd like to supply custom config file, please add it in container_training folder, and provide file name here
                   "resume":"True", # whether to re-use weights from pre-trained model
                   "eval-only":"False", # whether to perform only D2 model evaluation
                  # opts are D2 model configuration as defined here: https://detectron2.readthedocs.io/modules/config.html#config-references
                  # this is a way to override individual parameters in D2 configuration from Sagemaker API
                   "opts": "SOLVER.MAX_ITER 20000"
                   }
```
There are 3 ways how you can fine-tune your Detectron2 configuration:
- you can use one of Detectron2 authored config files (e.g. `"config-file":"COCO-InstanceSegmentation/mask_rcnn_R_101_C4_3x.yaml"`).
- you can define your own config file and stored it `container_training` folder. In this case you need to define `local-config-file` parameter with name of desired config file. **Note**, that you can choose either `config-file` or `local-config-file`.
- you can modify individual parameters of Detectron2 configuration via `opts` list (e.g. `"opts": "SOLVER.MAX_ITER 20000"` above.


## Training and serving Detectron2 model for custom problem
See `d2_custom_drone_dataset.ipynb` notebook for details.


## Pytorch Distributed Training Specific to Detectron2:
* Dockerfile – for our training we use customer container, which is extended from Sagemaker Pytorch 1.5.1 container. We do so because we need to build Detectron2 from sources. 
1. Define base container;
2. Install required dependencies for Detectron2;
3. Copies training script and utilities to container;
4. Builds Detectron2 from source.
    * A few highlights in container:
        * L32 – copying directory with training script into container;
        * L38 – define env variable for directory where training code will be uploaded
        * L39 – define training script via pre-defined Sagemaker variable.
        * Note, Sagemaker will start training job by running “python SAGEMAKER_PROGRAM – arg 1 -arg 2 …”
* Train_drone.py – main training script file. Highlights:
    * #L199-L215 – argument parsing. These arguments will be defined by K8S YAML file.
        * Line #207 - we use Detectron2 launch utility to start training on multiple nodes.
    * #L217-L225 – start distributed training job. We use Detectron2 launch utility, which in its turn uses Pytorch native distributed.launch. As communication backend, we use NCCL as it gives best performance for GPU-based communication.
    * #L125-L147 – parameters of training world. We use environmental variables defined by Sagemaker to identify number of hosts, processes, and host rank and pass these parameters to Detectron2 launch utility.
        * Line #143 - in order to coordinate communication between different nodes, we need to gather details about our training world such as number of nodes, number of devices etc.
    * #L181-L193 – main method for training. Once training is done, we save model in one of processes. Model artifact will be stored in SM_OUTPUT_MODEL_DATA dir. Content of this directory is automatically uploaded by Sagemaker to S3.
        * Line #186 - Sagemaker will start training by running code in __main__ guard on all training nodes;
    * Line #8 /#169 - actual training script which will be executed on all processes and all GPU devices. Under the hood, PyTorch DistributedDataParallel class will ensure coordination of tasks between GPU devices.
        * DisrtibutedDataParallel is a model wrapper to implement multiprocessing across multiple nodes.
            * DistributedDataParallel (DDP) implements data parallelism at the module level which can run across multiple machines. Applications using DDP should spawn multiple processes and create a single DDP instance per process. DDP uses collective communications in the torch.distributed package to synchronize gradients and buffers. More specifically, DDP registers an autograd hook for each parameter given by model.parameters() and the hook will fire when the corresponding gradient is computed in the backward pass. Then DDP uses that signal to trigger gradient synchronization across processes. 
 

## Amazon Sagemaker Distributed Training: 

Training heavy-weight DNNs such as Mask R-CNN require high per GPU memory so you can pump one or more high-resolution images through the training pipeline. They also require high-speed GPU-to-GPU interconnect and high-speed networking interconnecting machines so synchronized allreduce of gradients can be done efficiently. Amazon SageMaker ml.p3.16xlarge and ml.p3dn.24xlarge instance types meet all these requirements. For more information, see Amazon SageMaker ML Instance Types. With eight Nvidia Tesla V100 GPUs, 128–256 GB GPU memory, 25–100 Gbps networking interconnect, and high-speed Nvidia NVLink GPU-to-GPU interconnect, they are ideally suited for distributed TensorFlow training on Amazon SageMaker.

* Amazon SageMaker requires the training algorithm and frameworks packaged in a Docker image.
* The Docker image must be enabled for Amazon SageMaker training. This enablement is simplified through the use of Amazon SageMaker containers, which is a library that helps create Amazon SageMaker-enabled Docker images.
* You need to provide an entry point script (typically a Python script) in the Amazon SageMaker training image to act as an intermediary between Amazon SageMaker and your algorithm code.
* To start training on a given host, Amazon SageMaker runs a Docker container from the training image and invokes the entry point script with entry point environment variables that provide information such as hyperparameters and the location of input data.
* The entry point script uses the information passed to it in the entry point environment variables to start your algorithm program with the correct args and polls the running algorithm process.
* When the algorithm process exits, the entry point script exits with the exit code of the algorithm process. Amazon SageMaker uses this exit code to determine the success or failure of the training job.
* The entry point script redirects the output of the algorithm process’ stdout and stderr to its own stdout. In turn, Amazon SageMaker captures the stdout from the entry point script and sends it to Amazon CloudWatch Logs. Amazon SageMaker parses the stdout output for algorithm metrics defined in the training job and sends the metrics to Amazon CloudWatch metrics.
* When Amazon SageMaker starts a training job that requests multiple training instances, it creates a set of hosts and logically names each host as algo-k, where k is the global rank of the host. For example, if a training job requests four training instances, Amazon SageMaker names the hosts as algo-1, algo-2, algo-3, and algo-4. The hosts can connect on the network using these hostnames.






