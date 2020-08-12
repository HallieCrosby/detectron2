#The below code should be run from the notebook to compile this file. 
#!pip install kfp --upgrade
#!which dsl-compile
#!dsl-compile --py d2_drone_train_deploy.py --output d2_drone_train_deploy.tar.gz


#!/usr/bin/env python3
import kfp
import json
import copy
from kfp import components
from kfp import dsl

# Update this to match the name of your bucket
my_bucket_name = "sagemaker-detectron2-demo2-challie"

sagemaker_train_op = components.load_component_from_url('https://raw.githubusercontent.com/kubeflow/pipelines/c52a73e52c64a2d1414d0294e8617da42445dfd8/components/aws/sagemaker/train/component.yaml')
sagemaker_model_op = components.load_component_from_url('https://raw.githubusercontent.com/kubeflow/pipelines/c52a73e52c64a2d1414d0294e8617da42445dfd8/components/aws/sagemaker/model/component.yaml')
sagemaker_deploy_op = components.load_component_from_url('https://raw.githubusercontent.com/kubeflow/pipelines/c52a73e52c64a2d1414d0294e8617da42445dfd8/components/aws/sagemaker/deploy/component.yaml')

def training_input(input_name, s3_uri):
    return {
        "ChannelName": input_name,
        "DataSource": {
            "S3DataSource": {
                "S3Uri": s3_uri,
                "S3DataType": "S3Prefix",
                "S3DataDistributionType": "FullyReplicated",
            }
        },
        "CompressionType": "None",
        "RecordWrapperType": "None",
        "InputMode": "File",
    }


trainChannels = [
    training_input(
        "training", f"s3://{my_bucket_name}/semantic_drone_dataset"
    )
]


@dsl.pipeline(
    name="SMKF drone detectron pipeline",
    description="train and deploy Detectron2 object detection model using Amazon Sagemaker",
)
def d2_drone_train_deploy(
    region="us-east-2",
    # General component inputs
    instance_type="ml.p3.16xlarge",
    instance_count=4,
    volume_size=100,
    max_run_time=86400,
    endpoint_url="",
    network_isolation=False,
    traffic_encryption=False,
    role_arn="",
    # Training inputs
    train_image="445452627216.dkr.ecr.us-east-2.amazonaws.com/d2-sm-drone-b:latest",
    train_input_mode="File",
    train_output_location=f"s3://{my_bucket_name}/output10",
    train_channels=trainChannels,
    train_spot_instance=False,
    train_max_wait_time=3600,
    train_checkpoint_config={},
    # Serving inputs
    serving_image="445452627216.dkr.ecr.us-east-2.amazonaws.com/d2-drone-serving-b:latest"
    
):
 
    training = sagemaker_train_op(
        region=region,
        endpoint_url=endpoint_url,
        image=train_image,
        training_input_mode=train_input_mode,
        hyperparameters={"config-file":"COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml", 
                   "resume":"True",
                   "opts": "SOLVER.MAX_ITER 1000 \
                   MODEL.ROI_HEADS.NUM_CLASSES 1\
                   SOLVER.REFERENCE_WORLD_SIZE 8"
                   },
        channels=train_channels,
        instance_type=instance_type,
        instance_count=instance_count,
        volume_size=volume_size,
        max_run_time=max_run_time,
        model_artifact_path=train_output_location,
        network_isolation=network_isolation,
        traffic_encryption=traffic_encryption,
        spot_instance=train_spot_instance,
        max_wait_time=train_max_wait_time,
        checkpoint_config=train_checkpoint_config,
        role=role_arn,
    )

    create_model = sagemaker_model_op(
        region=region,
        endpoint_url=endpoint_url,
        model_name=training.outputs["job_name"],
        image=serving_image,
        model_artifact_url=training.outputs["model_artifact_url"],
        network_isolation=network_isolation,
        role=role_arn,
    )

    prediction = sagemaker_deploy_op(
        region=region,
        model_name_1=create_model.output,
        instance_type_1='ml.g4dn.xlarge',
        initial_instance_count_1='1'
    )


if __name__ == "__main__":
    kfp.compiler.Compiler().compile(d2_drone_train_deploy, __file__ + ".zip")

