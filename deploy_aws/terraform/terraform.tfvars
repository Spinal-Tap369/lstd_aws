aws_region = "eu-central-1"
project_name = "lstd-aws"

artifact_bucket_name = "my-lstd-data"
artifact_promotion_prefix = "training-artifacts"
raw_candles_queue_name = "raw-candles.fifo"
stream_state_table_name = "stream_state"
artifact_promotion_lambda_name = "artifact-promotion-lambda"

subnet_id = "subnet-0d216effc9f597515"
lambda_subnet_ids = []

train_ami_id = "ami-03e379925f8717e71"
inference_ami_id = "ami-03e379925f8717e71"
collector_ami_id = "ami-0de6934e87badb694"

train_instance_type = "g4dn.xlarge"
inference_instance_type = "t3.large"
collector_instance_type = "t3.micro"

train_root_volume_size = 125
inference_root_volume_size = 125
collector_root_volume_size = 10

train_key_name = "money-pair.pem"
inference_key_name = "money-pair.pem"

train_instance_profile_name = "ec2-lstd-train-role"
inference_instance_profile_name = "ec2-lstd-live-role"
collector_instance_profile_name = "pipeline-ec2-role"

lambda_role_arn = "arn:aws:iam::189599797847:role/artifact-promotion-lambda-role"

train_security_group_ids = ["sg-011b54a8d30c78388"]
inference_security_group_ids = ["sg-011b54a8d30c78388"]
collector_security_group_ids = ["sg-011b54a8d30c78388"]
lambda_security_group_ids = []

train_spot_enabled = true
inference_spot_enabled = true
collector_spot_enabled = true

train_spot_max_price = ""
inference_spot_max_price = ""
collector_spot_max_price = ""

repo_url = "https://github.com/Spinal-Tap369/lstd_aws.git"
repo_branch = "main"

artifact_promotion_source_dir = "../artifact_promotion_lambda"
artifact_promotion_handler = "artifact_promotion_lambda.lambda_handler"
artifact_promotion_runtime = "python3.11"
artifact_promotion_timeout = 60
artifact_promotion_memory_size = 256

common_tags = {
  Environment = "dev"
  Owner       = "Samuel"
}
