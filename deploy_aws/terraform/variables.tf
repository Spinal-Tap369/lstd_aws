variable "aws_region" {
  type        = string
  description = "AWS region."
  default     = "eu-central-1"
}

variable "project_name" {
  type        = string
  description = "Project name."
  default     = "lstd-aws"
}

variable "artifact_bucket_name" {
  type        = string
  description = "S3 bucket for artifacts and data."
  default     = "my-lstd-data"
}

variable "artifact_promotion_prefix" {
  type        = string
  description = "S3 prefix for training artifacts."
  default     = "training-artifacts"
}

variable "raw_candles_queue_name" {
  type        = string
  description = "Raw candles FIFO queue name."
  default     = "raw-candles.fifo"
}

variable "stream_state_table_name" {
  type        = string
  description = "DynamoDB stream state table name."
  default     = "stream_state"
}

variable "artifact_promotion_lambda_name" {
  type        = string
  description = "Artifact promotion Lambda name."
  default     = "artifact-promotion-lambda"
}

variable "subnet_id" {
  type        = string
  description = "Subnet ID for EC2 instances."
}

variable "lambda_subnet_ids" {
  type        = list(string)
  description = "Subnet IDs for Lambda VPC attachment."
  default     = []
}

variable "train_ami_id" {
  type        = string
  description = "Training instance AMI ID."
}

variable "inference_ami_id" {
  type        = string
  description = "Inference instance AMI ID."
}

variable "collector_ami_id" {
  type        = string
  description = "Collector instance AMI ID."
}

variable "train_instance_type" {
  type        = string
  description = "Training instance type."
  default     = "g4dn.xlarge"
}

variable "inference_instance_type" {
  type        = string
  description = "Inference instance type."
  default     = "t3.large"
}

variable "collector_instance_type" {
  type        = string
  description = "Collector instance type."
  default     = "t3.micro"
}

variable "train_root_volume_size" {
  type        = number
  description = "Training root volume size in GiB."
  default     = 125
}

variable "inference_root_volume_size" {
  type        = number
  description = "Inference root volume size in GiB."
  default     = 125
}

variable "collector_root_volume_size" {
  type        = number
  description = "Collector root volume size in GiB."
  default     = 10
}

variable "train_key_name" {
  type        = string
  description = "Training EC2 key pair name."
  default     = null
}

variable "inference_key_name" {
  type        = string
  description = "Inference EC2 key pair name."
  default     = null
}

variable "train_instance_profile_name" {
  type        = string
  description = "Training instance profile name."
}

variable "inference_instance_profile_name" {
  type        = string
  description = "Inference instance profile name."
}

variable "collector_instance_profile_name" {
  type        = string
  description = "Collector instance profile name."
}

variable "lambda_role_arn" {
  type        = string
  description = "Lambda execution role ARN."
}

variable "ssh_ingress_cidr_blocks" {
  type        = list(string)
  description = "SSH ingress CIDR blocks."
  default     = []
}

variable "train_security_group_ids" {
  type        = list(string)
  description = "Training security group IDs."
  default     = []
}

variable "inference_security_group_ids" {
  type        = list(string)
  description = "Inference security group IDs."
  default     = []
}

variable "collector_security_group_ids" {
  type        = list(string)
  description = "Collector security group IDs."
  default     = []
}

variable "lambda_security_group_ids" {
  type        = list(string)
  description = "Lambda security group IDs."
  default     = []
}

variable "train_spot_enabled" {
  type        = bool
  description = "Use Spot for training."
  default     = true
}

variable "inference_spot_enabled" {
  type        = bool
  description = "Use Spot for inference."
  default     = true
}

variable "collector_spot_enabled" {
  type        = bool
  description = "Use Spot for collector."
  default     = true
}

variable "train_spot_max_price" {
  type        = string
  description = "Training Spot max price."
  default     = ""
}

variable "inference_spot_max_price" {
  type        = string
  description = "Inference Spot max price."
  default     = ""
}

variable "collector_spot_max_price" {
  type        = string
  description = "Collector Spot max price."
  default     = ""
}

variable "repo_url" {
  type        = string
  description = "Repository URL."
  default     = "https://github.com/Spinal-Tap369/lstd_aws.git"
}

variable "repo_branch" {
  type        = string
  description = "Repository branch."
  default     = "main"
}

variable "artifact_promotion_source_dir" {
  type        = string
  description = "Artifact promotion Lambda source path."
  default     = "../artifact_promotion_lambda"
}

variable "artifact_promotion_handler" {
  type        = string
  description = "Lambda handler."
  default     = "artifact_promotion_lambda.lambda_handler"
}

variable "artifact_promotion_runtime" {
  type        = string
  description = "Lambda runtime."
  default     = "python3.11"
}

variable "artifact_promotion_timeout" {
  type        = number
  description = "Lambda timeout in seconds."
  default     = 60
}

variable "artifact_promotion_memory_size" {
  type        = number
  description = "Lambda memory size in MB."
  default     = 256
}

variable "common_tags" {
  type        = map(string)
  description = "Common resource tags."
  default     = {}
}