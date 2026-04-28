output "artifact_bucket_name" {
  value       = aws_s3_bucket.artifacts.bucket
  description = "Artifact and data bucket name."
}

output "raw_candles_queue_url" {
  value       = aws_sqs_queue.raw_candles.url
  description = "Raw candles FIFO queue URL."
}

output "raw_candles_queue_arn" {
  value       = aws_sqs_queue.raw_candles.arn
  description = "Raw candles FIFO queue ARN."
}

output "stream_state_table_name" {
  value       = aws_dynamodb_table.stream_state.name
  description = "DynamoDB stream state table name."
}

output "train_instance_id" {
  value       = aws_instance.train.id
  description = "Training EC2 instance ID."
}

output "train_private_ip" {
  value       = aws_instance.train.private_ip
  description = "Training EC2 private IP."
}

output "inference_instance_id" {
  value       = aws_instance.inference.id
  description = "Inference EC2 instance ID."
}

output "inference_private_ip" {
  value       = aws_instance.inference.private_ip
  description = "Inference EC2 private IP."
}

output "collector_instance_id" {
  value       = aws_instance.collector.id
  description = "Collector EC2 instance ID."
}

output "collector_private_ip" {
  value       = aws_instance.collector.private_ip
  description = "Collector EC2 private IP."
}

output "artifact_promotion_lambda_name" {
  value       = aws_lambda_function.artifact_promotion.function_name
  description = "Artifact promotion Lambda function name."
}

output "artifact_promotion_lambda_arn" {
  value       = aws_lambda_function.artifact_promotion.arn
  description = "Artifact promotion Lambda ARN."
}
