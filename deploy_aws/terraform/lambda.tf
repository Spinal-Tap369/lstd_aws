data "archive_file" "artifact_promotion_zip" {
  type        = "zip"
  source_dir  = var.artifact_promotion_source_dir
  output_path = "${path.module}/.terraform/artifact_promotion_lambda.zip"
}

resource "aws_lambda_function" "artifact_promotion" {
  function_name    = var.artifact_promotion_lambda_name
  role             = var.lambda_role_arn
  handler          = var.artifact_promotion_handler
  runtime          = var.artifact_promotion_runtime
  timeout          = var.artifact_promotion_timeout
  memory_size      = var.artifact_promotion_memory_size
  filename         = data.archive_file.artifact_promotion_zip.output_path
  source_code_hash = data.archive_file.artifact_promotion_zip.output_base64sha256

  environment {
    variables = {
      AWS_REGION               = var.aws_region
      ARTIFACT_BUCKET          = aws_s3_bucket.artifacts.bucket
      ARTIFACT_PREFIX          = var.artifact_promotion_prefix
      INFERENCE_INSTANCE_ID    = aws_instance.inference.id
      RAW_QUEUE_URL            = aws_sqs_queue.raw_candles.url
      STREAM_STATE_TABLE       = aws_dynamodb_table.stream_state.name
    }
  }

  dynamic "vpc_config" {
    for_each = length(var.lambda_subnet_ids) > 0 ? [1] : []
    content {
      subnet_ids         = var.lambda_subnet_ids
      security_group_ids = var.lambda_security_group_ids
    }
  }

  tags = local.lambda_tags
}
