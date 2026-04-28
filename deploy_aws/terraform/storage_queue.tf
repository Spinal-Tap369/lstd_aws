resource "aws_s3_bucket" "artifacts" {
  bucket = var.artifact_bucket_name
  tags   = merge(local.common_tags, { Name = var.artifact_bucket_name })
}

resource "aws_s3_bucket_versioning" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_sqs_queue" "raw_candles_dlq" {
  name                        = "${replace(var.raw_candles_queue_name, ".fifo", "")}-dlq.fifo"
  fifo_queue                  = true
  content_based_deduplication = false
  visibility_timeout_seconds  = 300
  message_retention_seconds   = 1209600
  tags                        = merge(local.common_tags, { Name = "${local.name_prefix}-raw-candles-dlq" })
}

resource "aws_sqs_queue" "raw_candles" {
  name                        = var.raw_candles_queue_name
  fifo_queue                  = true
  content_based_deduplication = false
  visibility_timeout_seconds  = 300
  message_retention_seconds   = 1209600
  receive_wait_time_seconds   = 20

  redrive_policy = jsonencode({
    deadLetterTargetArn = aws_sqs_queue.raw_candles_dlq.arn
    maxReceiveCount     = 5
  })

  tags = merge(local.common_tags, { Name = "${local.name_prefix}-raw-candles" })
}

resource "aws_dynamodb_table" "stream_state" {
  name         = var.stream_state_table_name
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "symbol"
  range_key    = "interval"

  attribute {
    name = "symbol"
    type = "S"
  }

  attribute {
    name = "interval"
    type = "S"
  }

  point_in_time_recovery {
    enabled = true
  }

  tags = merge(local.common_tags, { Name = var.stream_state_table_name })
}
