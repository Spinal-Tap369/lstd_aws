resource "aws_instance" "train" {
  ami                         = var.train_ami_id
  instance_type               = var.train_instance_type
  subnet_id                   = var.subnet_id
  iam_instance_profile        = var.train_instance_profile_name
  key_name                    = var.train_key_name
  vpc_security_group_ids      = var.train_security_group_ids
  user_data_replace_on_change = true
  user_data                   = local.train_user_data

  instance_market_options {
    market_type = var.train_spot_enabled ? "spot" : "on-demand"

    dynamic "spot_options" {
      for_each = var.train_spot_enabled ? [1] : []
      content {
        instance_interruption_behavior = "stop"
        max_price                      = var.train_spot_max_price != "" ? var.train_spot_max_price : null
        spot_instance_type             = "persistent"
      }
    }
  }

  root_block_device {
    volume_size           = var.train_root_volume_size
    volume_type           = "gp3"
    delete_on_termination = true
    encrypted             = true
  }

  tags = local.train_tags
}

resource "aws_instance" "inference" {
  ami                    = var.inference_ami_id
  instance_type          = var.inference_instance_type
  subnet_id              = var.subnet_id
  iam_instance_profile   = var.inference_instance_profile_name
  key_name               = var.inference_key_name
  vpc_security_group_ids = var.inference_security_group_ids

  instance_market_options {
    market_type = var.inference_spot_enabled ? "spot" : "on-demand"

    dynamic "spot_options" {
      for_each = var.inference_spot_enabled ? [1] : []
      content {
        instance_interruption_behavior = "stop"
        max_price                      = var.inference_spot_max_price != "" ? var.inference_spot_max_price : null
        spot_instance_type             = "persistent"
      }
    }
  }

  root_block_device {
    volume_size           = var.inference_root_volume_size
    volume_type           = "gp3"
    delete_on_termination = true
    encrypted             = true
  }

  tags = local.inference_tags
}

resource "aws_instance" "collector" {
  ami                         = var.collector_ami_id
  instance_type               = var.collector_instance_type
  subnet_id                   = var.subnet_id
  iam_instance_profile        = var.collector_instance_profile_name
  key_name                    = null
  vpc_security_group_ids      = var.collector_security_group_ids
  user_data_replace_on_change = true
  user_data                   = local.collector_user_data

  instance_market_options {
    market_type = var.collector_spot_enabled ? "spot" : "on-demand"

    dynamic "spot_options" {
      for_each = var.collector_spot_enabled ? [1] : []
      content {
        instance_interruption_behavior = "stop"
        max_price                      = var.collector_spot_max_price != "" ? var.collector_spot_max_price : null
        spot_instance_type             = "persistent"
      }
    }
  }

  root_block_device {
    volume_size           = var.collector_root_volume_size
    volume_type           = "gp3"
    delete_on_termination = true
    encrypted             = true
  }

  tags = local.collector_tags
}
