# ---------------------------------------------------------------------------
# S3 — raw document storage
#
# Bucket name uses a random 8-char hex suffix to guarantee global uniqueness
# without exposing the AWS account ID.
# ---------------------------------------------------------------------------

resource "random_id" "bucket_suffix" {
  byte_length = 4
}

locals {
  bucket_name = "${var.project_name}-${var.environment}-docs-${random_id.bucket_suffix.hex}"
}

resource "aws_s3_bucket" "documents" {
  bucket        = local.bucket_name
  force_destroy = false

  tags = { Name = local.bucket_name }
}

resource "aws_s3_bucket_versioning" "documents" {
  bucket = aws_s3_bucket.documents.id
  versioning_configuration { status = "Enabled" }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "documents" {
  bucket = aws_s3_bucket.documents.id
  rule {
    apply_server_side_encryption_by_default { sse_algorithm = "AES256" }
  }
}

resource "aws_s3_bucket_public_access_block" "documents" {
  bucket                  = aws_s3_bucket.documents.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}
