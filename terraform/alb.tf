# ---------------------------------------------------------------------------
# Application Load Balancer
#
# Spans both public subnets for AZ redundancy.
# Two listeners:
#   :80   → API target group  (port 8000) — FastAPI
#   :8501 → UI  target group  (port 8501) — Streamlit chat interface
#
# Health checks:
#   API: GET /health  (FastAPI liveness probe)
#   UI:  GET /_stcore/health  (Streamlit built-in health endpoint)
#
# To add HTTPS: create an ACM certificate, add an aws_lb_listener on port 443,
# and add an HTTP→HTTPS redirect on port 80.
# ---------------------------------------------------------------------------

resource "aws_lb" "main" {
  name               = "${var.project_name}-${var.environment}-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets            = aws_subnet.public[*].id

  enable_deletion_protection = false

  tags = { Name = "${var.project_name}-${var.environment}-alb" }
}

resource "aws_lb_target_group" "api" {
  name     = "${var.project_name}-${var.environment}-api-tg"
  port     = 8000
  protocol = "HTTP"
  vpc_id   = aws_vpc.main.id

  health_check {
    path                = "/health"
    interval            = 30
    timeout             = 5
    healthy_threshold   = 2
    unhealthy_threshold = 3
    matcher             = "200"
  }

  tags = { Name = "${var.project_name}-${var.environment}-api-tg" }
}

resource "aws_lb_target_group_attachment" "api" {
  target_group_arn = aws_lb_target_group.api.arn
  target_id        = aws_instance.api.id
  port             = 8000
}

resource "aws_lb_listener" "http" {
  load_balancer_arn = aws_lb.main.arn
  port              = 80
  protocol          = "HTTP"

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.api.arn
  }
}

# ---------------------------------------------------------------------------
# Streamlit UI — target group + listener on port 8501
# ---------------------------------------------------------------------------

resource "aws_lb_target_group" "ui" {
  name     = "${var.project_name}-${var.environment}-ui-tg"
  port     = 8501
  protocol = "HTTP"
  vpc_id   = aws_vpc.main.id

  health_check {
    path                = "/_stcore/health"
    interval            = 30
    timeout             = 5
    healthy_threshold   = 2
    unhealthy_threshold = 3
    matcher             = "200"
  }

  tags = { Name = "${var.project_name}-${var.environment}-ui-tg" }
}

resource "aws_lb_target_group_attachment" "ui" {
  target_group_arn = aws_lb_target_group.ui.arn
  target_id        = aws_instance.api.id
  port             = 8501
}

resource "aws_lb_listener" "ui" {
  load_balancer_arn = aws_lb.main.arn
  port              = 8501
  protocol          = "HTTP"

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.ui.arn
  }
}
