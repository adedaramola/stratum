#!/bin/bash
# API node bootstrap — Amazon Linux 2023
# Installs Python 3.11, creates the app user and directory, writes the systemd
# service, and templates the Weaviate connection config.
#
# Deploy flow after provisioning:
#   1. SSH into the instance
#   2. git clone your repo to /opt/stratum
#   3. Fill in API keys in /opt/stratum/.env
#   4. cd /opt/stratum && python3.11 -m venv .venv && .venv/bin/pip install -e ".[api]"
#   5. systemctl start stratum-api
set -euo pipefail
exec > >(tee /var/log/stratum-bootstrap.log | logger -t stratum-api) 2>&1

echo "=== Stratum: API bootstrap starting ==="

# ---------------------------------------------------------------------------
# 1. System packages
# ---------------------------------------------------------------------------
dnf install -y git python3.11 python3.11-pip
echo "Python $(python3.11 --version) installed"

# ---------------------------------------------------------------------------
# 2. App user and directory
# ---------------------------------------------------------------------------
useradd -r -m -d /opt/stratum -s /bin/bash stratum || true
mkdir -p /opt/stratum
chown stratum:stratum /opt/stratum

# ---------------------------------------------------------------------------
# 3. Write connection config (Weaviate private IP injected by Terraform)
# ---------------------------------------------------------------------------
cat > /opt/stratum/.env << EOF
# Stratum environment — populated by Terraform
STRATUM_STORE_BACKEND=weaviate
STRATUM_WEAVIATE_HOST=${weaviate_host}
STRATUM_WEAVIATE_PORT=${weaviate_port}

# Add your API keys (do NOT commit these):
STRATUM_ANTHROPIC_API_KEY=
STRATUM_OPENAI_API_KEY=
EOF
chown stratum:stratum /opt/stratum/.env
chmod 600 /opt/stratum/.env
echo "Connection config written to /opt/stratum/.env"

# ---------------------------------------------------------------------------
# 4. Systemd service for uvicorn
# ---------------------------------------------------------------------------
cat > /etc/systemd/system/stratum-api.service << 'UNIT'
[Unit]
Description=Stratum RAG FastAPI server
After=network.target
# Give Weaviate a moment to be reachable before the pipeline tries to connect
Wants=network-online.target

[Service]
User=stratum
Group=stratum
WorkingDirectory=/opt/stratum
EnvironmentFile=/opt/stratum/.env
ExecStart=/opt/stratum/.venv/bin/uvicorn rag.api.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 2 \
    --log-config /dev/null
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=stratum-api

[Install]
WantedBy=multi-user.target
UNIT

systemctl daemon-reload
systemctl enable stratum-api
# Service is enabled but NOT started — it needs the repo to be deployed first.
echo "stratum-api.service registered (start manually after code deployment)"

echo "=== Stratum: API bootstrap complete ==="
echo ""
echo "Next steps:"
echo "  1. SSH:   ssh -i ~/.ssh/${project_name}.pem ec2-user@$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4)"
echo "  2. Clone: git clone <your-repo> /opt/stratum && chown -R stratum:stratum /opt/stratum"
echo "  3. Keys:  edit /opt/stratum/.env and fill in ANTHROPIC and OPENAI keys"
echo "  4. Deps:  cd /opt/stratum && sudo -u stratum python3.11 -m venv .venv && sudo -u stratum .venv/bin/pip install -e '.[api]'"
echo "  5. Start: systemctl start stratum-api && journalctl -u stratum-api -f"
