#!/bin/bash
# API node bootstrap — Amazon Linux 2023
# Clones the repo, writes secrets, installs deps, and starts the FastAPI service.
# All template variables are injected by Terraform at apply time.
set -euo pipefail
exec > >(tee /var/log/stratum-bootstrap.log | logger -t stratum-api) 2>&1

echo "=== Stratum: API bootstrap starting ==="

# ---------------------------------------------------------------------------
# 1. System packages
# ---------------------------------------------------------------------------
# curl-minimal is already on AL2023 — installing curl conflicts with it
# python3.11-pip is not a valid AL2023 package; use ensurepip instead
dnf install -y git python3.11
python3.11 -m ensurepip --upgrade
echo "Python $(python3.11 --version) installed"

# ---------------------------------------------------------------------------
# 2. App user (home at /home/stratum — separate from the app directory)
# ---------------------------------------------------------------------------
useradd -r -m -s /bin/bash stratum || true

# ---------------------------------------------------------------------------
# 3. Clone the private repo using GitHub PAT, then strip token from remote
#    /opt/stratum must not exist before clone — git creates it
# ---------------------------------------------------------------------------
REPO_DIR="/opt/stratum"
rm -rf "$REPO_DIR"
git clone "${github_clone_url}" "$REPO_DIR"

# Strip the PAT so it is never stored in .git/config after clone
git -C "$REPO_DIR" remote set-url origin "${github_repo}"
chown -R stratum:stratum "$REPO_DIR"
echo "Repo cloned to $REPO_DIR"

# ---------------------------------------------------------------------------
# 4. Write .env — connection config + API keys (injected by Terraform)
# ---------------------------------------------------------------------------
cat > "$REPO_DIR/.env" << 'ENVEOF'
STRATUM_STORE_BACKEND=weaviate
STRATUM_EMBED_BACKEND=local
ENVEOF

# Append interpolated values separately to avoid Terraform/bash quoting issues
cat >> "$REPO_DIR/.env" << ENVEOF
STRATUM_WEAVIATE_HOST=${weaviate_host}
STRATUM_WEAVIATE_PORT=${weaviate_port}
STRATUM_ANTHROPIC_API_KEY=${anthropic_api_key}
STRATUM_OPENAI_API_KEY=${openai_api_key}
ENVEOF

chown stratum:stratum "$REPO_DIR/.env"
chmod 600 "$REPO_DIR/.env"
echo ".env written"

# ---------------------------------------------------------------------------
# 5. Create venv and install dependencies (api + ui extras)
# ---------------------------------------------------------------------------
sudo -u stratum python3.11 -m venv "$REPO_DIR/.venv"
sudo -u stratum "$REPO_DIR/.venv/bin/pip" install --quiet --upgrade pip
sudo mkdir -p /opt/pip-tmp && chmod 1777 /opt/pip-tmp
sudo -u stratum TMPDIR=/opt/pip-tmp "$REPO_DIR/.venv/bin/pip" install --no-cache-dir -e "$REPO_DIR[api,ui,local-embed]"
echo "Dependencies installed"

# ---------------------------------------------------------------------------
# 6. Systemd services — FastAPI and Streamlit UI
# ---------------------------------------------------------------------------
cat > /etc/systemd/system/stratum-api.service << 'UNIT'
[Unit]
Description=Stratum RAG FastAPI server
After=network-online.target
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
    --no-access-log
Restart=on-failure
RestartSec=15
StandardOutput=journal
StandardError=journal
SyslogIdentifier=stratum-api

[Install]
WantedBy=multi-user.target
UNIT

cat > /etc/systemd/system/stratum-ui.service << 'UNIT'
[Unit]
Description=Stratum RAG Streamlit UI
After=stratum-api.service
Wants=stratum-api.service

[Service]
User=stratum
Group=stratum
WorkingDirectory=/opt/stratum
EnvironmentFile=/opt/stratum/.env
Environment=STRATUM_API_URL=http://localhost:8000
ExecStart=/opt/stratum/.venv/bin/streamlit run app.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false
Restart=on-failure
RestartSec=15
StandardOutput=journal
StandardError=journal
SyslogIdentifier=stratum-ui

[Install]
WantedBy=multi-user.target
UNIT

systemctl daemon-reload
systemctl enable stratum-api stratum-ui

# ---------------------------------------------------------------------------
# 7. Wait for Weaviate to be ready before starting the API
#    (cross-encoder model also downloads on first startup — allow time)
# ---------------------------------------------------------------------------
echo "Waiting for Weaviate at ${weaviate_host}:${weaviate_port}..."
for i in $(seq 1 36); do
  if curl -sf "http://${weaviate_host}:${weaviate_port}/v1/.well-known/ready" > /dev/null 2>&1; then
    echo "Weaviate is ready after $((i * 10))s"
    break
  fi
  if [ "$i" -eq 36 ]; then
    echo "WARNING: Weaviate not ready after 360s — starting API anyway (systemd will retry)"
  else
    echo "Attempt $i/36 — retrying in 10s..."
    sleep 10
  fi
done

systemctl start stratum-api
echo "stratum-api.service started"

systemctl start stratum-ui
echo "stratum-ui.service started"

echo "=== Stratum: API bootstrap complete ==="
echo "Monitor API: journalctl -u stratum-api -f"
echo "Monitor UI:  journalctl -u stratum-ui -f"
