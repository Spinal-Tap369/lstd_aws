#!/bin/bash
set -euxo pipefail

APP_DIR="/opt/pipeline"
ENV_DIR="/opt/pipeline-env"
VENV_DIR="/opt/pipeline-venv"
APP_USER="ec2-user"
APP_GROUP="ec2-user"

BOOTSTRAP_BUCKET="my-lstd-data"
BOOTSTRAP_PREFIX="websocket-scripts"

# -------- basic dirs --------
mkdir -p "${APP_DIR}"
mkdir -p "${ENV_DIR}"
mkdir -p "${VENV_DIR}"

chown -R "${APP_USER}:${APP_GROUP}" "${APP_DIR}"
chown -R "${APP_USER}:${APP_GROUP}" "${ENV_DIR}"
chown -R "${APP_USER}:${APP_GROUP}" "${VENV_DIR}"

chmod 755 "${APP_DIR}"
chmod 755 "${ENV_DIR}"
chmod 755 "${VENV_DIR}"

# -------- ensure aws cli exists --------
if ! command -v aws >/dev/null 2>&1; then
  echo "aws CLI not found"
  exit 1
fi

# -------- pull files from S3 --------
aws s3 cp "s3://${BOOTSTRAP_BUCKET}/${BOOTSTRAP_PREFIX}/install_ws_collector.sh" "${APP_DIR}/install_ws_collector.sh"
aws s3 cp "s3://${BOOTSTRAP_BUCKET}/${BOOTSTRAP_PREFIX}/requirements-ws.txt" "${APP_DIR}/requirements-ws.txt"
aws s3 cp "s3://${BOOTSTRAP_BUCKET}/${BOOTSTRAP_PREFIX}/ws-collector.service" "${APP_DIR}/ws-collector.service"
aws s3 cp "s3://${BOOTSTRAP_BUCKET}/${BOOTSTRAP_PREFIX}/ws_collector.py" "${APP_DIR}/ws_collector.py"
aws s3 cp "s3://${BOOTSTRAP_BUCKET}/${BOOTSTRAP_PREFIX}/ws_collector.env" "${ENV_DIR}/ws_collector.env"

# -------- ownership --------
chown "${APP_USER}:${APP_GROUP}" "${APP_DIR}/install_ws_collector.sh"
chown "${APP_USER}:${APP_GROUP}" "${APP_DIR}/requirements-ws.txt"
chown "${APP_USER}:${APP_GROUP}" "${APP_DIR}/ws-collector.service"
chown "${APP_USER}:${APP_GROUP}" "${APP_DIR}/ws_collector.py"
chown "${APP_USER}:${APP_GROUP}" "${ENV_DIR}/ws_collector.env"

# -------- permissions --------
chmod 755 "${APP_DIR}/install_ws_collector.sh"
chmod 644 "${APP_DIR}/requirements-ws.txt"
chmod 644 "${APP_DIR}/ws-collector.service"
chmod 644 "${APP_DIR}/ws_collector.py"
chmod 600 "${ENV_DIR}/ws_collector.env"

# -------- optional sanity log --------
echo "Bootstrap complete at $(date -u)" > /var/log/ws-collector-bootstrap.log
ls -ld "${APP_DIR}" >> /var/log/ws-collector-bootstrap.log
ls -ld "${ENV_DIR}" >> /var/log/ws-collector-bootstrap.log
ls -ld "${VENV_DIR}" >> /var/log/ws-collector-bootstrap.log
ls -l "${APP_DIR}" >> /var/log/ws-collector-bootstrap.log
ls -l "${ENV_DIR}" >> /var/log/ws-collector-bootstrap.log