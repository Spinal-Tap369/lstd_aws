#!/usr/bin/env bash
set -euo pipefail

APP_DIR=/opt/pipeline
ENV_DIR=/opt/pipeline-env
VENV_DIR=/opt/pipeline-venv
SERVICE_NAME=ws-collector.service
SERVICE_PATH="/etc/systemd/system/${SERVICE_NAME}"

sudo mkdir -p "${APP_DIR}" "${ENV_DIR}" "${VENV_DIR}"
sudo chown -R ec2-user:ec2-user "${APP_DIR}" "${ENV_DIR}" "${VENV_DIR}"

python3 -m venv "${VENV_DIR}"
"${VENV_DIR}/bin/pip" install --upgrade pip wheel
"${VENV_DIR}/bin/pip" install -r "${APP_DIR}/requirements-ws.txt"

sudo cp "${APP_DIR}/${SERVICE_NAME}" "${SERVICE_PATH}"
sudo systemctl daemon-reload
sudo systemctl enable "${SERVICE_NAME}"
sudo systemctl restart "${SERVICE_NAME}"
sudo systemctl status "${SERVICE_NAME}" --no-pager