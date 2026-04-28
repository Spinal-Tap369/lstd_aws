#!/bin/bash
set -Eeuo pipefail

exec > >(tee /var/log/user-data-training.log | logger -t user-data -s 2>/dev/console) 2>&1

REPO_URL="https://github.com/Spinal-Tap369/lstd_aws.git"
REPO_BRANCH="main"
APP_ROOT="/opt/lstd_aws"
REPO_DIR="${APP_ROOT}/repo"
WORK_DIR="${APP_ROOT}/work"
VENV_DIR="/opt/lstd_aws-venv"
RUN_USER="ec2-user"
PYTHON_BIN="/usr/bin/python3.11"

log() {
  echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] $*"
}

retry() {
  local tries="$1"
  shift
  local attempt=1
  until "$@"; do
    if [ "$attempt" -ge "$tries" ]; then
      return 1
    fi
    log "command failed, retry ${attempt}/${tries}: $*"
    attempt=$((attempt + 1))
    sleep 5
  done
}

log "starting training instance bootstrap"

if command -v dnf >/dev/null 2>&1; then
  PKG_MGR="dnf"
else
  echo "dnf not found; this userdata expects Amazon Linux 2023" >&2
  exit 1
fi

log "installing system packages"
retry 3 sudo ${PKG_MGR} install -y \
  git \
  python3.11 \
  python3.11-pip \
  python3.11-setuptools \
  python3.11-wheel

log "verifying python version"
${PYTHON_BIN} - <<'PY'
import sys
assert sys.version_info >= (3, 10), sys.version
print(sys.version)
PY

log "creating application directories"
sudo mkdir -p "${WORK_DIR}/data" "${WORK_DIR}/outputs" "${WORK_DIR}/checkpoints" "${WORK_DIR}/scripts" "${WORK_DIR}/logs"
sudo chown -R ${RUN_USER}:${RUN_USER} "${APP_ROOT}"

if [ -d "${REPO_DIR}/.git" ]; then
  log "repo already exists; refreshing"
  sudo -u ${RUN_USER} git -C "${REPO_DIR}" fetch --all --prune
  sudo -u ${RUN_USER} git -C "${REPO_DIR}" checkout "${REPO_BRANCH}"
  sudo -u ${RUN_USER} git -C "${REPO_DIR}" pull --ff-only origin "${REPO_BRANCH}"
else
  log "cloning repo"
  retry 3 sudo -u ${RUN_USER} git clone --branch "${REPO_BRANCH}" "${REPO_URL}" "${REPO_DIR}"
fi

log "recreating venv with system site packages so DLAMI torch stays visible"
sudo rm -rf "${VENV_DIR}"
${PYTHON_BIN} -m venv --system-site-packages "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

log "upgrading pip tooling inside venv"
python -m pip install --upgrade pip setuptools wheel

if [ -f "${REPO_DIR}/requirements.txt" ]; then
  log "installing repo requirements"
  python -m pip install -r "${REPO_DIR}/requirements.txt"
fi

log "installing package"
python -m pip install -e "${REPO_DIR}"

log "running import checks"
python - <<'PY'
import sys
print('python_executable=', sys.executable)
import lstd_aws
print('lstd_aws=', lstd_aws.__file__)
try:
    import torch
    print('torch=', torch.__version__)
    print('cuda_available=', torch.cuda.is_available())
except Exception as exc:
    print('torch_import_failed=', repr(exc))
PY

log "writing helper activation file"
cat > /etc/profile.d/lstd_aws_training.sh <<EOF
export LSTD_AWS_APP_ROOT=${APP_ROOT}
export LSTD_AWS_WORK_DIR=${WORK_DIR}
source ${VENV_DIR}/bin/activate
cd ${WORK_DIR}
EOF

log "writing bootstrap summary"
cat > "${WORK_DIR}/BOOTSTRAP_OK.txt" <<EOF
bootstrap_completed_utc=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
repo_url=${REPO_URL}
repo_branch=${REPO_BRANCH}
repo_dir=${REPO_DIR}
work_dir=${WORK_DIR}
venv_dir=${VENV_DIR}
python_bin=${PYTHON_BIN}
EOF

sudo chown -R ${RUN_USER}:${RUN_USER} "${APP_ROOT}"

log "bootstrap finished successfully"