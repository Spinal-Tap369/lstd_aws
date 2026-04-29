Online time-series forecaster pipeline based on the LSTD paper for AWS deployment.

This repository contains components for:

- historical market data collection
- feature engineering and dataset preparation
- offline model training
- live inference and stateful adaptation
- AWS-oriented deployment and orchestration

lstd_aws/ —  forecaster modules
deploy_aws/ — infrastructure, instance setup, and service files

The project is designed to explore a production-style workflow for sequential forecasting systems, including training, artifact handoff, live state recovery, and streaming inference.