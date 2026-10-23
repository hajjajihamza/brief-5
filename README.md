'''
mlflow server \
  --backend-store-uri sqlite:///mlflow/mlflow.db \
  --artifacts-destination ./mlflow/mlartifacts \
  --host 127.0.0.1 \
  --port 5000
'''