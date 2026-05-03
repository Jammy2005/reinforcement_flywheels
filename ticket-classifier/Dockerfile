FROM public.ecr.aws/lambda/python:3.11

# Copy requirements and install
COPY lambda_requirements.txt .
RUN pip install -r lambda_requirements.txt --target ${LAMBDA_TASK_ROOT}

# Copy the model cache
COPY model_cache ${LAMBDA_TASK_ROOT}/model_cache

# Copy the handler
COPY lambda_handler.py ${LAMBDA_TASK_ROOT}

# Tell Lambda which function to invoke
CMD ["lambda_handler.handler"]