FROM public.ecr.aws/lambda/python:3.12

COPY pyproject.toml README.md ${LAMBDA_TASK_ROOT}/
COPY src/ ${LAMBDA_TASK_ROOT}/src/

# Install CPU-only PyTorch first — the estimator never runs inference,
# it only needs vLLM's config/spec classes.  Avoids pulling ~4GB CUDA runtime.
RUN pip install --no-cache-dir \
        torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir "${LAMBDA_TASK_ROOT}[web,lambda]"

CMD ["memory_estimator.api.lambda_handler.handler"]
