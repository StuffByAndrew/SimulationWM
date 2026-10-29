FROM python:3.12-slim
# Build and review locally; runtime always uses --pull=never.
RUN apt-get update && apt-get install -y --no-install-recommends libegl1 libegl-mesa0 libgl1 libgl1-mesa-dri libopengl0 libosmesa6 && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/sim2sim
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .
COPY examples/scripted_agent.py /opt/agent/scripted_agent.py
ENV MUJOCO_GL=osmesa PYTHONDONTWRITEBYTECODE=1
USER 65532:65532
WORKDIR /workspace
CMD ["python", "/opt/agent/scripted_agent.py"]
