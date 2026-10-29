FROM python:3.12-slim
# Build and review locally; runtime always uses --pull=never.
RUN apt-get update && apt-get install -y --no-install-recommends libegl1 libegl-mesa0 libgl1 libgl1-mesa-dri libopengl0 libosmesa6 && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/sim2sim
COPY pyproject.toml README.md ./
RUN python -c 'import subprocess,tomllib; deps=tomllib.load(open("pyproject.toml","rb"))["project"]["dependencies"]; subprocess.check_call(["pip","install","--no-cache-dir",*deps])'
COPY src ./src
RUN pip install --no-cache-dir --no-deps .
COPY examples/scripted_agent.py /opt/agent/scripted_agent.py
ENV MUJOCO_GL=osmesa PYTHONDONTWRITEBYTECODE=1
USER 65532:65532
WORKDIR /workspace
CMD ["python", "/opt/agent/scripted_agent.py"]
