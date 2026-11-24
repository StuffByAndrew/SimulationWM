# Two images from one recipe; build and review locally (scripts/build-image.slurm); runtime always uses --pull=never.
FROM python:3.12-slim AS base
RUN apt-get update && apt-get install -y --no-install-recommends libegl1 libegl-mesa0 libgl1 libgl1-mesa-dri libopengl0 libosmesa6 && rm -rf /var/lib/apt/lists/*
RUN --mount=type=bind,source=pyproject.toml,target=/tmp/pyproject.toml \
    python -c 'import subprocess,tomllib; deps=tomllib.load(open("/tmp/pyproject.toml","rb"))["project"]["dependencies"]; subprocess.check_call(["pip","install","--no-cache-dir",*deps])'
ENV MUJOCO_GL=osmesa PYTHONDONTWRITEBYTECODE=1

# Agent image (agent container, stream-mode predictors): only the agent-side modules. client (sandbox_agent.py, the channel
# tool), simulation + robots + candidate_worker (tools/local_controller), predictor + submission (stream-mode worker).
# Evaluation and metric code never enter it; the build inspection asserts this.
FROM base AS agent
COPY src/sim2sim/__init__.py src/sim2sim/client.py src/sim2sim/robots.py src/sim2sim/simulation.py src/sim2sim/candidate_worker.py \
     src/sim2sim/predictor.py src/sim2sim/submission.py /usr/local/lib/python3.12/site-packages/sim2sim/
COPY examples/sandbox_agent.py /opt/agent/sandbox_agent.py
LABEL sim2sim.image-role="agent"
USER 65532:65532
WORKDIR /workspace
CMD ["python", "/opt/agent/sandbox_agent.py"]

# Evaluator image (candidate worker, evaluator worker, call-mode predictors): the whole package.
FROM base AS evaluator
WORKDIR /opt/sim2sim
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir --no-deps .
LABEL sim2sim.image-role="evaluator"
USER 65532:65532
WORKDIR /workspace
