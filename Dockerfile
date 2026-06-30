FROM continuumio/miniconda3:4.8.2

WORKDIR /workspace/megan-on-USPTO
COPY env.yml /tmp/megan-env.yml
RUN conda env create -f /tmp/megan-env.yml && conda clean -afy
SHELL ["/bin/bash", "-lc"]
ENV PROJECT_ROOT=/workspace/megan-on-USPTO \
    DATA_DIR=/workspace/megan-on-USPTO/data \
    CONFIGS_DIR=/workspace/megan-on-USPTO/configs \
    LOGS_DIR=/workspace/megan-on-USPTO/logs \
    MODELS_DIR=/workspace/megan-on-USPTO/models \
    RANDOM_SEED=132435 \
    PYTHONPATH=/workspace/megan-on-USPTO \
    N_JOBS=4
COPY . .
RUN echo "conda activate megan" >> ~/.bashrc
CMD ["bash"]
