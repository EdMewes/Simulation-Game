# syntax=docker/dockerfile:1
FROM condaforge/mambaforge:latest AS builder_dedalus

WORKDIR /opt

# Dedalus env
RUN mamba create -y -n dedalus-env -c conda-forge \
    python=3.12 \
    dedalus \
    h5py \
    mpi4py \
    ffmpeg \
    && mamba clean -afy
RUN conda env config vars set -n dedalus-env OMP_NUM_THREADS=1 && \
    conda env config vars set -n dedalus-env NUMEXPR_MAX_THREADS=1



COPY pyproject.toml .
COPY . .

RUN /opt/conda/envs/dedalus-env/bin/pip install . 


FROM condaforge/mambaforge:latest AS builder_fenics

# FEniCS env
RUN mamba create -y -n fenics-env -c conda-forge \
    python=3.12 \
    fenics-dolfinx \
    matplotlib \
    pyvista \
    scipy \
    h5py \
    mpich \
    && mamba clean -afy

COPY pyproject.toml .
COPY . .

RUN /opt/conda/envs/fenics-env/bin/pip install . 




FROM condaforge/mambaforge:latest AS runtime

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]


WORKDIR /app

RUN touch /app/.last_env && chmod 666 /app/.last_env

COPY --from=builder_dedalus /opt/conda/envs/dedalus-env \
                    /opt/conda/envs/dedalus-env

COPY --from=builder_fenics /opt/conda/envs/fenics-env \
                    /opt/conda/envs/fenics-env


ENV PATH="/opt/conda/envs/dedalus-env/bin:/opt/conda/envs/fenics-env/bin:/app/.venv/bin:$PATH"

COPY . /app

# ENTRYPOINT ["conda", "run", "-n", "dedalus-env", "physics-sim"]
CMD ["conda", "run", "-n", "dedalus-env", "physics-sim"]
