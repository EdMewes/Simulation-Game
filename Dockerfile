# syntax=docker/dockerfile:1
FROM condaforge/mambaforge:latest AS builder

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

RUN /opt/conda/envs/dedalus-env/bin/pip install . 

FROM condaforge/mambaforge:latest AS runtime

WORKDIR /app




COPY --from=builder /opt/conda/envs/dedalus-env \
                    /opt/conda/envs/dedalus-env
COPY --from=builder /opt/.venv /app/.venv

RUN mkdir /plots

ENV PATH="/opt/conda/envs/dedalus-env/bin:/app/.venv/bin:$PATH"

COPY . /app

# ENTRYPOINT ["conda", "run", "-n", "dedalus-env", "physics-sim"]
CMD ["conda", "run", "-n", "dedalus-env", "physics-sim"]
