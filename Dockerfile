# syntax=docker/dockerfile:1
FROM condaforge/mambaforge:latest AS builder

WORKDIR /opt

RUN mamba create -y -n dedalus-env -c conda-forge \
    python=3.12 \
    dedalus \
    h5py \
    mpi4py \
    && mamba clean -afy


COPY pyproject.toml ./

RUN /opt/conda/envs/dedalus-env/bin/pip install uv && \
    uv pip install -r pyproject.toml


# RUN python3 -m venv /opt/firedrake-env && \
#     . /opt/firedrake-env/bin/activate && \
#     curl -O https://firedrakeproject.org/firedrake-install && \
#     python3 firedrake-install --disable-ssh


FROM condaforge/mambaforge:latest AS runtime

WORKDIR /app

COPY --from=builder /opt/conda/envs/dedalus-env \
                    /opt/conda/envs/dedalus-env

ENV PATH="/opt/conda/envs/dedalus-env:$PATH"



COPY . .

RUN pip install .

# ENV PATH="/opt/conda/bin:/opt/firedrake-env/bin:/venv/bin:$PATH"

ENTRYPOINT [ "physics-sim" ]