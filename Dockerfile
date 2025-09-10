# syntax=docker/dockerfile:1
# FROM mambaorg/micromamba:git-fb21d17-amazon2023 AS builder_dedalus
# WORKDIR /opt/project
# # Dedalus env


# RUN micromamba create -y -n dedalus-env -c conda-forge \
#     python=3.12 \
#     dedalus \
#     h5py \
#     mpi4py \
#     ffmpeg \
#     conda-pack \
#     && micromamba clean --all --yes

# SHELL ["micromamba", "run", "-n", "dedalus-env", "/bin/bash", "-c"]

# COPY pyproject.toml README.md .
# COPY src/ ./src/

# RUN pip install . \
#     && pip cache purge

# RUN conda-pack -n dedalus-env -o /tmp/env.tar.gz

# FROM mambaorg/micromamba:git-fb21d17-amazon2023 AS builder_fenics
# WORKDIR /opt/project
# # FEniCS env
# RUN mamba create -y -n fenics-env -c conda-forge \
#     # python=3.12 \
#     # fenics-dolfinx \
#     h5py \
#     mpich \
#     && mamba clean -afy

# COPY pyproject.toml README.md .
# COPY src/ ./src/

# RUN /opt/conda/envs/fenics-env/bin/pip install . \
#     && /opt/conda/envs/fenics-env/bin/pip cache purge

# FROM condaforge/mambaforge:latest AS runtime

# # WORKDIR /app

# # # COPY entrypoint.sh /usr/local/bin/entrypoint.sh
# # # RUN chmod +x /usr/local/bin/entrypoint.sh
# # # ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

# # # RUN touch /app/.last_env && chmod 666 /app/.last_env

# # COPY --from=builder_dedalus /opt/conda/envs/dedalus-env \
# #                             /opt/conda/envs/dedalus-env

# # # COPY --from=builder_fenics /opt/conda/envs/fenics-env \
# # #                            /opt/conda/envs/fenics-env


# # COPY . /app





# WORKDIR /app
# # Unpack the env
# COPY --from=builder_dedalus /tmp/env.tar.gz /tmp/env.tar.gz
# RUN mkdir /dedalus-env && tar -xzf /tmp/env.tar.gz -C /dedalus-env && rm /tmp/env.tar.gz \
#     && /dedalus-env/bin/conda-unpack


# # Copy your app
# COPY . /app

# # ENTRYPOINT ["conda", "run", "-n", "dedalus-env", "physics-sim"]
# # CMD ["physics-sim"]
# CMD ["conda", "run", "-n", "dedalus-env", "physics-sim"]






FROM mambaorg/micromamba:git-fb21d17-amazon2023 AS builder

WORKDIR /opt/project

RUN micromamba create -y -n sim-env -c conda-forge \
        python=3.12 \
        dedalus \
        h5py \
        mpi4py \
        ffmpeg \
        conda-pack \
        fenics-dolfinx \
        mpich \
    && micromamba clean --all --yes
SHELL ["micromamba", "run", "-n", "sim-env", "/bin/bash", "-c"]
COPY pyproject.toml README.md .
COPY src/ ./src/
RUN pip install . && pip cache purge


FROM mambaorg/micromamba:git-fb21d17-amazon2023 AS runtime
WORKDIR /app
COPY --from=builder /opt/conda/envs/sim-env /opt/conda/envs/sim-env
COPY . /app
CMD ["/opt/conda/envs/sim-env/bin/python", "-m", "physics-sim"]
