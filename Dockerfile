FROM continuumio/miniconda3

# Crear usuario y directorio de trabajo
WORKDIR /root/trypanodeepscreen

# Copiar archivos excluyendo los innecesarios con .dockerignore
COPY . ./

# Crear el entorno Conda y activarlo correctamente
RUN conda env create -f ./requirements_conda.yml && \
    conda clean --all -y

SHELL ["/bin/bash", "-c"]

# Usar conda run en lugar de activar el entorno manualmente
ENTRYPOINT ["conda", "run","-v", "-n", "trypanodeepscreen_enviroment","--no-capture-output", "python", "/root/trypanodeepscreen/ml/main.py"]
