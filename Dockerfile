FROM pytorch/pytorch

ARG GID UID USER_NAME
RUN groupadd --gid $GID $USER_NAME
RUN useradd --gid $GID --create-home --shell /bin/bash --uid $UID $USER_NAME

RUN conda update -n base conda
COPY environment.yaml ./
ARG ENV_NAME
RUN conda env create -f environment.yaml -n $ENV_NAME
RUN rm environment.yaml
USER $UID:$GID
RUN conda init
RUN echo conda activate $ENV_NAME >> /home/$USER_NAME/.bashrc
