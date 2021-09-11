FROM jupyter/scipy-notebook:ubuntu-18.04
ENV GRANT_SUDO yes
USER root

RUN pip install --upgrade pip

RUN pip install tensorflow==2.6.0
RUN pip install gym && pip install keras_metrics

EXPOSE 8888

ENTRYPOINT ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root"]