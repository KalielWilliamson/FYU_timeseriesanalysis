FROM bitnami/jupyter-base-notebook:latest

USER root

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

USER 1001

RUN pip install wandb python-dotenv

COPY .env /workspace/.env

WORKDIR /workspace

RUN ls -la /workspace

EXPOSE 8888

CMD ["/entrypoint.sh"]
