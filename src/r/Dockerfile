FROM rocker/r-ver:4.6.1

WORKDIR /app

# install packages first so code changes don't redo this step
COPY install_packages.R .
RUN Rscript install_packages.R

COPY titanic.R .

CMD ["Rscript", "titanic.R"]
