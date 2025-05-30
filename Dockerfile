FROM rocm/tensorflow:rocm5.7-tf2.13-dev

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && apt install -y --no-install-recommends \
    python3-pip python3-dev python3-venv \
    build-essential \
    libbz2-dev liblzma-dev zlib1g-dev libdeflate-dev \
    libicu-dev libtirpc-dev libzstd-dev \
    libopenblas-dev \
    && apt clean && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml /app/pyproject.toml
WORKDIR /app

RUN pip install --upgrade pip \
 && pip install \
    tensorflow==2.13.0 \
    pandas>=1.5.0 \
    numpy>=1.23.0 \
    pydantic>=2.0.0 \
    pyyaml>=6.0.0 \
    pytz>=2023.3 \
    pyproj>=3.7.0 \
    pandas-stubs>=2.0.0 \
    types-pytz>=2025.1 \
    matplotlib>=3.7.0 \
    scikit-learn>=1.2.0 \
    seaborn>=0.12.0 \
    statsmodels>=0.14.4 \
    openpyxl>=3.1.5 \
    ephem>=4.2 \
    tabulate>=0.9.0 \
    scikit-optimize>=0.10.2


RUN useradd -m appuser
USER appuser
WORKDIR /home/appuser

ENTRYPOINT ["bash"]
