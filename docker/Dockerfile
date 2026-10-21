# ShelfSense AI edge image: dashboard + API + Qwen2.5-3B, CPU only, no internet needed at run time.
#
#   docker build -f docker/Dockerfile -t shelfsense .
#   docker compose -f docker/compose.yml up            (4 vCPU / 8 GB, like a Raspberry Pi 5)
#
# Needs the trained models in models/ (see README) when building.

# ---------------------------------------------------------------- llama.cpp server, built for this image's Linux
FROM python:3.11-slim-bookworm AS llama
ARG LLAMA_CPP_TAG=b11720
RUN apt-get update && apt-get install -y --no-install-recommends build-essential cmake git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN git clone --depth 1 --branch ${LLAMA_CPP_TAG} https://github.com/ggml-org/llama.cpp /src
# GGML_NATIVE=OFF + CPU_ALL_VARIANTS: one binary that picks the best kernels on whatever x86 CPU it lands on
RUN cmake -S /src -B /build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DGGML_BACKEND_DL=ON \
        -DGGML_CPU_ALL_VARIANTS=ON -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
        -DLLAMA_BUILD_SERVER=ON \
    && cmake --build /build --target llama-server -j"$(nproc)"
# the server plus its shared libraries and per-CPU backend plugins
RUN mkdir -p /out && cp /build/bin/llama-server /out/ \
    && find /build -name "*.so*" -exec cp -P {} /out/ \; && ls -la /out

# ---------------------------------------------------------------- runtime
FROM python:3.11-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    SHELFSENSE_OUTPUTS=/data/outputs SHELFSENSE_THREADS=4 LLAMA_SERVER=/usr/local/bin/llama-server
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 libglib2.0-0 tini \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-runtime.txt .
RUN pip install --no-cache-dir -r requirements-runtime.txt
# fetch RapidOCR's ONNX models now, so the container never needs the internet
RUN python -c "from rapidocr import RapidOCR; RapidOCR()"

COPY --from=llama /out/ /usr/local/bin/
# one thread per allowed CPU: the container sees every host core but may only use 4
ENV LD_LIBRARY_PATH=/usr/local/bin OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    SHELFSENSE_SPIN=0

COPY models/qwen2.5-3b-instruct-q4_k_m.gguf models/
COPY models/detector/shelfsense_yolo11n.onnx models/detector/
COPY models/dinov2-small-int8.onnx models/
COPY models/gallery/ models/gallery/
COPY assets/ assets/
COPY data/catalogue/ data/catalogue/
COPY data/planograms/ data/planograms/
COPY shelfsense/ shelfsense/
COPY docker/start.sh /start.sh
RUN chmod +x /start.sh && mkdir -p /data/outputs

EXPOSE 8501 8000
VOLUME /data
ENTRYPOINT ["tini", "--", "/start.sh"]
CMD ["serve"]
