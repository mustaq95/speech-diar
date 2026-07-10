# NeMo Clustering Diarizer (MarbleNet VAD + TitaNet + spectral clustering)
# behind a thin FastAPI wrapper. GPU inference runs in its own container,
# outside the API/worker process -- the worker calls it over HTTP.
#
# No turnkey NIM exists for this cascaded pipeline (NIM only ships
# `diarizer=sortformer`), so this builds directly on the official NGC NeMo
# image, which already has the ASR/speaker_tasks collection and CUDA/PyTorch
# preinstalled.
FROM nvcr.io/nvidia/nemo:26.02

WORKDIR /app

RUN pip install --no-cache-dir "fastapi>=0.115" "uvicorn[standard]>=0.30" "python-multipart>=0.0.9"

COPY server.py diar_infer.yaml ./

EXPOSE 9020

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "9020"]
