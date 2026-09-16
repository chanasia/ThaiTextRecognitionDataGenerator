FROM python:3.12-slim

# opencv-python needs these two even without a display
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt pyarrow
COPY . .

# python -m trdg.corpus --count 200000 --out out/corpus.txt --charset_out out/charset.txt
# python trdg/run.py -i out/corpus.txt -f 64 -e png -t 8 -aug doc --output_dir out
ENTRYPOINT ["python"]
CMD ["trdg/run.py", "--help"]
