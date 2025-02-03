# sources


## Getting Started

```sh
pip install -r requirements.txt

# Run the model builder
python model-builder.py --help

# Run the model builder with custom values
python model-builder.py \
  --dataset-root-path data \
  --model-root-path models \
  --model-name "pilot" \
  --model-version "1" \
  --learning-rate 0.001 \
  --momentum 0.9 \
  --batch-size 100 \
  --t-max 200 \
  --num-workers 0 \
  --num-epochs 3 \
  --train-fc-layer-only \
  --log-level INFO \

# Run the model builder with dry-run
python model-builder.py \
  --dataset-root-path data \
  --model-root-path models \
  --model-name "pilot" \
  --model-version "1" \
  --learning-rate 0.001 \
  --momentum 0.9 \
  --batch-size 100 \
  --t-max 200 \
  --train-fc-layer-only \
  --log-level DEBUG \
  --dry-run
```
