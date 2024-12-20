# zata-agent

Fine-tuning LLM to impersonate influencer's tweeting styles.

![screenshot](./reports/screenshot.png)

## Requirements

- Python 3.12
- [uv](https://docs.astral.sh/uv/)

## How to Run

1. Install dependencies

```bash
$ make install
```

2. Run web app

```bash
$ username=trump
$ uv run zata serve --model /models/qwen2.5-0.5b-instruct_${username} --config ./config/${username}.toml
```

## Development

### Dataset preparation

```bash
# user to pull tweets from
$ username=Ashcryptoreal

# scrape tweets
$ zata scrape --site X --username ${username} --output .data/raw/${username}.parquet

# preprocess scraped tweets
$ zata data-prep --raw ./data/raw/${username}.parquet --output ./data/processed/${username}.parquet
```

### Fine-tuning model

```bash
$ zata train --output ./models/qwen2.5-0.5b-instruct_$username --data ./data/processed/$username.parquet --config ./config/trump.toml
```
