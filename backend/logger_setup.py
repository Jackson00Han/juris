import logging
import yaml

with open('config.yaml') as f:
    cfg = yaml.safe_load(f)

logging.basicConfig(
    level=getattr(logging, cfg['log_level']),
    format=cfg['log_format'],
)

logger = logging.getLogger("DanishLawRAG")

