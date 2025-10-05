import logging
import json
import time
from pathlib import Path
def save_config(args, output_path):
    """Save config.json with merge information"""
    config_info = {
        **vars(args),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    config_path = Path(output_path) / args.method / args.run_name / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config_info, f, indent=2)

    logging.info(f"Saved config to {config_path}")
    
    
def setup_logging(output_path, method, run_name):
    """Setup logging to save logs to output_path/method/run_name/"""
    
    from rich.console import Console
    from rich.logging import RichHandler
    log_dir = Path(output_path) / method / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "merge.log"
    
    # Create rich console handler
    console = Console()
    rich_handler = RichHandler(
        console=console,
        show_time=True,
        show_path=False,
        rich_tracebacks=True,
        tracebacks_show_locals=True
    )
    rich_handler.setLevel(logging.INFO)

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        # format="[%(asctime)s]%(name)s: %(message)s",
        datefmt="%m-%d %H:%M:%S",
        handlers=[
            # logging.StreamHandler()
            rich_handler,
            logging.FileHandler(log_file),
        ]
    )
    return logging.getLogger(__name__)
