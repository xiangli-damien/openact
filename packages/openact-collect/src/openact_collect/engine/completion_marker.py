import json
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any
class CompletionMarker:
    @staticmethod
    def mark_complete(
        run_dir: Path,
        stats: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        run_dir = Path(run_dir)
        completion_info = {
            "completed_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "hostname": socket.gethostname(),
            "run_dir": str(run_dir),
            "stats": stats or {},
        }
        success_file = run_dir / "_SUCCESS"
        with open(success_file, "w") as f:
            json.dump(completion_info, f, indent=2, default=str)
        return completion_info
    @staticmethod
    def check_complete(run_dir: Path) -> bool:
        return (Path(run_dir) / "_SUCCESS").exists()
