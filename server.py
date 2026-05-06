"""
Real-Time Neural Network Visualizer + Training Controller untuk PaddleOCR
WebSocket server + HTTP API + Log file watcher + Training Process Manager

Jalankan: python server.py [path_ke_log_file]
Default log: ./output/det_db_finetune/train.log
"""

import asyncio
import json
import os
import re
import signal
import subprocess
import sys
import threading
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse

try:
    import websockets
except ImportError:
    print("ERROR: websockets belum terinstall.")
    print("Jalankan: pip install websockets")
    sys.exit(1)

try:
    import yaml
except ImportError:
    print("ERROR: pyyaml belum terinstall.")
    print("Jalankan: pip install pyyaml")
    sys.exit(1)

# === Configuration ===
CONFIG_PATH = "./ocr_finetune_det.yml"
LOG_PATH = sys.argv[1] if len(sys.argv) > 1 else "./output/det_db_finetune/train.log"
WS_PORT = 8765
HTTP_PORT = 8080
MAX_HISTORY = 500
POLL_INTERVAL = 0.5
BASE_DIR = Path(__file__).parent.resolve()
TRAIN_SCRIPT = str(BASE_DIR / "PaddleOCR" / "tools" / "train.py")

# === Global State ===
clients = set()
history = []
file_position = 0
train_process = None
train_status = "idle"  # idle, running, stopping, error


# ============================================================
#  YAML Config Helpers
# ============================================================

def load_config() -> dict:
    """Load the YAML config file and return as dict."""
    config_file = BASE_DIR / CONFIG_PATH
    if not config_file.exists():
        return {}
    with open(config_file, 'r', encoding='utf-8') as f:
        # Use safe_load but handle YAML anchors by reading raw
        content = f.read()
    # Remove YAML anchors/aliases for JSON compatibility
    # Parse with yaml
    return yaml.safe_load(content) or {}


def save_config(data: dict):
    """Save dict back to YAML config file."""
    config_file = BASE_DIR / CONFIG_PATH
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def config_to_editable(cfg: dict) -> dict:
    """Extract editable fields from full config into a flat structure for the UI."""
    g = cfg.get('Global', {})
    opt = cfg.get('Optimizer', {})
    lr_cfg = opt.get('lr', {})
    pp = cfg.get('PostProcess', {})
    train = cfg.get('Train', {})
    train_ds = train.get('dataset', {})
    train_loader = train.get('loader', {})
    ev = cfg.get('Eval', {})
    eval_ds = ev.get('dataset', {})
    eval_loader = ev.get('loader', {})

    return {
        "epoch_num": g.get('epoch_num', 50),
        "learning_rate": lr_cfg.get('learning_rate', 0.001),
        "warmup_epoch": lr_cfg.get('warmup_epoch', 0),
        "batch_size_train": train_loader.get('batch_size_per_card', 4),
        "batch_size_eval": eval_loader.get('batch_size_per_card', 1),
        "num_workers": train_loader.get('num_workers', 4),
        "train_data_dir": train_ds.get('data_dir', ''),
        "train_label_file": (train_ds.get('label_file_list', ['']) or [''])[0],
        "eval_data_dir": eval_ds.get('data_dir', ''),
        "eval_label_file": (eval_ds.get('label_file_list', ['']) or [''])[0],
        "checkpoints": g.get('checkpoints') or '',
        "pretrained_model": g.get('pretrained_model', ''),
        "save_model_dir": g.get('save_model_dir', ''),
        "save_epoch_step": g.get('save_epoch_step', 10),
        "print_batch_step": g.get('print_batch_step', 10),
        "eval_batch_step_start": (g.get('eval_batch_step', [0, 100]) or [0, 100])[0],
        "eval_batch_step_interval": (g.get('eval_batch_step', [0, 100]) or [0, 100])[1] if len(g.get('eval_batch_step', [0, 100]) or [0, 100]) > 1 else 100,
        "box_thresh": pp.get('box_thresh', 0.6),
        "unclip_ratio": pp.get('unclip_ratio', 1.5),
        "use_gpu": g.get('use_gpu', True),
    }


def apply_editable(cfg: dict, editable: dict) -> dict:
    """Apply editable fields back into the full config dict."""
    if 'Global' not in cfg:
        cfg['Global'] = {}
    g = cfg['Global']
    g['epoch_num'] = int(editable.get('epoch_num', 50))
    g['checkpoints'] = editable.get('checkpoints', '') or None
    g['pretrained_model'] = editable.get('pretrained_model', '')
    g['save_model_dir'] = editable.get('save_model_dir', '')
    g['save_epoch_step'] = int(editable.get('save_epoch_step', 10))
    g['print_batch_step'] = int(editable.get('print_batch_step', 10))
    g['eval_batch_step'] = [
        int(editable.get('eval_batch_step_start', 0)),
        int(editable.get('eval_batch_step_interval', 100))
    ]
    g['use_gpu'] = bool(editable.get('use_gpu', True))

    if 'Optimizer' not in cfg:
        cfg['Optimizer'] = {}
    if 'lr' not in cfg['Optimizer']:
        cfg['Optimizer']['lr'] = {}
    cfg['Optimizer']['lr']['learning_rate'] = float(editable.get('learning_rate', 0.001))
    cfg['Optimizer']['lr']['warmup_epoch'] = int(editable.get('warmup_epoch', 0))

    if 'PostProcess' not in cfg:
        cfg['PostProcess'] = {}
    cfg['PostProcess']['box_thresh'] = float(editable.get('box_thresh', 0.6))
    cfg['PostProcess']['unclip_ratio'] = float(editable.get('unclip_ratio', 1.5))

    if 'Train' not in cfg:
        cfg['Train'] = {}
    if 'dataset' not in cfg['Train']:
        cfg['Train']['dataset'] = {}
    if 'loader' not in cfg['Train']:
        cfg['Train']['loader'] = {}
    cfg['Train']['dataset']['data_dir'] = editable.get('train_data_dir', '')
    cfg['Train']['dataset']['label_file_list'] = [editable.get('train_label_file', '')]
    cfg['Train']['loader']['batch_size_per_card'] = int(editable.get('batch_size_train', 4))
    cfg['Train']['loader']['num_workers'] = int(editable.get('num_workers', 4))

    if 'Eval' not in cfg:
        cfg['Eval'] = {}
    if 'dataset' not in cfg['Eval']:
        cfg['Eval']['dataset'] = {}
    if 'loader' not in cfg['Eval']:
        cfg['Eval']['loader'] = {}
    cfg['Eval']['dataset']['data_dir'] = editable.get('eval_data_dir', '')
    cfg['Eval']['dataset']['label_file_list'] = [editable.get('eval_label_file', '')]
    cfg['Eval']['loader']['batch_size_per_card'] = int(editable.get('batch_size_eval', 1))

    return cfg


# ============================================================
#  Training Process Manager
# ============================================================

def start_training():
    """Start the PaddleOCR training subprocess."""
    global train_process, train_status, file_position

    if train_process and train_process.poll() is None:
        return {"ok": False, "error": "Training sudah berjalan"}

    config_file = str(BASE_DIR / CONFIG_PATH)
    if not Path(TRAIN_SCRIPT).exists():
        return {"ok": False, "error": f"Train script tidak ditemukan: {TRAIN_SCRIPT}"}

    # Reset log watcher position to read new training output
    file_position = 0
    history.clear()

    cmd = [sys.executable, TRAIN_SCRIPT, "-c", config_file]
    print(f"[Train] Menjalankan: {' '.join(cmd)}")

    try:
        train_process = subprocess.Popen(
            cmd,
            cwd=str(BASE_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
        )
        train_status = "running"

        # Thread to pipe subprocess output to the log file
        def pipe_output():
            global train_status
            log_file = BASE_DIR / LOG_PATH
            log_file.parent.mkdir(parents=True, exist_ok=True)
            with open(log_file, 'a', encoding='utf-8') as lf:
                for line in train_process.stdout:
                    lf.write(line)
                    lf.flush()
                    print(f"[Train] {line.rstrip()}")
            retcode = train_process.wait()
            train_status = "idle" if retcode == 0 else "error"
            print(f"[Train] Selesai dengan kode: {retcode}")

        t = threading.Thread(target=pipe_output, daemon=True)
        t.start()
        return {"ok": True, "message": "Training dimulai"}
    except Exception as e:
        train_status = "error"
        return {"ok": False, "error": str(e)}


def stop_training():
    """Stop the training subprocess."""
    global train_process, train_status

    if not train_process or train_process.poll() is not None:
        train_status = "idle"
        return {"ok": False, "error": "Tidak ada training yang berjalan"}

    train_status = "stopping"
    print("[Train] Menghentikan training...")
    try:
        if os.name == 'nt':
            train_process.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            train_process.terminate()
        train_process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        train_process.kill()
    train_status = "idle"
    return {"ok": True, "message": "Training dihentikan"}


def get_train_status():
    """Get current training status."""
    global train_status
    if train_process and train_process.poll() is not None and train_status == "running":
        train_status = "idle"
    return {"status": train_status, "pid": train_process.pid if train_process and train_process.poll() is None else None}


# ============================================================
#  Directory Scanner
# ============================================================

def list_datasets():
    """List available dataset directories."""
    results = []
    for p in BASE_DIR.iterdir():
        if p.is_dir() and p.name.startswith('dataset'):
            # Check if it has train.txt or images/
            has_train = (p / 'train.txt').exists()
            has_valid = (p / 'valid.txt').exists()
            has_images = (p / 'images').is_dir() if (p / 'images').exists() else False
            results.append({
                "name": p.name,
                "path": f"./{p.name}/",
                "has_train": has_train,
                "has_valid": has_valid,
                "has_images": has_images
            })
    return results


def list_checkpoints():
    """List available checkpoint directories."""
    results = []
    # Check output dir
    output_dir = BASE_DIR / "output"
    if output_dir.exists():
        for p in output_dir.rglob("best_accuracy.pdparams"):
            rel = p.parent.relative_to(BASE_DIR)
            results.append({"path": f"./{rel}", "name": str(rel)})
        for p in output_dir.rglob("latest.pdparams"):
            rel = p.parent.relative_to(BASE_DIR)
            entry = {"path": f"./{rel}", "name": f"{rel} (latest)"}
            if entry not in results:
                results.append(entry)
    # Check other checkpoint dirs
    for d in BASE_DIR.iterdir():
        if d.is_dir() and ('cek_poin' in d.name or 'checkpoint' in d.name):
            for p in d.rglob("*.pdparams"):
                rel = p.parent.relative_to(BASE_DIR)
                name = f"./{rel}"
                if not any(r['path'] == name for r in results):
                    results.append({"path": name, "name": str(rel)})
    # Pretrained
    pretrained_dir = BASE_DIR / "pretrained"
    if pretrained_dir.exists():
        for p in pretrained_dir.iterdir():
            if p.is_dir():
                results.append({"path": f"./pretrained/{p.name}", "name": f"pretrained/{p.name}"})
    return results


# ============================================================
#  Log Parser
# ============================================================

def parse_log_line(line: str) -> dict | None:
    """Parse a PaddleOCR training log line."""
    if 'loss:' not in line:
        return None
    result = {}
    epoch_match = re.search(r'epoch:\s*\[(\d+)/(\d+)\]', line)
    if epoch_match:
        result['epoch'] = int(epoch_match.group(1))
        result['max_epoch'] = int(epoch_match.group(2))
    else:
        return None

    step_match = re.search(r'global_step:\s*(\d+)', line)
    if step_match:
        result['iter'] = int(step_match.group(1))
    iter_match = re.search(r'(?<!\w)iter:\s*(\d+)/(\d+)', line)
    if iter_match:
        result['iter'] = int(iter_match.group(1))
        result['max_iter'] = int(iter_match.group(2))
    if 'iter' not in result:
        return None

    lr_match = re.search(r'lr:\s*([\d.eE+-]+)', line)
    result['lr'] = float(lr_match.group(1)) if lr_match else 0.0
    loss_match = re.search(r'(?<![_\w])loss:\s*([\d.]+)', line)
    if loss_match:
        result['loss'] = float(loss_match.group(1))
    else:
        return None

    shrink_match = re.search(r'loss_shrink_maps:\s*([\d.]+)', line)
    result['loss_shrink'] = float(shrink_match.group(1)) if shrink_match else None
    threshold_match = re.search(r'loss_threshold_maps:\s*([\d.]+)', line)
    result['loss_threshold'] = float(threshold_match.group(1)) if threshold_match else None
    binary_match = re.search(r'loss_binary_maps:\s*([\d.]+)', line)
    result['loss_binary'] = float(binary_match.group(1)) if binary_match else None
    cbn_match = re.search(r'loss_cbn:\s*([\d.]+)', line)
    result['loss_cbn'] = float(cbn_match.group(1)) if cbn_match else None

    acc_match = re.search(r'(?<![_\w])acc:\s*([\d.]+)', line)
    result['acc'] = float(acc_match.group(1)) if acc_match else 0.0
    ned_match = re.search(r'norm_edit_dis:\s*([\d.]+)', line)
    result['norm_edit_dis'] = float(ned_match.group(1)) if ned_match else 0.0

    if 'max_iter' not in result:
        result['max_iter'] = 0
    eta_match = re.search(r'eta:\s*([\d\w\s:,]+?)(?:,\s*max_mem|$)', line)
    result['eta'] = eta_match.group(1).strip() if eta_match else ''
    ips_match = re.search(r'ips:\s*([\d.]+)', line)
    result['ips'] = float(ips_match.group(1)) if ips_match else 0.0
    mem_match = re.search(r'max_mem_reserved:\s*(\d+)\s*MB', line)
    result['gpu_mem'] = int(mem_match.group(1)) if mem_match else 0
    mem_alloc_match = re.search(r'max_mem_allocated:\s*(\d+)\s*MB', line)
    result['gpu_mem_alloc'] = int(mem_alloc_match.group(1)) if mem_alloc_match else 0
    reader_match = re.search(r'avg_reader_cost:\s*([\d.]+)', line)
    result['avg_reader_cost'] = float(reader_match.group(1)) if reader_match else 0.0
    batch_match = re.search(r'avg_batch_cost:\s*([\d.]+)', line)
    result['avg_batch_cost'] = float(batch_match.group(1)) if batch_match else 0.0
    samples_match = re.search(r'avg_samples:\s*([\d.]+)', line)
    result['avg_samples'] = float(samples_match.group(1)) if samples_match else 0.0

    result['timestamp'] = datetime.now().isoformat()
    result['type'] = 'training_update'
    return result


def parse_dataloader_info(line: str) -> int | None:
    match = re.search(r'train dataloader has (\d+) iters', line)
    return int(match.group(1)) if match else None


def parse_eval_progress(line: str) -> dict | None:
    """Parse eval progress bar line like: eval model:: 100%|###| 1785/1786 [06:25<00:00, 4.63it/s]"""
    match = re.search(r'eval model::\s*(\d+)%\|.*?\|\s*(\d+)/(\d+)\s*\[([^\]]+)\]', line)
    if not match:
        return None
    return {
        "type": "eval_progress",
        "percent": int(match.group(1)),
        "current": int(match.group(2)),
        "total": int(match.group(3)),
        "time_info": match.group(4).strip(),
        "timestamp": datetime.now().isoformat()
    }


def parse_eval_metrics(line: str) -> dict | None:
    """Parse cur metric line: precision, recall, hmean, fps."""
    if 'cur metric' not in line:
        return None
    result = {"type": "eval_result", "timestamp": datetime.now().isoformat()}
    for key in ['precision', 'recall', 'hmean', 'fps']:
        m = re.search(rf'{key}:\s*([\d.]+)', line)
        result[key] = float(m.group(1)) if m else 0.0
    return result


def parse_best_metric(line: str) -> dict | None:
    """Parse best metric line: hmean, precision, recall, fps, best_epoch."""
    if 'best metric' not in line:
        return None
    result = {"type": "best_metric", "timestamp": datetime.now().isoformat()}
    for key in ['hmean', 'precision', 'recall', 'fps']:
        m = re.search(rf'{key}:\s*([\d.]+)', line)
        result[key] = float(m.group(1)) if m else 0.0
    ep = re.search(r'best_epoch:\s*(\d+)', line)
    result['best_epoch'] = int(ep.group(1)) if ep else 0
    return result


# ============================================================
#  WebSocket
# ============================================================

async def log_watcher():
    global file_position
    log_path = Path(LOG_PATH)
    total_iters = 0
    last_size = 0
    print(f"[Log Watcher] Mengawasi: {log_path.resolve()}")

    while True:
        try:
            if not log_path.exists():
                msg = json.dumps({"type": "waiting", "message": f"Menunggu log: {log_path}..."})
                await broadcast(msg)
                file_position = 0
                await asyncio.sleep(2)
                continue

            current_size = log_path.stat().st_size
            if current_size < last_size:
                file_position = 0
                total_iters = 0
            last_size = current_size

            if current_size > file_position:
                with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                    f.seek(file_position)
                    chunk = f.read()
                    file_position = f.tell()
                
                if not chunk:
                    await asyncio.sleep(POLL_INTERVAL)
                    continue

                ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
                raw_lines = re.split(r'[\r\n]+', chunk)

                for line in raw_lines:
                    line = line.strip()
                    if not line:
                        continue
                    line = ansi_escape.sub('', line)
                    iters = parse_dataloader_info(line)
                    if iters:
                        total_iters = iters
                        continue

                    # Eval progress
                    eval_prog = parse_eval_progress(line)
                    if eval_prog:
                        await broadcast(json.dumps(eval_prog))
                        continue

                    # Eval result (cur metric)
                    eval_res = parse_eval_metrics(line)
                    if eval_res:
                        await broadcast(json.dumps(eval_res))
                        continue

                    # Best metric
                    best = parse_best_metric(line)
                    if best:
                        await broadcast(json.dumps(best))
                        continue

                    parsed = parse_log_line(line)
                    if parsed:
                        if total_iters > 0 and parsed.get('max_iter', 0) == 0:
                            parsed['max_iter'] = total_iters
                        history.append(parsed)
                        if len(history) > MAX_HISTORY:
                            history.pop(0)
                        await broadcast(json.dumps(parsed))
        except Exception as e:
            print(f"[Log Watcher] Error: {e}")
        await asyncio.sleep(POLL_INTERVAL)


async def broadcast(message: str):
    if not clients:
        return
    disconnected = set()
    for client in clients:
        try:
            await client.send(message)
        except Exception:
            disconnected.add(client)
    for c in disconnected:
        clients.discard(c)


async def ws_handler(websocket):
    clients.add(websocket)
    try:
        if history:
            await websocket.send(json.dumps({"type": "history", "data": history}))
        # Also send current train status
        await websocket.send(json.dumps({"type": "train_status", **get_train_status()}))
        async for message in websocket:
            pass
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        clients.discard(websocket)


# ============================================================
#  HTTP Server with API
# ============================================================

class APIHandler(BaseHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        self.static_dir = str(BASE_DIR)
        super().__init__(*args, **kwargs)

    def log_message(self, format, *args):
        pass

    def _send_json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', len(body))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> dict:
        length = int(self.headers.get('Content-Length', 0))
        raw = self.rfile.read(length)
        return json.loads(raw) if raw else {}

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_GET(self):
        path = urlparse(self.path).path

        if path == '/api/config':
            cfg = load_config()
            self._send_json(config_to_editable(cfg))
        elif path == '/api/datasets':
            self._send_json(list_datasets())
        elif path == '/api/checkpoints':
            self._send_json(list_checkpoints())
        elif path == '/api/train/status':
            self._send_json(get_train_status())
        else:
            # Serve static files
            if path == '/' or path == '':
                path = '/index.html'
            file_path = Path(self.static_dir) / path.lstrip('/')
            if file_path.exists() and file_path.is_file():
                self.send_response(200)
                ct = 'text/html' if path.endswith('.html') else \
                     'text/css' if path.endswith('.css') else \
                     'application/javascript' if path.endswith('.js') else \
                     'application/octet-stream'
                self.send_header('Content-Type', ct)
                content = file_path.read_bytes()
                self.send_header('Content-Length', len(content))
                self.end_headers()
                self.wfile.write(content)
            else:
                self.send_error(404)

    def do_POST(self):
        path = urlparse(self.path).path

        if path == '/api/config':
            try:
                editable = self._read_body()
                cfg = load_config()
                cfg = apply_editable(cfg, editable)
                save_config(cfg)
                self._send_json({"ok": True, "message": "Config disimpan"})
            except Exception as e:
                self._send_json({"ok": False, "error": str(e)}, 500)

        elif path == '/api/train/start':
            result = start_training()
            # Broadcast status update
            threading.Thread(target=lambda: _broadcast_train_status(), daemon=True).start()
            self._send_json(result)

        elif path == '/api/train/stop':
            result = stop_training()
            threading.Thread(target=lambda: _broadcast_train_status(), daemon=True).start()
            self._send_json(result)

        else:
            self.send_error(404)


def _broadcast_train_status():
    """Helper to broadcast train status via the async event loop."""
    import time
    time.sleep(0.5)
    status = get_train_status()
    msg = json.dumps({"type": "train_status", **status})
    loop = asyncio.get_event_loop()
    if loop.is_running():
        asyncio.run_coroutine_threadsafe(broadcast(msg), loop)

_event_loop = None

def start_http_server():
    server = HTTPServer(('0.0.0.0', HTTP_PORT), APIHandler)
    server.serve_forever()


# ============================================================
#  Main
# ============================================================

async def main():
    global _event_loop
    _event_loop = asyncio.get_event_loop()

    print("=" * 56)
    print("  PaddleOCR Training Visualizer + Controller")
    print("=" * 56)
    print()

    http_thread = threading.Thread(target=start_http_server, daemon=True)
    http_thread.start()
    print(f"  [HTTP] Server   : http://localhost:{HTTP_PORT}")

    ws_server = await websockets.serve(ws_handler, "0.0.0.0", WS_PORT)
    print(f"  [WS]  WebSocket : ws://localhost:{WS_PORT}")

    log_path = Path(LOG_PATH).resolve()
    print(f"  [LOG] File      : {log_path}")
    print(f"  [CFG] Config    : {(BASE_DIR / CONFIG_PATH).resolve()}")
    print()

    if not Path(LOG_PATH).exists():
        print(f"  [!] Log file belum ada. Mulai training dari web UI.")
    else:
        print(f"  [OK] Log file ditemukan.")

    print()
    print(f"  Buka browser: http://localhost:{HTTP_PORT}")
    print("  Ctrl+C untuk stop")
    print()

    asyncio.create_task(log_watcher())
    await asyncio.Future()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[Server] Dihentikan.")
