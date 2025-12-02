import requests, json, sys, socket, threading, re, time, os, traceback, datetime, random
from flask import Flask, render_template_string, redirect, request

CONFIG_FILE = "config.json"
BITMAP_DIR = "bitmaps"
BITMAP_FILE = os.path.join(BITMAP_DIR, "backup.json")
WIDTH, HEIGHT = 64, 16
PIXEL_COUNT = WIDTH * HEIGHT
# Randomizer settings (change these at top of the script)
RANDOMIZE_INTERVAL_SECONDS = 3600    # default: 1 hour
RANDOMIZE_PERCENT = 1                # default changed to 1% of pixels
verbose = "--verbose" in sys.argv
shutdown_event = threading.Event()

ERROR_LOG = "error.log"

# -------------------------
# Helpers for timestamped output
# -------------------------
from datetime import datetime
from zoneinfo import ZoneInfo
def now_ts():
    dt = datetime.now(ZoneInfo("America/New_York"))
    return dt.strftime("%Y-%m-%dT%H:%M:%S")

def log_print(msg):
    """
    Print a timestamped message to stdout. Use for all user-visible messages.
    """
    try:
        print(f"[{now_ts()}] {msg}")
    except Exception:
        # fallback to bare print if something goes wrong
        try:
            print(msg)
        except Exception:
            pass

# -------------------------
# Global runtime diagnostics
# -------------------------
last_twitch_line = None
last_twitch_time = None
last_twitch_username = None
last_pixel_command = None
last_http_request = None  # dict with url, payload (truncated), timestamp

# Randomizer enabled flag and lock (CLI/web toggle will change this)
randomizer_enabled = True
randomizer_lock = threading.Lock()

# Thread registry / factories so watchdog can restart named threads
thread_registry = {}     # name -> Thread instance
thread_factories = {}    # name -> callable returning target function (or the target function itself)

# Keep a small snapshot of bitmap for diagnostics (not full if huge)
def bitmap_snapshot():
    try:
        # show first 50 pixels and a count
        return {"first_pixels": bitmap[:50], "total_pixels": len(bitmap)}
    except Exception:
        return {"bitmap_available": False}

# -------------------------
# Error logging utilities
# -------------------------
def log_error(msg, exc=None, extra=None):
    """
    Append an entry to ERROR_LOG containing timestamp, thread name, message,
    optional stack trace and optional extra info (serialized).
    Also prints a short, timestamped notice to stdout.
    """
    try:
        ts = now_ts()
        thread_name = threading.current_thread().name
        with open(ERROR_LOG, "a") as f:
            f.write(f"[{ts}] {thread_name}: {msg}\n")
            if extra:
                try:
                    f.write("Extra: " + json.dumps(extra, default=str) + "\n")
                except Exception:
                    f.write("Extra (str): " + str(extra) + "\n")
            if exc:
                # exc is expected to be an exception object; include full traceback
                f.write(traceback.format_exc())
                f.write("\n")
            # add some runtime diagnostics
            try:
                f.write("Runtime diagnostics:\n")
                f.write(f"  last_twitch_time: {last_twitch_time}\n")
                f.write(f"  last_twitch_username: {last_twitch_username}\n")
                f.write(f"  last_twitch_line: {repr(last_twitch_line)[:1000]}\n")
                f.write(f"  last_pixel_command: {repr(last_pixel_command)[:1000]}\n")
                f.write(f"  last_http_request: {json.dumps(last_http_request, default=str) if last_http_request else None}\n")
                f.write(f"  bitmap_snapshot: {json.dumps(bitmap_snapshot(), default=str)}\n")
                f.write(f"  randomizer_enabled: {randomizer_enabled}\n")
                f.write(f"  RANDOMIZE_PERCENT: {RANDOMIZE_PERCENT}\n")
                f.write(f"  RANDOMIZE_INTERVAL_SECONDS: {RANDOMIZE_INTERVAL_SECONDS}\n")
            except Exception:
                f.write("Failed to write some runtime diagnostics\n")
            f.write("\n" + ("-" * 120) + "\n\n")
    except Exception as e:
        # If logging fails, at least print to stdout so operator knows
        try:
            log_print(f"Failed to write to error log: {e}")
        except Exception:
            pass
    # Also give a short notice on stdout so the operator sees something even when not verbose
    try:
        log_print(f"Error logged: {msg}")
    except Exception:
        pass

def run_with_catch(target, name):
    """
    Return a wrapper that runs target and logs any uncaught exception to error.log.
    Use this as thread target to capture all exceptions.
    """
    def wrapper(*args, **kwargs):
        try:
            target(*args, **kwargs)
        except Exception as e:
            log_error(f"Uncaught exception in thread '{name}': {e}", exc=e, extra={
                "thread_name": name
            })
            # Return to let watchdog restart the thread
    return wrapper

# Helper to create threads so watchdog can restart them
def make_thread(name, target_func):
    return threading.Thread(target=run_with_catch(target_func, name), name=name, daemon=True)

def register_thread(name, target_func):
    """
    Register a named thread factory and create/start its initial thread.
    target_func is the callable to run in the thread.
    """
    thread_factories[name] = target_func
    t = make_thread(name, target_func)
    thread_registry[name] = t
    t.start()
    if verbose:
        log_print(f"Started thread '{name}'")

# -------------------------
# Config and constants
# -------------------------
NAMED_COLORS = {
    "black":   "000000",
    "white":   "FFFFFF",
    "red":     "FF0000",
    "green":   "00FF00",
    "blue":    "0000FF",
    "yellow":  "FFFF00",
    "cyan":    "00FFFF",
    "magenta": "FF00FF",
    "purple":  "800080",
    "orange":  "FFA500",
    "pink":    "FFC0CB",
    "gray":    "808080",
    "off":     "000000",
    "grey":    "808080",
    "lime":    "BFFF00",
    "teal":    "008080",
    "navy":    "000080",
    "maroon":  "800000",
    "olive":   "808000",
    "aqua":    "00FFFF",
    "fuchsia": "FF00FF",
    "indigo":  "4B0082",
    "violet":  "EE82EE",
    "gold":    "FFD700",
    "silver":  "C0C0C0",
    "brown":   "CC9B65",
    "beige":   "F5F5DC",
    "coral":   "FF7F50",
    "salmon":  "FA8072",
    "chocolate": "D2691E",
    "crimson": "DC143C",
    "turquoise": "40E0D0",
    "skyblue": "87CEEB",
    "lavender": "E6E6FA",
    "mint":    "98FF98",
    "peach":   "FFDAB9",
    "hotpink": "FF69B4",
    "plum":    "DDA0DD",
    "orchid":  "DA70D6",
    "khaki":   "F0E68C"
}

# Create bitmaps directory if it doesn't exist
if not os.path.exists(BITMAP_DIR):
    os.makedirs(BITMAP_DIR)
    if verbose:
        log_print(f"Created {BITMAP_DIR} directory")

# Load or create config
def load_config():
    if not os.path.exists(CONFIG_FILE):
        default_file = "config_default.json"
        default = {
            "wled_ip": "192.168.1.1",
            "twitch_channel": "#your_channel",
            "twitch_nick": "justinfan12345",
            "twitch_oauth": "oauth:your_token",
            "wled_preset_name": "bitmap",
            "backup_interval_seconds": 300,
            "web_server_port": 9370,
            "enable_web_server": True
        }
        with open(default_file, "w") as f:
            f.write("// Rename this file to config.json and edit with your actual values\n")
            json.dump(default, f, indent=2)
        
        log_print("=" * 60)
        log_print("Configuration file not found!")
        log_print(f"Created {default_file} with default values.")
        log_print(f"Please rename {default_file} to config.json")
        log_print("and edit it with your actual WLED IP and Twitch credentials.")
        log_print("=" * 60)
        sys.exit(1)
    
    try:
        with open(CONFIG_FILE) as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        log_print(f"ERROR: Invalid JSON in {CONFIG_FILE}: {e}")
        log_print("Please fix the JSON syntax or delete the file to regenerate it.")
        sys.exit(1)
    except Exception as e:
        log_error("Failed to load config", exc=e)
        raise

config = load_config()
BASE_URL = f"http://{config['wled_ip']}"

# --------------
# HTTP wrappers
# --------------
def _record_http_request(url, payload):
    global last_http_request
    try:
        # Truncate payload for storage to avoid massive logs
        truncated_payload = None
        try:
            truncated_payload = json.dumps(payload)[:2000]
        except Exception:
            truncated_payload = str(payload)[:2000]
        last_http_request = {
            "url": url,
            "payload": truncated_payload,
            "time": time.time()
        }
    except Exception:
        pass

def safe_post(url, json_payload=None, timeout=10):
    _record_http_request(url, json_payload)
    try:
        resp = requests.post(url, json=json_payload, timeout=timeout)
        resp.raise_for_status()
        return resp
    except Exception as e:
        log_error(f"HTTP POST failed for {url}: {e}", exc=e, extra={"payload": json_payload})
        raise

def safe_get(url, timeout=10):
    _record_http_request(url, None)
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        return resp
    except Exception as e:
        log_error(f"HTTP GET failed for {url}: {e}", exc=e)
        raise

# -------------------------
# Bitmap management
# -------------------------
def load_bitmap():
    if os.path.exists(BITMAP_FILE):
        if verbose:
            log_print("Loaded existing bitmap from bitmaps/backup.json")
        with open(BITMAP_FILE) as f:
            return json.load(f)
    else:
        if verbose:
            log_print("No bitmaps/backup.json found – initializing blank bitmap")
        blank = ["000000"] * PIXEL_COUNT
        with open(BITMAP_FILE, "w") as f:
            json.dump(blank, f)
        return blank

def send_bitmap_fast():
    """Send the full bitmap to WLED in 256-pixel chunks."""
    CHUNK_SIZE = 256
    total = len(bitmap)

    for start in range(0, total, CHUNK_SIZE):
        chunk = bitmap[start:start + CHUNK_SIZE]

        payload = {
            "seg": [{
                "i": [start] + chunk
            }]
        }

        try:
            safe_post(f"{BASE_URL}/json/state", json_payload=payload)
            if verbose:
                log_print(f"Sent pixels {start}–{start + len(chunk) - 1} as bulk chunk")
        except Exception as e:
            # safe_post already logged; just print short notice
            log_print(f"Failed sending chunk starting at {start}: {e}")

def activate_bitmap_preset():
    preset_name = config.get('wled_preset_name', 'bitmap').strip().lower()
    try:
        resp = safe_get(f"{BASE_URL}/presets.json")
        presets = resp.json()

        for preset_id, preset_data in presets.items():
            if isinstance(preset_data, dict) and preset_data.get("n", "").strip().lower() == preset_name:
                safe_post(f"{BASE_URL}/json/state", json_payload={"ps": int(preset_id)})
                if verbose:
                    log_print(f"Activated WLED preset '{preset_name}' (ID {preset_id})")
                return
        log_print(f"Preset titled '{preset_name}' not found in presets.json")
    except Exception as e:
        log_error(f"Failed to activate '{preset_name}' preset: {e}", exc=e)

def save_named_bitmap(name):
    filename = os.path.join(BITMAP_DIR, f"{name}.json")

    # Convert flat 1D list → 2D grid
    grid = [bitmap[i:i+WIDTH] for i in range(0, PIXEL_COUNT, WIDTH)]

    try:
        with open(filename, "w") as f:
            json.dump(grid, f, indent=2)
        log_print(f"Saved current bitmap as grid to {filename}")
    except Exception as e:
        log_error(f"Failed to save {filename}: {e}", exc=e)

def load_named_bitmap(name):
    filename = os.path.join(BITMAP_DIR, f"{name}.json")
    if not os.path.exists(filename):
        log_print(f"File {filename} does not exist.")
        return

    try:
        with open(filename) as f:
            data = json.load(f)

        # Accept either 2D grid OR old-style 1D list
        if isinstance(data[0], list):  
            # Flatten 2D grid into 1D list
            loaded = [pixel for row in data for pixel in row]
        else:
            loaded = data

        if len(loaded) != PIXEL_COUNT:
            log_print(f"Error: {filename} does not contain a valid {PIXEL_COUNT}-pixel bitmap.")
            return

        # Copy into in-memory bitmap
        for i in range(PIXEL_COUNT):
            bitmap[i] = loaded[i]

        log_print(f"Loaded {filename}. Sending to WLED...")
        send_bitmap_fast()
        log_print("WLED updated with loaded bitmap.")

    except Exception as e:
        log_error(f"Failed to load {filename}: {e}", exc=e)

def restore_bitmap_to_display():
    log_print("Restoring full bitmap to display using fast upload...")
    try:
        send_bitmap_fast()
    except Exception as e:
        log_error("Failed to restore bitmap to display", exc=e)

bitmap = load_bitmap()

import hashlib

def save_bitmap_periodically():
    last_hash = None
    backup_interval = config.get('backup_interval_seconds', 300)
    while not shutdown_event.is_set():
        try:
            current_hash = hashlib.md5(json.dumps(bitmap).encode()).hexdigest()
            if current_hash != last_hash:
                with open(BITMAP_FILE, "w") as f:
                    json.dump(bitmap, f)
                last_hash = current_hash
                if verbose:
                    log_print("Bitmap changed – backed up to backup.json")
            elif verbose:
                log_print("Bitmap unchanged – no backup needed")
        except Exception as e:
            log_error("Failed to save bitmap", exc=e)
        time.sleep(backup_interval)

def xy_to_index(x, y):
    return (HEIGHT - y) * WIDTH + (x - 1)

def clean_hex(color_input):
    color_input = color_input.strip().lstrip('#').lower()
    if color_input in NAMED_COLORS:
        return NAMED_COLORS[color_input].upper()
    if len(color_input) == 6:
        try:
            int(color_input, 16)
            return color_input.upper()
        except ValueError:
            pass
    raise ValueError(f"Invalid color: '{color_input}' is not a valid hex or named color")

def send_pixel(index, hex_color):
    payload = {"seg": [{"i": [index, hex_color]}]}
    try:
        safe_post(f"{BASE_URL}/json/state", json_payload=payload)
        bitmap[index] = hex_color
        if verbose:
            log_print(f"Set pixel index {index} to HEX #{hex_color}")
    except Exception as e:
        log_error(f"Failed to update pixel index {index}", exc=e, extra={"index": index, "color": hex_color})
        log_print(f"Failed to update pixel index {index}: {e}")

def clear_board():
    payload = {"seg": [{"i": [0, PIXEL_COUNT, "000000"]}]}
    try:
        safe_post(f"{BASE_URL}/json/state", json_payload=payload)

        # ALSO clear internal memory to keep everything in sync
        for i in range(PIXEL_COUNT):
            bitmap[i] = "000000"

        if verbose:
            log_print("Cleared all pixels to HEX #000000 (display + internal bitmap)")
    except Exception as e:
        log_error("Failed to clear board", exc=e)
        log_print(f"Failed to clear board: {e}")

def parse_pixel_command(message):
    global last_pixel_command
    msg = message.strip()
    
    # Check if message starts with "Pixel" (case-insensitive)
    if not msg.lower().startswith("pixel"):
        return
    
    last_pixel_command = msg
    try:
        lowered = msg.lower()

        # Extract color (named or hex)
        color_match = re.search(r"#?[0-9a-f]{6}|\b(" + "|".join(NAMED_COLORS.keys()) + r")\b", lowered)
        if not color_match:
            if verbose:
                log_print("No valid color found in message.")
            return

        hex_color = clean_hex(color_match.group(0))

        # Extract all x,y coordinate pairs
        coords = re.findall(r"(\d+)\s*,\s*(\d+)", lowered)
        if not coords:
            if verbose:
                log_print("No valid coordinates found in message.")
            return

        # Build a batch payload for all valid pixels
        pixel_data = []
        queued_pixels = []
        for x_str, y_str in coords:
            x, y = int(x_str), int(y_str)
            if 1 <= x <= WIDTH and 1 <= y <= HEIGHT:
                index = xy_to_index(x, y)
                pixel_data.extend([index, hex_color])
                bitmap[index] = hex_color  # Update internal bitmap
                queued_pixels.append(f"({x},{y})")
            else:
                if verbose:
                    log_print(f"Coordinates out of bounds: ({x},{y})")

        # Log all queued pixels on one line (always print regardless of verbose)
        if queued_pixels:
            # Color info at the beginning as requested
            log_print(f"Color #{hex_color} - Queued {len(queued_pixels)} pixels: {', '.join(queued_pixels)}")

        # Send all pixels in one request
        if pixel_data:
            payload = {"seg": [{"i": pixel_data}]}
            # record in diagnostics
            _record_http_request(f"{BASE_URL}/json/state", payload)
            try:
                safe_post(f"{BASE_URL}/json/state", json_payload=payload)
                if verbose:
                    log_print(f"Sent {len(pixel_data)//2} pixels in one batch request")
            except Exception as e:
                # safe_post logged the error
                log_print(f"Failed to update pixels: {e}")
    except Exception as e:
        log_error("Exception while parsing pixel command", exc=e, extra={"message": message})
        if verbose:
            log_print(f"Exception while parsing pixel command: {e}")

def print_help():
    """
    Print available console commands.
    """
    log_print("Console commands:")
    log_print("  pixel clear            - Clear the display and in-memory bitmap")
    log_print("  save <name>            - Save current bitmap as bitmaps/<name>.json")
    log_print("  load <name>            - Load bitmap from bitmaps/<name>.json")
    log_print("  list                   - List all saved bitmaps in the bitmaps/ directory")
    log_print("  reset                  - Re-activate preset, clear, and restore backup.json")
    log_print("  random                 - Toggle random overlay")
    log_print("  random on|off|status   - Control random overlay explicitly")
    log_print("  help                   - Show this help")

def console_listener():
    """
    Console listens for commands via input(). Supports:
      - save <name>
      - load <name>
      - pixel clear
      - reset
      - random on|off|toggle|status
    If stdin is closed (EOFError), it logs and waits a bit, then continues — preventing a silent thread death.
    """
    while not shutdown_event.is_set():
        try:
            try:
                cmd = input().strip()
            except EOFError as e:
                # stdin closed (e.g., running as service); log and sleep, keep thread alive
                log_error("Console listener EOF (stdin closed), continuing and retrying", exc=e)
                time.sleep(5)
                continue

            if not cmd:
                continue

            lower = cmd.lower()
            if lower == "pixel clear":
                clear_board()
                log_print("Console: Display cleared.")

            elif lower.startswith("save "):
                name = cmd.split(" ", 1)[1].strip()
                if name:
                    save_named_bitmap(name)
                else:
                    log_print("Usage: save <name>")

            elif lower.startswith("load "):
                name = cmd.split(" ", 1)[1].strip()
                if name:
                    load_named_bitmap(name)
                else:
                    log_print("Usage: load <name>")

            elif lower == "list":
                try:
                    files = sorted(
                        f for f in os.listdir(BITMAP_DIR)
                        if f.endswith(".json")
                    )
                    if not files:
                        log_print(f"No bitmap files found in {BITMAP_DIR}")
                    else:
                        log_print(f"Bitmap files in {BITMAP_DIR}:")
                        for fname in files:
                            # Strip .json to show just the logical name
                            if fname.lower() == "backup.json":
                                continue  # keep backup out of the list if you prefer
                            base = fname[:-5] if fname.lower().endswith(".json") else fname
                            log_print(f"  {base}")
                except Exception as e:
                    log_error("Failed to list bitmaps", exc=e)

            elif lower == "help":
                print_help()

            elif lower == "reset":
                log_print("Resetting: activating preset, clearing display, restoring backup...")

                # 1. Activate WLED preset again
                try:
                    activate_bitmap_preset()
                except Exception as e:
                    log_error("activate_bitmap_preset failed during reset", exc=e)

                # 2. Clear board
                try:
                    clear_board()
                except Exception as e:
                    log_error("clear_board failed during reset", exc=e)

                # 3. Reload backup from disk
                global bitmap
                try:
                    bitmap = load_bitmap()  # reload backup.json
                except Exception as e:
                    log_error("load_bitmap failed during reset", exc=e)

                # 4. Push bitmap to WLED using bulk uploading
                try:
                    send_bitmap_fast()
                except Exception as e:
                    log_error("Failed to send bitmap during reset", exc=e)

                log_print("Reset complete.")

            elif lower.startswith("random"):
                parts = lower.split()
                if len(parts) == 1:
                    # toggle
                    with randomizer_lock:
                        global randomizer_enabled
                        randomizer_enabled = not randomizer_enabled
                        state = "enabled" if randomizer_enabled else "disabled"
                    log_print(f"Random overlay toggled → {state}")
                elif parts[1] in ("on", "enable", "enabled"):
                    with randomizer_lock:
                        randomizer_enabled = True
                    log_print("Random overlay enabled")
                elif parts[1] in ("off", "disable", "disabled"):
                    with randomizer_lock:
                        randomizer_enabled = False
                    log_print("Random overlay disabled")
                elif parts[1] == "status":
                    with randomizer_lock:
                        state = "enabled" if randomizer_enabled else "disabled"
                    log_print(f"Random overlay is currently {state}")
                else:
                    log_print("Usage: random on|off|toggle|status")

            else:
                log_print(f"Unknown console command: {cmd}")

        except Exception as e:
            log_error("Console listener error", exc=e)

def twitch_listener():
    """
    This version will reconnect on socket closure/exception. It will:
      - loop forever while not shutdown_event
      - try to connect and authenticate
      - read data, handle PRIVMSG
      - if the socket closes or any error occurs, close socket and retry after backoff
    """
    global last_twitch_line, last_twitch_time, last_twitch_username

    backoff = 1
    while not shutdown_event.is_set():
        sock = None
        try:
            sock = socket.socket()
            sock.settimeout(10.0)
            sock.connect(("irc.chat.twitch.tv", 6667))
            sock.settimeout(2.0)  # read timeout
            try:
                sock.send(f"PASS {config['twitch_oauth']}\r\n".encode("utf-8"))
                sock.send(f"NICK {config['twitch_nick']}\r\n".encode("utf-8"))
                sock.send(f"JOIN {config['twitch_channel']}\r\n".encode("utf-8"))
            except Exception as e:
                log_error("Failed to send initial Twitch IRC auth/join", exc=e)
                raise

            if verbose:
                log_print(f"Connected to Twitch chat at {config['twitch_channel']}")

            buffer = ""
            # reset backoff after a successful connection
            backoff = 1

            while not shutdown_event.is_set():
                try:
                    resp = sock.recv(4096).decode("utf-8", errors="replace")
                    if resp == "":
                        # socket closed, attempt reconnect
                        log_error("Empty response from Twitch IRC socket (socket closed) — will reconnect")
                        break

                    buffer += resp
                    # update last_twitch_time on any data
                    last_twitch_time = time.time()
                    # split into lines
                    lines = buffer.split("\r\n")
                    buffer = lines.pop()  # leftover partial line
                    for line in lines:
                        last_twitch_line = line
                        if line.startswith("PING"):
                            try:
                                sock.send("PONG :tmi.twitch.tv\r\n".encode("utf-8"))
                            except Exception as e:
                                log_error("Failed to respond to PING", exc=e, extra={"line": line})
                        else:
                            if "PRIVMSG" in line:
                                parts = line.split(":", 2)
                                if len(parts) == 3:
                                    prefix = parts[1].strip()
                                    # Extract username from prefix like "username!username@host"
                                    username = prefix.split("!", 1)[0] if "!" in prefix else prefix
                                    last_twitch_username = username
                                    user_msg = parts[2]
                                    # Always print username and the chat message (timestamped by log_print)
                                    log_print(f"{username}: {user_msg}")
                                    # update last_twitch_time for PRIVMSG specifically
                                    last_twitch_time = time.time()
                                    try:
                                        parse_pixel_command(user_msg)
                                    except Exception as e:
                                        log_error("parse_pixel_command raised", exc=e, extra={"user_msg": user_msg, "username": username})
                except socket.timeout:
                    # normal keepalive behavior
                    continue
                except Exception as e:
                    log_error("Twitch listener inner read error", exc=e)
                    # break to outer loop to reconnect
                    break

        except Exception as e:
            log_error("Twitch listener connection/setup error", exc=e)
        finally:
            try:
                if sock:
                    sock.close()
            except Exception:
                pass

        # if we reach here we will attempt to reconnect unless shutdown_event set
        if shutdown_event.is_set():
            break

        # exponential backoff with cap
        sleep_time = min(backoff, 60)
        log_error(f"Twitch listener disconnected — will retry in {sleep_time}s", extra={"backoff": backoff})
        time.sleep(sleep_time)
        backoff = min(backoff * 2, 60)

# ---------------------
# Randomizer thread
# ---------------------
def randomize_pixels_periodically():
    """
    Periodically replace a percentage of pixels with random colors.
    Uses RANDOMIZE_INTERVAL_SECONDS and RANDOMIZE_PERCENT set at top of file.

    NOTE: This will update the physical display via WLED but will NOT change the
    in-memory 'bitmap' list. This preserves saved/backup bitmaps without the
    random overlay. The overlay can be enabled/disabled via CLI or web.
    """
    try:
        interval = max(1, int(RANDOMIZE_INTERVAL_SECONDS))
    except Exception:
        interval = 3600
    while not shutdown_event.is_set():
        try:
            # Wait interval but wake early if shutdown_event set
            wait_until = time.time() + interval
            while time.time() < wait_until and not shutdown_event.is_set():
                time.sleep(1)
            if shutdown_event.is_set():
                break

            with randomizer_lock:
                enabled = randomizer_enabled

            if not enabled:
                if verbose:
                    log_print("Randomizer: currently disabled; skipping this cycle")
                continue

            # Determine number of pixels to change
            pct = float(RANDOMIZE_PERCENT)
            if pct <= 0:
                if verbose:
                    log_print("Randomizer: percentage <= 0, skipping this cycle")
                continue
            if pct >= 100:
                num_pixels = PIXEL_COUNT
            else:
                num_pixels = max(1, int((pct / 100.0) * PIXEL_COUNT))

            # Pick unique random indexes
            if num_pixels >= PIXEL_COUNT:
                indexes = list(range(PIXEL_COUNT))
            else:
                try:
                    indexes = random.sample(range(PIXEL_COUNT), k=num_pixels)
                except ValueError:
                    # fallback: pick without replacement by shuffle
                    all_idx = list(range(PIXEL_COUNT))
                    random.shuffle(all_idx)
                    indexes = all_idx[:num_pixels]

            # Build payload but DO NOT alter the in-memory bitmap
            pixel_data = []
            colors_used = {}
            for idx in indexes:
                color = f"{random.randint(0, 0xFFFFFF):06X}"
                # Do NOT write to bitmap[idx] here — display-only change
                pixel_data.extend([idx, color])
                colors_used[color] = colors_used.get(color, 0) + 1

            if pixel_data:
                payload = {"seg": [{"i": pixel_data}]}
                _record_http_request(f"{BASE_URL}/json/state", payload)
                try:
                    safe_post(f"{BASE_URL}/json/state", json_payload=payload)
                    # Always print a short notice so you can see activity even when not verbose
                    # Show the number of pixels changed and a sample color breakdown
                    sample_colors = ", ".join([f"#{c}x{n}" for c, n in list(colors_used.items())[:5]])
                    log_print(f"Randomizer: Changed {len(indexes)} pixels ({RANDOMIZE_PERCENT}%) - colors: {sample_colors}")
                except Exception as e:
                    log_error("Randomizer failed to POST pixel update", exc=e, extra={"num_pixels": len(indexes)})
        except Exception as e:
            log_error("Randomizer thread exception", exc=e)
            # avoid tight error loop
            time.sleep(5)

# ---------------------
# Simple watchdog thread with restart
# ---------------------
def watchdog():
    """
    Periodically check the registered threads in thread_factories.
    If any registered thread is not alive, attempt to restart it and log the restart.
    """
    while not shutdown_event.is_set():
        try:
            time.sleep(30)
            for name, factory in list(thread_factories.items()):
                t = thread_registry.get(name)
                if t is None or not t.is_alive():
                    log_error("Thread not alive — attempting restart", extra={"thread_name": name})
                    try:
                        new_t = make_thread(name, factory)
                        thread_registry[name] = new_t
                        new_t.start()
                        log_error("Thread restarted", extra={"thread_name": name})
                    except Exception as e:
                        log_error("Failed restarting thread", exc=e, extra={"thread_name": name})
        except Exception as e:
            log_error("Watchdog error", exc=e)

# -------------------------
# Flask web server UI + toggle
# -------------------------
app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>WLED Pixel Grid</title>
    <style>
        body {
            font-family: sans-serif;
            margin: 0;
            padding: 0;
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        h1 {
            margin-top: 20px;
        }
        form {
            margin: 10px;
        }
        .grid-container {
            display: flex;
            justify-content: center;
            margin: 20px;
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(64, 10px);
            gap: 1px;
        }
        .pixel {
            width: 10px;
            height: 10px;
            cursor: crosshair;
        }
        button {
            padding: 10px 20px;
            font-size: 16px;
        }
        .controls {
            display:flex;
            gap:10px;
            align-items:center;
        }
        .status {
            margin-left:10px;
            font-weight:bold;
        }
    </style>
</head>
<body>
    <h1>WLED Pixel Grid</h1>
    <div class="controls">
      <form action="/clear" method="post" style="display:inline;">
        <button type="submit">Clear Display</button>
      </form>

      <form action="/toggle_random" method="post" style="display:inline;">
        {% if randomizer_enabled %}
          <button type="submit">Disable Random Overlay</button>
        {% else %}
          <button type="submit">Enable Random Overlay</button>
        {% endif %}
      </form>

      <div class="status">Random overlay: {% if randomizer_enabled %}ENABLED{% else %}DISABLED{% endif %}</div>
    </div>

    <div class="grid-container">
        <div class="grid">
            {% for i in range(bitmap|length) %}
                {% set x = (i % 64) + 1 %}
                {% set y = 16 - (i // 64) %}
                <div class="pixel" 
                     style="background-color: #{{ bitmap[i] }}"
                     title="({{ x }},{{ y }})"></div>
            {% endfor %}
        </div>
    </div>
</body>
</html>
"""

@app.route("/")
def index():
    # pass the randomizer state into the template
    with randomizer_lock:
        enabled = randomizer_enabled
    return render_template_string(HTML_TEMPLATE, bitmap=bitmap, randomizer_enabled=enabled)

@app.route("/clear", methods=["POST"])
def clear_route():
    clear_board()
    return redirect("/")

@app.route("/toggle_random", methods=["POST"])
def toggle_random_route():
    global randomizer_enabled
    with randomizer_lock:
        randomizer_enabled = not randomizer_enabled
        state = "enabled" if randomizer_enabled else "disabled"
    log_print(f"Random overlay toggled via web → {state}")
    return redirect("/")

# -------------------------
# Main startup
# -------------------------
if __name__ == "__main__":
    try:
        activate_bitmap_preset()
    except Exception as e:
        log_error("activate_bitmap_preset failed at startup", exc=e)

    try:
        clear_board()
    except Exception as e:
        log_error("clear_board failed at startup", exc=e)

    try:
        bitmap = load_bitmap()
    except Exception as e:
        log_error("load_bitmap failed at startup", exc=e)

    try:
        restore_bitmap_to_display()
    except Exception as e:
        log_error("restore_bitmap_to_display failed at startup", exc=e)
    
    # Start threads via register_thread so watchdog can restart them
    try:
        register_thread("console_listener", console_listener)
        register_thread("twitch_listener", twitch_listener)
        register_thread("save_bitmap_periodically", save_bitmap_periodically)
        register_thread("watchdog", watchdog)
        register_thread("randomize_pixels_periodically", randomize_pixels_periodically)

        log_print("Pixel server running. Listening for Twitch chat commands...")
    except Exception as e:
        log_error("Failed to start threads", exc=e)
        raise

    if config.get('enable_web_server', True):
        def start_web_server():
            try:
                port = config.get('web_server_port', 9370)
                app.run(host="0.0.0.0", port=port)
            except Exception as e:
                log_error("Web server crashed", exc=e)
        register_thread("web_server", start_web_server)
        log_print(f"Web server started on port {config.get('web_server_port', 9370)}")
    else:
        log_print("Web server disabled in config")

    try:
        while not shutdown_event.is_set():
            time.sleep(1)
    except KeyboardInterrupt:
        log_print("Shutting down.")
        shutdown_event.set()
    except Exception as e:
        log_error("Main loop exception", exc=e)
        shutdown_event.set()