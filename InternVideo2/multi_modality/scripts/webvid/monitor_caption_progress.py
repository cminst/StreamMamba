import curses
import json
import random
import time
import sys
import os

def format_entry(entry, width):
    """Pretty-print a single JSONL entry for display."""
    lines = []
    def wrap(text, w):
        return [text[i:i+w] for i in range(0, len(text), w)]
    for key, value in entry.items():
        key_str = f"{key}:"
        val_str = str(value) if value is not None else 'None'
        # Handle newlines in strings
        if isinstance(value, str) and '\n' in value:
            val_lines = []
            for segment in value.split('\n'):
                val_lines += wrap(segment, width-len(key_str)-2)
            lines.append(f"{key_str:<20} {val_lines[0]}")
            for l in val_lines[1:]:
                lines.append(f"{'':<20} {l}")
        else:
            val_lines = wrap(val_str, width-len(key_str)-2)
            lines.append(f"{key_str:<20} {val_lines[0]}")
            for l in val_lines[1:]:
                lines.append(f"{'':<20} {l}")
    return lines

def analyze_file(path):
    entries = []
    successes = 0
    total = 0
    # Robust: handle truncated lines etc
    try:
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except Exception:
                    continue
                entries.append(entry)
                total += 1
                if entry.get("status", "") == "success":
                    successes += 1
    except FileNotFoundError:
        # For the startup—no file yet, just wait.
        pass
    return entries, successes, total

def main(stdscr, filepath):
    curses.curs_set(0)
    curses.use_default_colors()
    # Define colors
    curses.init_pair(1, curses.COLOR_GREEN, -1)  # Success bar
    curses.init_pair(2, curses.COLOR_WHITE, -1)  # Normal text
    curses.init_pair(3, curses.COLOR_CYAN, -1)   # Example label
    curses.init_pair(4, curses.COLOR_RED, -1)    # Error

    example_entry = None
    last_mtime = None

    while True:
        stdscr.erase()
        height, width = stdscr.getmaxyx()

        # Check if file changed, only re-analyze if needed
        try:
            mtime = os.path.getmtime(filepath)
        except Exception:
            mtime = None

        if mtime != last_mtime:
            entries, successes, total = analyze_file(filepath)
            last_mtime = mtime
            if entries:
                example_entry = random.choice(entries)
            else:
                example_entry = None

        # Draw headline
        stdscr.addstr(0, 2, f"Monitoring: {filepath}", curses.A_BOLD)
        stdscr.addstr(1, 2, "[Press 'q' to quit]", curses.A_DIM)
        stdscr.hline(2, 0, "-", width)

        # Draw stats
        if total == 0:
            stdscr.addstr(4, 4, "Waiting for data...", curses.color_pair(4) | curses.A_BOLD)
        else:
            percent = 100.0 * successes / total
            bar_width = width - 16
            fill = int(bar_width * percent / 100)
            bar = "[" + "#" * fill + "-" * (bar_width - fill) + "]"
            stdscr.addstr(4, 4, f"Success: {successes}/{total} ({percent:.1f}%)", curses.color_pair(2) | curses.A_BOLD)
            stdscr.addstr(5, 4, bar, curses.color_pair(1))

        # Draw an example entry
        stdscr.hline(7, 0, "-", width)
        stdscr.addstr(8, 2, "Sample Entry:", curses.color_pair(3) | curses.A_BOLD | curses.A_UNDERLINE)
        if example_entry:
            formatted = format_entry(example_entry, width-6)
            for idx, line in enumerate(formatted):
                if 9+idx < height-1:
                    stdscr.addstr(9+idx, 6, line)
        else:
            stdscr.addstr(10, 6, "(No entries yet)")

        stdscr.refresh()
        # Non-blocking quit
        stdscr.timeout(2000)  # ms
        c = stdscr.getch()
        if c == ord('q'):
            break

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 monitor_caption_progress.py <path_to_jsonl_file>")
        sys.exit(1)
    curses.wrapper(main, sys.argv[1])
