import curses
import json
import random
import time
import sys
import os
from collections import Counter

# --- Configuration ---
# You can change the bar character here if you like
BAR_CHAR = "█"
# The file to monitor, can be passed as a command-line argument
PROGRESS_FILE = "progress.jsonl"


def format_entry(entry, width):
    """Pretty-print a single JSONL entry for display, wrapping long lines."""
    lines = []
    # Set a reasonable max width for the key column
    key_col_width = 22

    def wrap(text, w):
        if w <= 0: return [text] # Avoid wrapping in tiny windows
        return [text[i:i+w] for i in range(0, len(text), w)]

    for key, value in entry.items():
        key_str = f"{key}:"
        val_str = str(value) if value is not None else 'None'
        
        # Calculate width available for the value string
        val_width = width - key_col_width - 1
        
        # Wrap the value string
        val_lines = wrap(val_str, val_width)
        
        # Add the first line with the key
        lines.append(f"{key_str:<{key_col_width}} {val_lines[0] if val_lines else ''}")
        
        # Add subsequent wrapped lines, indented
        for l in val_lines[1:]:
            lines.append(f"{'':<{key_col_width}} {l}")
            
    return lines

def analyze_file(path):
    """
    Analyzes the JSONL file to get stats and action score distribution.
    """
    entries = []
    successes = 0
    total = 0
    total_tokens = 0
    score_distribution = Counter()

    try:
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue # Skip corrupted lines

                entries.append(entry)
                total += 1
                
                if entry.get("status") == "success":
                    successes += 1
                    total_tokens += entry.get("tokens_used", 0)
                    score = entry.get("action_score", -1)
                    score_distribution[score] += 1
                else:
                    # Explicitly count errors under the -1 key for the histogram
                    score_distribution[-1] += 1

    except FileNotFoundError:
        # It's okay if the file doesn't exist yet
        pass
        
    return entries, successes, total, score_distribution, total_tokens

def draw_histogram(stdscr, y_start, x_start, width, distribution):
    """Draws the action score histogram."""
    # Define labels and order for the histogram
    score_labels = {
        -1: "Error/Parse",
        0: "0 (Static)",
        1: "1 (Ambient)",
        2: "2 (Low-Energy)",
        3: "3 (Moderate)",
        4: "4 (High-Energy)",
        5: "5 (Peak Action)",
    }
    
    # Get max count to scale the bars (avoid division by zero)
    max_count = max(distribution.values()) if distribution else 0
    
    # Set bar chart parameters
    label_width = 16
    bar_area_width = width - label_width - 10 # -10 for padding and count
    
    # Draw title
    stdscr.addstr(y_start, x_start, "Action Score Distribution", curses.color_pair(3) | curses.A_BOLD | curses.A_UNDERLINE)
    y_pos = y_start + 2

    for score, label in score_labels.items():
        if y_pos >= curses.LINES - 1: break # Don't draw off-screen

        count = distribution.get(score, 0)
        
        bar_len = 0
        if max_count > 0 and bar_area_width > 0:
            bar_len = int((count / max_count) * bar_area_width)
            
        bar_str = BAR_CHAR * bar_len
        
        # Use red for errors, green for scores
        bar_color = curses.color_pair(4) if score == -1 else curses.color_pair(1)
        
        line = f"{label:<{label_width}} [{count:<5}] "
        stdscr.addstr(y_pos, x_start, line)
        stdscr.addstr(y_pos, x_start + len(line), bar_str, bar_color)
        
        y_pos += 1
    return y_pos


def main(stdscr, filepath):
    # --- Curses Setup ---
    curses.curs_set(0) # Hide cursor
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_GREEN, -1)   # Success / Bar
    curses.init_pair(2, curses.COLOR_BLACK, -1)   # Normal text
    curses.init_pair(3, curses.COLOR_CYAN, -1)    # Labels
    curses.init_pair(4, curses.COLOR_RED, -1)     # Error
    curses.init_pair(5, curses.COLOR_YELLOW, -1)  # Info

    last_mtime = None
    # Initialize data holders
    entries, successes, total, score_dist, total_tokens = [], 0, 0, Counter(), 0

    while True:
        stdscr.erase()
        height, width = stdscr.getmaxyx()

        # --- Data Loading ---
        # Only re-analyze the file if it has been modified
        try:
            mtime = os.path.getmtime(filepath)
        except FileNotFoundError:
            mtime = None

        if mtime != last_mtime:
            entries, successes, total, score_dist, total_tokens = analyze_file(filepath)
            last_mtime = mtime

        # --- Drawing UI ---
        # Header
        stdscr.addstr(0, 2, f"Monitoring: {filepath}", curses.A_BOLD)
        stdscr.addstr(1, 2, "[Press 'q' to quit]", curses.A_DIM)
        stdscr.hline(2, 0, "-", width)
        y_pos = 4

        # Stats Section
        if total == 0:
            stdscr.addstr(y_pos, 4, "Waiting for data...", curses.color_pair(4) | curses.A_BOLD)
            y_pos += 2
        else:
            errors = total - successes
            avg_tokens = (total_tokens / successes) if successes > 0 else 0
            stdscr.addstr(y_pos, 4, f"Total Processed: {total}", curses.color_pair(2) | curses.A_BOLD)
            stdscr.addstr(y_pos + 1, 4, f"Successes:       {successes}", curses.color_pair(1))
            stdscr.addstr(y_pos + 2, 4, f"Errors:          {errors}", curses.color_pair(4))
            stdscr.addstr(y_pos + 3, 4, f"Avg Tokens/Req:  {avg_tokens:.1f}", curses.color_pair(5))
            y_pos += 5

        stdscr.hline(y_pos, 0, "-", width)
        y_pos += 2

        # Histogram Section
        y_pos = draw_histogram(stdscr, y_pos, 4, width, score_dist)
        y_pos += 2
        
        stdscr.hline(y_pos, 0, "-", width)
        y_pos += 1

        # Sample Entry Section
        stdscr.addstr(y_pos, 2, "Random Sample Entry:", curses.color_pair(3) | curses.A_BOLD | curses.A_UNDERLINE)
        y_pos += 2
        
        if entries:
            example_entry = random.choice(entries)
            formatted_lines = format_entry(example_entry, width-6)
            for line in formatted_lines:
                if y_pos < height - 1:
                    stdscr.addstr(y_pos, 6, line)
                    y_pos += 1
        else:
            stdscr.addstr(y_pos, 6, "(No entries yet)")

        stdscr.refresh()
        
        # Non-blocking input with a 2-second timeout
        stdscr.timeout(2000)
        key = stdscr.getch()
        if key == ord('q'):
            break

if __name__ == "__main__":
    filepath = sys.argv[1] if len(sys.argv) > 1 else PROGRESS_FILE
    if not os.path.isabs(filepath):
        filepath = os.path.join(os.getcwd(), filepath)

    print(f"Starting monitor for: {filepath}")
    print("Press 'q' in the monitor window to exit.")
    time.sleep(1)

    try:
        curses.wrapper(main, filepath)
    except curses.error as e:
        print(f"\nCurses error: {e}")
        print("Your terminal window might be too small to run the monitor.")
    except KeyboardInterrupt:
        print("\nMonitor stopped.")