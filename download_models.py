import os
import curses
import sys
import shutil
from huggingface_hub import hf_hub_download

MODEL_FILES = [
    {
        "repo_id": "qingy2024/InternVideo2-B14",
        "path_in_repo": "internvideo2_vision.pt",
        "local_path": "./models/internvideo2_vision.pt",
    },
    {
        "repo_id": "qingy2024/InternVideo2-B14",
        "path_in_repo": "internvideo2_clip.pt",
        "local_path": "./models/internvideo2_clip.pt",
    },
    {
        "repo_id": "qingy2024/InternVideo2-B14",
        "path_in_repo": "mobileclip_blt.pt",
        "local_path": "./models/mobileclip_blt.pt",
    }
]

def curses_buttons(stdscr):
    curses.curs_set(0)
    stdscr.nodelay(0)
    stdscr.timeout(-1)

    options = ["Override", "Exit"]
    idx = 0

    while True:
        stdscr.erase()
        h, w = stdscr.getmaxyx()
        title = "The 'models' folder already exists."
        stdscr.addstr(h//2-2, (w-len(title))//2, title, curses.A_BOLD)
        for i, opt in enumerate(options):
            x = w//2 - 10 + i*15
            style = curses.A_REVERSE if i == idx else curses.A_NORMAL
            btn = f" {opt} "
            stdscr.addstr(h//2, x, btn, style)
        stdscr.addstr(h//2+3, (w-34)//2, "Use arrow keys or Tab to switch. Enter to select.")
        stdscr.refresh()
        c = stdscr.getch()
        if c in [curses.KEY_LEFT, ord('\t')]:
            idx = (idx - 1) % len(options)
        elif c in [curses.KEY_RIGHT]:
            idx = (idx + 1) % len(options)
        elif c in [10, 13, curses.KEY_ENTER]:
            return options[idx]
        elif c in [27]:  # Esc
            return "Exit"

def ensure_parent_folder(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

def download_models_curses(stdscr):
    curses.curs_set(0)
    stdscr.nodelay(0)
    stdscr.timeout(-1)
    h, w = stdscr.getmaxyx()

    total_files = len(MODEL_FILES)
    downloaded = 0
    for i, item in enumerate(MODEL_FILES):
        stdscr.erase()
        msg = f"Downloading {i+1}/{total_files}: {os.path.basename(item['path_in_repo'])}"
        stdscr.addstr(h//2-2, (w-len(msg))//2, msg, curses.A_BOLD)
        stdscr.refresh()
        # Ensure parent folder exists
        ensure_parent_folder(item['local_path'])
        try:
            hf_hub_download(
                repo_id=item['repo_id'],
                filename=item['path_in_repo'],
                local_dir=os.path.dirname(item['local_path']),
                local_dir_use_symlinks=False,
                cache_dir=None,
                resume_download=True,
                force_download=False,
            )
            # Move to exact local path if needed
            # (huggingface_hub will put file at models/stage1/B14/B14_dist_1B_stage2/pytorch_model.bin)
            # So this is only needed if any structure mismatch
        except Exception as e:
            stdscr.addstr(h//2, (w-40)//2, f"Error: {e}", curses.A_BOLD | curses.color_pair(1))
            stdscr.refresh()
            curses.napms(2000)
            return False

        downloaded += 1
        stdscr.addstr(h//2, (w-22)//2, "Download successful.", curses.A_DIM)
        stdscr.refresh()
        curses.napms(800)

    stdscr.erase()
    msg = "All models downloaded to './models'!"
    stdscr.addstr(h//2, (w-len(msg))//2, msg, curses.A_BOLD | curses.color_pair(2))
    stdscr.addstr(h//2+2, (w-34)//2, "Press any key to exit.")
    stdscr.refresh()
    stdscr.getch()
    return True

def main(stdscr):
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_RED, -1)
    curses.init_pair(2, curses.COLOR_GREEN, -1)

    if os.path.exists("models"):
        choice = curses_buttons(stdscr)
        if choice == "Override":
            shutil.rmtree("models")
            # Show progress while deleting, since it could take time on big folders
            stdscr.erase()
            msg = "Deleting existing 'models' folder..."
            h, w = stdscr.getmaxyx()
            stdscr.addstr(h//2, (w-len(msg))//2, msg)
            stdscr.refresh()
            curses.napms(1000)
            # proceed to download after
        else:
            stdscr.erase()
            bye = "Exit selected. Nothing changed."
            h, w = stdscr.getmaxyx()
            stdscr.addstr(h//2, (w-len(bye))//2, bye, curses.A_DIM)
            stdscr.refresh()
            curses.napms(1200)
            return

    # Download files if models/ doesn't exist (or after override)
    os.makedirs("models", exist_ok=True)
    download_models_curses(stdscr)

if __name__ == "__main__":
    curses.wrapper(main)
