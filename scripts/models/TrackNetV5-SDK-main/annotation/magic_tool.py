"""
    Outil Magique 🧙‍♂️🪄
"""

import numpy as np
import cv2
import json
import argparse
import os
import time
from pathlib import Path
import traceback
from spash_utils.video_reader import OpenCV_VideoReader, M3u8VideoReader
from spash_utils.data_alias import VideoAliasPathAction
from spash_utils.ball_types import (
    NON_SPECIFIED_KEY,
    SERVICE_KEY,
    BALL_EXCHANGE_KEY,
    FOREHAND_KEY,
    BACKHAND_KEY,   
    GROUND_REBOUND_KEY,
    GLASS_REBOUND_KEY,
    FOREHAND_VOLLEY_KEY,
    BACKHAND_VOLLEY_KEY,
    BANDEJA_KEY,
    FLAT_SMASH_KEY,
    KICK_SMASH_KEY,
    GANCHO_KEY,
    BAJADA_FOREHAND_KEY,
    BAJADA_BACKHAND_KEY,
    NET_KEY,
    NET_TOUCH_KEY,
    SERVICE_HIT_VOLLEY_SMASH_IDS,
    IS_LOB_KEY,
    IS_DIAGONAL_KEY,
    BALL_TYPE_TO_ID,
    BALL_TYPES,
)


BALL_TYPE_TO_KEY = {
    NON_SPECIFIED_KEY: "z",

    SERVICE_KEY: "e",
    BALL_EXCHANGE_KEY: "E",

    FOREHAND_KEY: "r",
    BACKHAND_KEY: "R",

    GROUND_REBOUND_KEY: "t",
    GLASS_REBOUND_KEY: "T",

    FOREHAND_VOLLEY_KEY: "y",
    BACKHAND_VOLLEY_KEY: "Y",

    BANDEJA_KEY: "u",
    FLAT_SMASH_KEY: "U",

    KICK_SMASH_KEY: "i",
    GANCHO_KEY: "I",

    BAJADA_FOREHAND_KEY: "o",
    BAJADA_BACKHAND_KEY: "O",

    NET_KEY: "p",
    NET_TOUCH_KEY: "P",
}

BALL_TYPES_COLOR = {
    0: (0, 0, 255),  # noir
    1: (255, 0, 0),  # bleu
    2: (0, 255, 0),  # vert
    3: (0, 255, 255),  # jaune
    4: (128, 0, 0),
    5: (0, 128, 0),
    6: (0, 0, 128),
    7: (255, 0, 255),
    8: (255, 255, 0),
    9: (0, 255, 255),
    10: (255, 0, 255),
    11: (255, 255, 0),
    12: (0, 255, 255),
    13: (128, 128, 0),
    14: (0, 128, 128),
    15: (0, 0, 255),
    16: (64, 64, 255),
    17: (0, 0, 128),
}


SPECIAL_KEYS = {
    "TAB": 9,
}

TEXT_HEIGHT = 36
TEXT_COLOR = (255, 255, 255)
TEXT_THICKNESS = 1
TEXT_Y = 22


class MagicTool:

    def __init__(self, video_path: Path, data: list):
        self.video_path = str(video_path)
        self.data = data
        self._frame_number = 0
        self._prev_frame_number = None
        self.do_save = True
        self.do_quit = False

        # Load the video file
        if self.video_path.endswith(".m3u8"):
            self.vr = M3u8VideoReader()
        else:
            self.vr = OpenCV_VideoReader()

        def _make_change_point_command(bt):
            def cmd():
                self.change_points_type(BALL_TYPE_TO_ID[bt])
            return cmd
        
        self.ball_type_commands = [
            (str(btkey), _make_change_point_command(bt), "Change ball type to " + bt)
            for bt, btkey in BALL_TYPE_TO_KEY.items()
        ]
        self.commands = [
            ("q", self.quit, "Quit"),
            ("x", self.quit_no_save, "Quit without save"),
            ("h", self.usage, "Help"),
            ("TAB", self.go_previous_frame_number, "Go last viewed frame"),
            ("n", self.go_next_frame, "Next frame"),
            ("N", self.go_next10_frames, "Next 10 frames"),
            ("b", self.go_previous_frame, "Previous frame"),
            ("B", self.go_previous10_frames, "Previous 10 frames"),
            ("f", self.go_previous_hit, "Previous hit"),
            ("g", self.go_next_hit, "Next hit"),
            ("l", self.go_last_point, "Last point"),
            ("L", self.go_last_non_zero_point, "Last non zero point"),
            ("m", self.go_next_point, "Next point"),
            ("M", self.go_previous_point, "Previous point"),
            ("Z", self.go_last_player_associated, "Last player associated"),
            *self.ball_type_commands,
            ("j", self.toggle_diagonal, "Toggle diagonal"),
            ("k", self.toggle_lob, "Toggle lob"),
            ("s", self.save, "Save"),
            ("d", self.remove_points, "Remove points"),
        ]
        self.command_dict = {}
        for i, (key, fct, helper) in enumerate(self.commands):
            if key in self.command_dict:
                raise ValueError(f"Key {key} already exists in command_dict")
            if len(key) == 1:
                key_code = ord(key)
            elif key in SPECIAL_KEYS:
                key_code = SPECIAL_KEYS[key]
            else:
                raise ValueError(f"Unknown key: {key}")
            self.command_dict[key_code] = i

    @property
    def frame_number(self):
        return self._frame_number
    
    @frame_number.setter
    def frame_number(self, value):
        self._prev_frame_number = self._frame_number
        self._frame_number = value

    def usage(self):
        print("#########################################")
        print("Magique Tool for annotation of hits ....")
        print("#########################################")
        print("Help : ")
        for key, _, helper in self.commands:
            print(f" - {key}: {helper}")

    def run(self, start_idx=0):
        self.vr.open(self.video_path)
        self.n_frames = len(self.vr)

        if start_idx is not None:
            self.frame_number = start_idx

        # Create a window to display the video frame
        cv2.namedWindow("Video Frame")

        # Loop to wait for the starting key
        self.usage()
        print("Press any key to start the video playback.")
        video_started = False
        while not video_started:
            key = cv2.waitKey(0) & 0xFF
            if key != 255:  # A key has been pressed
                video_started = True

        # Once a frame is selected, allow the user to select points on that frame
        cv2.setMouseCallback("Video Frame", self.select_points_callback)

        while True:
            # Display the frame
            if self.display_frame() is None:
                break
            try:
                key_code = (
                    cv2.waitKey(0) & 0xFF
                )  # Wait for user to press a key or click the mouse to select the frame
                key_char = chr(key)
                if key_code in self.command_dict:
                    _, fct, _ = self.commands[self.command_dict[key_code]]
                    fct()
                else:
                    print(f"Unknown command: {key_code} ({key_char})")

                if self.do_quit:
                    break

            except Exception as e:
                print(f"Error: {e}")
                print(traceback.format_exc())
                continue

        if self.do_save:
            self.save()

        # Release video capture and close windows
        cv2.destroyAllWindows()
        cv2.waitKey(1)

    def save(self):  # save anotations
        # Write selected frames and points to a JSON file
        base_name, extension = os.path.splitext(self.video_path)
        output_file = f"{base_name}_selected_points.json"
        print(f"Saving data to {output_file}")
        with open(output_file, "w") as js:
            d = sorted(self.data, key=lambda x: x["image_id"] if "image_id" in x else 0)
            json.dump(d, js, indent=2)

    def add_player_points(self, x, y):  # add points on anotations
        for di in range(len(self.data)):
            d = self.data[di]
            if d["image_id"] == self.frame_number:
                data[di]["Player_x"] = x
                data[di]["Player_y"] = y
                return
        # not exist, so add
        print("WARNING !! The player info is better to be set when hit is set before.")
        self.data.append(
            {
                "image_id": self.frame_number,
                "Player_x": x,
                "Player_y": y,
                "Ball_type": 0,
            }
        )

    def add_points(self, x, y, t=0):  # add points on anotations
        for di in range(len(self.data)):
            d = self.data[di]
            if d["image_id"] == self.frame_number:
                b_type = d["Ball_type"] if "Ball_type" in d else t
                data[di] = {
                    "image_id": self.frame_number,
                    "Ball_x": x,
                    "Ball_y": y,
                    "Ball_type": b_type,
                }
                return
        # not exist, so add
        data.append(
            {"image_id": self.frame_number, "Ball_x": x, "Ball_y": y, "Ball_type": t}
        )

    def change_points_type(self, t):  # change points type
        for di in range(len(self.data)):
            d = self.data[di]
            if d["image_id"] == self.frame_number:
                self.data[di]["Ball_type"] = t

    def _data_key(self, key, value=None, default=None):
        for di in range(len(self.data)):
            d = self.data[di]
            if d["image_id"] == self.frame_number:
                if value is not None:
                    d[key] = value
                return d.get(key, default)
        return default

    def is_lob(self):
        return self._data_key(IS_LOB_KEY, default=False)

    def set_lob(self, v: bool):
        self._data_key(IS_LOB_KEY, v)

    def is_diagonal(self):
        return self._data_key(IS_DIAGONAL_KEY, default=False)

    def set_diagonal(self, v: bool):
        self._data_key(IS_DIAGONAL_KEY, v)

    def toggle_lob(self):
        v = not self.is_lob()
        self.set_lob(v)

    def toggle_diagonal(self):
        v = not self.is_diagonal()
        self.set_diagonal(v)

    def current_point_idx(self, idx):
        try:
            ds = sorted([int(d["image_id"]) for d in self.data])
            return ds.index(idx)
        except ValueError:
            return 0

    def remove_points(self):  # remove points from the list
        i = 0
        while i < len(self.data):
            if self.data[i]["image_id"] == self.frame_number:
                self.data.pop(i)
            else:
                i += 1

    def get_points(self):  # get points
        for di in range(len(self.data)):
            d = self.data[di]
            if d["image_id"] == self.frame_number:
                return d
        return None

    def get_last_points(self):  # get next points idx
        idxs = sorted([d["image_id"] for d in self.data])
        return idxs[-1]

    def get_last_non_zero_points(self):
        non_zero_data = [d for d in self.data if d["Ball_type"] != 0]
        idxs = sorted([d["image_id"] for d in non_zero_data])
        return idxs[-1]

    def get_next_points(self):  # get next points idx
        idxs = sorted([d["image_id"] for d in self.data])
        for i in idxs:
            if i > self.frame_number:
                return i
        return -1

    def get_previous_points(self):  # get next points idx
        idxs = sorted([d["image_id"] for d in self.data], reverse=True)
        for i in idxs:
            if i < self.frame_number:
                return i
        return -1

    def get_next_hit(self):  # get next hit idx
        idxs = sorted(
            [
                d["image_id"]
                for d in self.data
                if d["Ball_type"] in SERVICE_HIT_VOLLEY_SMASH_IDS
            ]
        )
        for i in idxs:
            if i > self.frame_number:
                return i
        return -1

    def get_previous_hit(self):  # get next hit idx
        idxs = sorted(
            [
                d["image_id"]
                for d in self.data
                if d["Ball_type"] in SERVICE_HIT_VOLLEY_SMASH_IDS
            ],
            reverse=True,
        )
        for i in idxs:
            if i < self.frame_number:
                return i
        return -1

    def get_last_player_associated(self):
        idxs = sorted([d["image_id"] for d in self.data if "Player_x" in d.keys()])
        return idxs[-1]

    def go_next_n_frames(self, n: int):
        self.frame_number += n
        self.frame_number = min(self.frame_number, self.n_frames - 1)

    def go_previous_n_frames(self, n: int):
        self.frame_number -= n
        self.frame_number = max(self.frame_number, 0)

    def go_next_frame(self):
        self.go_next_n_frames(1)

    def go_next10_frames(self):
        self.go_next_n_frames(10)

    def go_previous_frame(self):
        self.go_previous_n_frames(1)

    def go_previous10_frames(self):
        self.go_previous_n_frames(10)

    def go_next_point(self):
        i = self.get_next_points()
        self.frame_number = self.n_frames - 1 if i == -1 else i

    def go_previous_point(self):
        i = self.get_previous_points()
        self.frame_number = 0 if i == -1 else i

    def go_next_hit(self):
        i = self.get_next_hit()
        self.frame_number = self.n_frames - 1 if i == -1 else i

    def go_previous_hit(self):
        i = self.get_previous_hit()
        self.frame_number = 0 if i == -1 else i

    def go_last_player_associated(self):
        i = self.get_last_player_associated()
        self.frame_number = self.n_frames - 1 if i == -1 else i

    def go_last_point(self):
        self.frame_number = self.get_last_points()

    def go_last_non_zero_point(self):
        self.frame_number = self.get_last_non_zero_points()

    def go_previous_frame_number(self):
        if self._prev_frame_number is not None:
            self.frame_number = self._prev_frame_number

    def quit(self):
        self.do_quit = True

    def quit_no_save(self):
        self.do_quit = True
        self.do_save = False

    def display_frame(self):
        frame = self.vr[self.frame_number].asnumpy()  # check frame number in here ?
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # OpenCV is BGR
        if frame is None:
            return None
        h, w, c = frame.shape
        d = self.get_points()

        # cv2.rectangle(frame, (0, 0), (1920, 36), (0, 0, 0), -1)
        info = np.zeros((36, frame.shape[1], 3), dtype=frame.dtype)

        if d is not None and "Ball_x" in d and "Ball_y" in d:  # A points exist, draw information
            cv2.putText(
                info,
                f'Points: [{d["Ball_x"]}, {d["Ball_y"]}]',
                (750, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                TEXT_COLOR,
                TEXT_THICKNESS,
                cv2.LINE_AA,
            )
            b_type = d["Ball_type"] if "Ball_type" in d else 0
            color = BALL_TYPES_COLOR[b_type]
            cv2.putText(
                info,
                f"Coups: {BALL_TYPES[b_type]}",
                (1000, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                TEXT_COLOR,
                TEXT_THICKNESS,
                cv2.LINE_AA,
            )
            cv2.circle(
                frame, (d["Ball_x"], d["Ball_y"]), 7, color, -1
            )  # Draw circle on the ball

        if d is not None and "Player_x" in d:  # A player info exist, draw information
            cv2.circle(
                frame, (d["Player_x"], d["Player_y"]), 7, (255, 0, 255), -1
            )  # Draw circle on the player

        cv2.putText(
            info,
            f"Frame: {self.frame_number} / {len(self.vr)}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            TEXT_COLOR,
            TEXT_THICKNESS,
            cv2.LINE_AA,
        )
        pidx = self.current_point_idx(self.frame_number)
        cv2.putText(
            info,
            f"Nb Coups: {pidx}/{len(self.data)}",
            (350, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            TEXT_COLOR,
            TEXT_THICKNESS,
            cv2.LINE_AA,
        )

        cv2.putText(
            info,
            f"Diagonal? {self.is_diagonal()}",
            (1500, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0) if self.is_diagonal() else TEXT_COLOR,
            TEXT_THICKNESS,
            cv2.LINE_AA,
        )
        cv2.putText(
            info,
            f"Is Lob? {self.is_lob()}",
            (1750, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0) if self.is_lob() else TEXT_COLOR,
            TEXT_THICKNESS,
            cv2.LINE_AA,
        )
        img = np.vstack([frame, info])

        cv2.imshow("Video Frame", img)
        return img

    def select_points_callback(self, event, x, y, flags, params):
        try:
            # print(f"Event: {event}, x: {x}, y: {y}, flags: {flags}, params: {params}")
            if event == cv2.EVENT_LBUTTONDOWN:
                self.add_points(x, y)
                self.display_frame()
            elif event == cv2.EVENT_RBUTTONDOWN:
                self.add_player_points(x, y)  # player coordonate
                self.display_frame()
        except Exception as e:
            print(f"Error: {e}")
            print(traceback.format_exc())
            return


if __name__ == "__main__":
    print('Starting "Outils Magique" ...')

    # Initialize ArgumentParser
    parser = argparse.ArgumentParser(description="Extract points from a video.")
    parser.add_argument("--video_path", "-v", type=str, help="Path to the video file",
                        action=VideoAliasPathAction)
    parser.add_argument(
        "--json_path",
        type=str,
        default=None,
        help="Path to the json file containing annotations",
    )
    parser.add_argument("--start_idx", type=int, default=0, help="Start index of the video")
    args = parser.parse_args()
    load_json = False

    # Check if the annotation file exists and is valid
    video_path = Path(args.video_path)
    assert video_path.exists(), f"Video file does not exist: {video_path}"

    data = []
    if args.json_path is None:
        json_path = video_path.parent / (video_path.stem + "_selected_points.json")

    else:
        json_path = Path(args.json_path)

    if json_path.exists():
        with open(json_path, "r") as f:
            data = json.load(f)
        print(f"Loaded {len(data)} points from {json_path}")
    else:
        print(f"No annotation file found at {json_path}, creating a new one")
        data = []

    magic_tool = MagicTool(video_path, data)
    magic_tool.run(start_idx=args.start_idx)
