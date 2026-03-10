import os
import time
import urllib.request

import cv2
from tts import speak
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

try:
    import pygame
    pygame.mixer.pre_init(44100, -16, 2, 512)
    pygame.mixer.init()
    _PYGAME_AVAILABLE = True
except ImportError:
    _PYGAME_AVAILABLE = False

MODEL_PATH = "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

# Hand skeleton connections (pairs of landmark indices)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),           # Index
    (5, 9), (9, 10), (10, 11), (11, 12),      # Middle
    (9, 13), (13, 14), (14, 15), (15, 16),    # Ring
    (13, 17), (17, 18), (18, 19), (19, 20),   # Pinky
    (0, 17),                                   # Palm base
]


def play_fah():
    """Play fah.mp3 as a non-blocking sound effect."""
    if not _PYGAME_AVAILABLE:
        return
    try:
        pygame.mixer.music.load(os.path.join("assets", "fah.mp3"))
        pygame.mixer.music.play()
    except Exception:
        pass


class HandTracker:
    """
    Hand tracking module using the MediaPipe Tasks API and OpenCV.
    Detects hand landmarks and provides utility methods for landmark access.
    """

    def __init__(self, max_hands=2, detection_confidence=0.7, tracking_confidence=0.7):
        self.max_hands = max_hands
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence

        self._ensure_model()

        base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=self.max_hands,
            min_hand_detection_confidence=self.detection_confidence,
            min_hand_presence_confidence=self.detection_confidence,
            min_tracking_confidence=self.tracking_confidence,
        )
        self.landmarker = mp_vision.HandLandmarker.create_from_options(options)
        self.results = None

        # Landmark indices for fingertip detection
        self.TIP_IDS = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky tips

    def _ensure_model(self):
        """Download the hand landmarker .task model if not already present."""
        if not os.path.exists(MODEL_PATH):
            print(f"Downloading hand landmarker model to '{MODEL_PATH}' ...")
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            print("Download complete.")

    def find_hands(self, frame, draw=True):
        """
        Detect hands in a BGR frame and optionally draw landmarks.

        Args:
            frame: BGR image from OpenCV.
            draw: Whether to draw landmarks and connections on the frame.

        Returns:
            Annotated frame (BGR).
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int(time.time() * 1000)
        self.results = self.landmarker.detect_for_video(mp_image, timestamp_ms)

        if draw and self.results.hand_landmarks:
            h, w, _ = frame.shape
            for hand_lms in self.results.hand_landmarks:
                # Draw connections
                for start_idx, end_idx in HAND_CONNECTIONS:
                    x1 = int(hand_lms[start_idx].x * w)
                    y1 = int(hand_lms[start_idx].y * h)
                    x2 = int(hand_lms[end_idx].x * w)
                    y2 = int(hand_lms[end_idx].y * h)
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)
                # Draw landmark circles
                for lm in hand_lms:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        return frame

    def find_positions(self, frame, hand_index=0, draw=True):
        """
        Get pixel coordinates for all 21 landmarks of a specific hand.

        Args:
            frame: BGR image (used for resolving pixel coords).
            hand_index: Index of the hand (0 = first detected hand).
            draw: Whether to draw circles on each landmark.

        Returns:
            List of [landmark_id, x, y] for the requested hand,
            or an empty list if the hand is not detected.
        """
        landmark_list = []

        if not self.results or not self.results.hand_landmarks:
            return landmark_list
        if hand_index >= len(self.results.hand_landmarks):
            return landmark_list

        h, w, _ = frame.shape
        hand_lms = self.results.hand_landmarks[hand_index]

        for lm_id, lm in enumerate(hand_lms):
            cx, cy = int(lm.x * w), int(lm.y * h)
            landmark_list.append([lm_id, cx, cy])
            if draw:
                cv2.circle(frame, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        return landmark_list

    def fingers_up(self, landmark_list):
        """
        Determine which fingers are extended.

        Args:
            landmark_list: Output of find_positions().

        Returns:
            List of 5 booleans [Thumb, Index, Middle, Ring, Pinky].
            Returns empty list if no landmarks available.
        """
        if len(landmark_list) < 21:
            return []

        fingers = []

        # Thumb: use index MCP vs pinky MCP to detect hand orientation in the
        # mirrored frame, then compare tip x to IP joint in the right direction.
        # index_mcp.x > pinky_mcp.x  →  right-hand orientation, thumb extends right
        # index_mcp.x < pinky_mcp.x  →  left-hand orientation,  thumb extends left
        if landmark_list[5][1] > landmark_list[17][1]:
            fingers.append(landmark_list[4][1] > landmark_list[3][1])
        else:
            fingers.append(landmark_list[4][1] < landmark_list[3][1])

        # Four fingers: tip y-position above PIP joint means extended
        for tip_id in self.TIP_IDS[1:]:
            fingers.append(landmark_list[tip_id][2] < landmark_list[tip_id - 2][2])

        return fingers

    def get_hand_label(self, hand_index=0):
        """
        Return 'Left' or 'Right' label for the detected hand.

        Args:
            hand_index: Index of the hand.

        Returns:
            String label or None.
        """
        if self.results and self.results.handedness and hand_index < len(self.results.handedness):
            return self.results.handedness[hand_index][0].display_name
        return None

    def is_back_palm(self, landmark_list):
        """
        Determine if the back of the hand faces the camera using the 2D
        cross product of wrist→index_MCP and wrist→pinky_MCP vectors.

        Works reliably with a horizontally mirrored (flipped) webcam feed.

        Returns:
            True  — back/dorsal side facing camera
            False — palm facing camera
        """
        if len(landmark_list) < 21:
            return False
        wrist   = landmark_list[0]   # [id, cx, cy]
        idx_mcp = landmark_list[5]   # index finger MCP
        pin_mcp = landmark_list[17]  # pinky MCP

        v1x = idx_mcp[1] - wrist[1]
        v1y = idx_mcp[2] - wrist[2]
        v2x = pin_mcp[1] - wrist[1]
        v2y = pin_mcp[2] - wrist[2]

        # Negative cross_z = clockwise ordering in screen coords (y-down)
        # = back of hand facing camera in a mirrored webcam feed
        cross_z = v1x * v2y - v1y * v2x
        return cross_z < 0

    def recognize_gesture(self, fingers, landmark_list=None):
        """
        Map a finger state list to a gesture word.

        Args:
            fingers:       List of 5 booleans from fingers_up().
            landmark_list: Optional output of find_positions() for
                           orientation-dependent gestures.

        Returns:
            Gesture label string, or empty string if unrecognized.
        """
        if len(fingers) < 5:
            return ""

        T, I, M, R, P = fingers  # Thumb, Index, Middle, Ring, Pinky

        # --- Gesture definitions ---
        if T and I and M and R and P:
            if landmark_list and self.is_back_palm(landmark_list):
                return "Please"
            return "Goodbye"

        if not T and I and M and R and P:
            return "Hello"

        if not T and not I and not M and not R and P:
            return "I"

        if not T and I and not M and not R and not P:
            if landmark_list and self.is_back_palm(landmark_list):
                return "Me"
            return "You"

        if not T and not I and not M and not R and not P:
            if landmark_list and self.is_back_palm(landmark_list):
                return "Sorry"
            return "No"

        if T and I and M and not R and not P:
            return "Yes"

        if not T and I and M and R and not P:
            return "Are"

        if not T and I and not M and not R and P:
            return "Thank You"

        if T and I and not M and not R and P:
            return "I Love You"

        if not T and not I and M and not R and not P:
            if landmark_list and self.is_back_palm(landmark_list):
                return "_fah_"

        return ""  # Unknown / unrecognized gesture

    def recognize_gesture_multi(self, all_fingers, all_landmarks=None):
        """
        Recognize gestures that require both hands.

        Args:
            all_fingers:   List of fingers_up() results for each detected hand.
            all_landmarks: Optional list of find_positions() results per hand.

        Returns:
            Gesture label string, or empty string if unrecognized.
        """
        if len(all_fingers) < 2:
            return ""

        def thumbs_up(f):
            return len(f) == 5 and f[0] and not f[1] and not f[2] and not f[3] and not f[4]

        def open_palm(f):
            return len(f) == 5 and all(f)

        def index_only(f):
            return len(f) == 5 and not f[0] and f[1] and not f[2] and not f[3] and not f[4]

        def index_and_pinky(f):
            return len(f) == 5 and not f[0] and f[1] and not f[2] and not f[3] and f[4]

        def fist(f):
            return len(f) == 5 and not any(f)

        if all(index_only(f) for f in all_fingers[:2]):
            return "My"

        if all(index_and_pinky(f) for f in all_fingers[:2]):
            return "Good"

        if all(fist(f) for f in all_fingers[:2]):
            return "."

        if all(thumbs_up(f) for f in all_fingers[:2]):
            return "Like"

        hands = all_fingers[:2]
        if any(thumbs_up(f) for f in hands) and any(open_palm(f) for f in hands):
            return "Help"

        if any(open_palm(f) for f in hands) and any(index_only(f) for f in hands):
            return "Is"

        if all(open_palm(f) for f in all_fingers[:2]):
            # Two open palms dorsal → Want; two open palms facing → Clear
            if all_landmarks and len(all_landmarks) >= 2:
                if all(self.is_back_palm(lm) for lm in all_landmarks[:2]):
                    return "Want"
            return "Clear"

        return ""


def draw_fps(frame, fps):
    cv2.putText(
        frame, f"FPS: {int(fps)}", (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
    )


def draw_finger_status(frame, fingers, hand_label, hand_index, frame_width):
    """Render per-hand finger status on screen."""
    names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
    x_offset = 10 if hand_index == 0 else frame_width // 2
    y_start = 70

    label_text = f"Hand {hand_index + 1}: {hand_label or 'Unknown'}"
    cv2.putText(frame, label_text, (x_offset, y_start - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    for i, (name, up) in enumerate(zip(names, fingers)):
        color = (0, 255, 0) if up else (0, 0, 255)
        status = "UP" if up else "DOWN"
        cv2.putText(frame, f"{name}: {status}",
                    (x_offset, y_start + 25 * (i + 1)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)


def draw_gesture_label(frame, gesture, frame_width, frame_height):
    """Display the recognized gesture word prominently at the bottom center."""
    if not gesture:
        return
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.8
    thickness = 4
    (text_w, text_h), baseline = cv2.getTextSize(gesture, font, font_scale, thickness)
    x = (frame_width - text_w) // 2
    y = frame_height - 30
    # Dark background rectangle for readability
    cv2.rectangle(frame,
                  (x - 12, y - text_h - 12),
                  (x + text_w + 12, y + baseline + 8),
                  (0, 0, 0), cv2.FILLED)
    cv2.putText(frame, gesture, (x, y), font, font_scale, (0, 255, 128), thickness)


def draw_sentence(frame, sentence, frame_width):
    """Display the current building sentence near the top center."""
    if not sentence:
        return
    text = " ".join(sentence)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.9
    thickness = 2
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x = max(10, (frame_width - text_w) // 2)
    y = 60
    cv2.rectangle(frame, (x - 8, y - text_h - 8), (x + text_w + 8, y + baseline + 6),
                  (20, 20, 20), cv2.FILLED)
    cv2.putText(frame, text, (x, y), font, font_scale, (255, 220, 50), thickness)


def draw_hold_progress(frame, gesture, elapsed, hold_time, frame_width, frame_height):
    """Draw a progress bar showing how long the current gesture has been held."""
    if not gesture:
        return
    bar_w = 300
    bar_h = 12
    x = (frame_width - bar_w) // 2
    y = frame_height - 80
    progress = min(elapsed / hold_time, 1.0)
    cv2.rectangle(frame, (x, y), (x + bar_w, y + bar_h), (60, 60, 60), cv2.FILLED)
    cv2.rectangle(frame, (x, y), (x + int(bar_w * progress), y + bar_h), (0, 200, 255), cv2.FILLED)
    cv2.rectangle(frame, (x, y), (x + bar_w, y + bar_h), (180, 180, 180), 1)


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    tracker = HandTracker(max_hands=2, detection_confidence=0.7, tracking_confidence=0.7)

    prev_time = 0

    # --- Sentence building state ---
    HOLD_TIME = 0.5          # seconds to hold a gesture before it's committed
    sentence = []            # words committed so far
    last_gesture = ""        # gesture currently being held
    hold_start = 0.0         # when the current hold started
    word_committed = False   # True once the held gesture has been added this hold
    flash_sentence = []      # completed sentence to flash briefly
    flash_until = 0.0        # time.time() until which to show the flash

    print("Hand Tracking started. Press 'q' to quit.")

    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to grab frame.")
            break

        frame = cv2.flip(frame, 1)  # Mirror view
        frame_width = frame.shape[1]
        frame_height = frame.shape[0]

        frame = tracker.find_hands(frame, draw=True)

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time

        draw_fps(frame, fps)

        # --- Gesture detection ---
        detected_gesture = ""
        if tracker.results and tracker.results.hand_landmarks:
            num_hands = len(tracker.results.hand_landmarks)
            all_fingers = []
            all_landmarks = []
            for i in range(num_hands):
                landmark_list = tracker.find_positions(frame, hand_index=i, draw=False)
                fingers = tracker.fingers_up(landmark_list)
                hand_label = tracker.get_hand_label(hand_index=i)
                if fingers:
                    all_fingers.append(fingers)
                    all_landmarks.append(landmark_list)
                    draw_finger_status(frame, fingers, hand_label, i, frame_width)

            detected_gesture = tracker.recognize_gesture_multi(all_fingers, all_landmarks)
            if not detected_gesture and len(all_fingers) == 1:
                lm_list = tracker.find_positions(frame, hand_index=0, draw=False)
                detected_gesture = tracker.recognize_gesture(all_fingers[0], lm_list)

        # --- Hold-to-commit logic ---
        if detected_gesture != last_gesture:
            last_gesture = detected_gesture
            hold_start = curr_time
            word_committed = False

        if detected_gesture and not word_committed:
            elapsed = curr_time - hold_start
            draw_hold_progress(frame, detected_gesture, elapsed, HOLD_TIME,
                               frame_width, frame_height)

            if elapsed >= HOLD_TIME:
                word_committed = True
                if detected_gesture == ".":
                    # End of sentence — flash it, speak it, then reset
                    if sentence:
                        flash_sentence = sentence + ["."]
                        flash_until = curr_time + 3.0
                        completed = " ".join(sentence)
                        print("Sentence: " + completed + " .")
                        speak(completed)
                    sentence = []
                elif detected_gesture == "Clear":
                    # Discard current sentence silently
                    sentence = []
                    flash_sentence = ["[Cleared]"]
                    flash_until = curr_time + 1.5
                    print("Sentence cleared.")
                elif detected_gesture == "_fah_":
                    play_fah()
                else:
                    # Add word only if not already in this sentence
                    if detected_gesture not in sentence:
                        sentence.append(detected_gesture)

        draw_gesture_label(frame, "" if detected_gesture == "_fah_" else detected_gesture, frame_width, frame_height)

        # Show completed sentence flash or building sentence
        if curr_time < flash_until:
            draw_sentence(frame, flash_sentence, frame_width)
        else:
            draw_sentence(frame, sentence, frame_width)

        total_hands = len(tracker.results.hand_landmarks) if (tracker.results and tracker.results.hand_landmarks) else 0
        cv2.putText(frame, f"Hands detected: {total_hands}", (10, frame_height - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        cv2.imshow("Hand Tracking - MediaPipe", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Hand Tracking stopped.")


if __name__ == "__main__":
    main()
