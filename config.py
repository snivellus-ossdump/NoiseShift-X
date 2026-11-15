"""Central config (paths are relative to this file, so the repo works on any machine)."""
import os
BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "content")          # put the SIDD folder and the two .mat files here
SIDD_SMALL_ROOT = os.path.join(DATA, "SIDD_Small_sRGB_Only", "Data")
OFFICIAL_VAL_NOISY = os.path.join(DATA, "ValidationNoisyBlocksSrgb.mat")
OFFICIAL_VAL_GT = os.path.join(DATA, "ValidationGtBlocksSrgb.mat")
CACHE_DIR = os.path.join(DATA, "sidd_patches")
CKPT_DIR = os.path.join(BASE, "ckpt")
RESULTS_DIR = os.path.join(BASE, "results")

# SIDD camera codes (3rd field of each scene-folder name): GP = Google Pixel, IP = iPhone 7,
# S6 = Samsung Galaxy S6 Edge, N6 = Motorola Nexus 6, G4 = LG G4.
CAMERAS = ["GP", "IP", "S6", "N6", "G4"]

# ---- experiment settings: EVERYONE in the team must use identical values ----
PATCH = 128               # training crop size
PATCHES_PER_IMAGE = 100   # random crops cached per training scene
EVAL_BLOCK = 256          # fixed, non-overlapping evaluation blocks
TEST_FRAC = 0.25          # fraction of scenes per camera reserved as the fixed test set (never trained on)
EPOCHS = 10
SEED = 0
