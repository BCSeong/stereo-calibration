'''
test_stereo_varification.py
'''

from matplotlib import pyplot as plt
import numpy as np


def generate_pseudo_error(num_target: int = 4, num_frames: int = 51, tvec_diff: np.ndarray = np.array([0.8, 0.0, 0.0]), reproj_err: np.ndarray = np.array([0.2])):
    tvec_diff = np.tile(tvec_diff, (num_frames, 1))
    reproj_errs = np.tile(reproj_err, (num_frames, 1))

    tvec_diff_stack = np.tile(tvec_diff, (num_target, 1, 1))
    reproj_errs_stack = np.tile(reproj_errs, (num_target, 1, 1))

    tvec_diff_stack_rand = tvec_diff_stack.copy()
    tvec_diff_stack_rand += np.random.randn(num_target, num_frames, 3) * 0.01
    reproj_errs_stack_rand = reproj_errs_stack.copy()
    reproj_errs_stack_rand += np.random.randn(num_target, num_frames, 1) * 0.01
    reproj_errs_stack_rand = np.abs(reproj_errs_stack_rand)

    return tvec_diff_stack_rand, reproj_errs_stack_rand

def plot_tvecs(tvecs: np.ndarray, dimension_names: list):
    num_target, num_frames, num_dimensions = tvecs.shape
    fig, ax = plt.subplots(num_dimensions, num_target)
    for i in range(num_target):
        for j in range(num_dimensions):
            ax[j, i].plot(range(num_frames), tvecs[i, :, j], label=f"Target {i+1}, {dimension_names[j]}")
            ax[j, i].set_title(f"Target {i+1}, {dimension_names[j]}")
            ax[j, i].set_xlabel("Frame")
            ax[j, i].set_ylabel(f"Translation {dimension_names[j]} (mm)")
            ax[j, i].grid(True)
            ax[j, i].set_xlim(0, num_frames)     
            ax[j, i].set_xticks(range(0, num_frames, 10))
            ax[j, i].set_ylim(-.1, 1.1)
            ax[j, i].set_yticks(np.arange(0, 1.1, 0.2))
    plt.tight_layout()
    plt.show()

def plot_reproj_errs(reproj_errs: np.ndarray):
    num_target, num_frames, _ = reproj_errs.shape
    fig, ax = plt.subplots(1,num_target)
    for i in range(num_target):
        ax[i].plot(range(num_frames), reproj_errs[i, :], label=f"Target {i+1}")
        ax[i].set_title(f"Target {i+1}")
        ax[i].set_xlabel("Frame")
        ax[i].set_ylabel(f"Reprojection Error (px)")
        ax[i].grid(True)
        ax[i].set_xlim(0, num_frames)     
        ax[i].set_xticks(range(0, num_frames, 10))
        ax[i].set_ylim(-.1, 1.1)
        ax[i].set_yticks(np.arange(0, 1.1, 0.2))
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    num_target = 4
    num_frames = 51
    tvec_diff = np.array([0.8, 0.0, 0.0])
    reproj_err = np.array([0.2])

    tvec_diff_stack_rand, reproj_errs_stack_rand = generate_pseudo_error(num_target, num_frames, tvec_diff, reproj_err)

    print(tvec_diff_stack_rand.shape) # (4, 51, 3)
    print(reproj_errs_stack_rand.shape) # (4, 51, 1)


    ideal_light_dir = np.array([[1.68, 50.73],
                                [179.92, 51.63],
                                [270.39, 51.03],
                                [90.39, 51.27]]) # shape (4, 2(azimuth, elevation))
    dimension_names = ["X", "Y", "Z"]

    plot_tvecs(tvec_diff_stack_rand, dimension_names)
    plot_reproj_errs(reproj_errs_stack_rand)

