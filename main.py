import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split


def load_video_data(file_paths, labels, target_size=(64, 64), batch_size=8, shuffle=True, save_dir='data/new_video'):
    """
    Load video data from file paths and save processed videos.

    Parameters:
    - file_paths (list): List of file paths to MP4 files.
    - labels (list): List of corresponding labels.
    - target_size (tuple): Target size to resize frames to (default: (64, 64)).
    - batch_size (int): Batch size for loading frames (default: 8).
    - shuffle (bool): Whether to shuffle the data (default: True).
    - save_dir (str): Directory to save processed videos (default: 'data/new_video').

    Returns:
    - generator: A generator yielding batches of video frames and corresponding labels.
    """

    indices = np.arange(len(file_paths))
    if shuffle:
        np.random.shuffle(indices)

    for start_idx in range(0, len(file_paths), batch_size):
        batch_indices = indices[start_idx:start_idx + batch_size]
        batch_file_paths = file_paths[batch_indices]
        batch_labels = labels[batch_indices]

        batch_frames = []
        for file_path in batch_file_paths:
            cap = cv2.VideoCapture(file_path)
            frame_list = []
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, target_size)
                # Process frame here if needed

                # Example of saving processed frames
                save_path = os.path.join(save_dir, os.path.basename(file_path))
                frame_list.append(frame)
            cap.release()
            cv2.destroyAllWindows()
            batch_frames.append(frame_list)

        # Example of saving processed batch frames
        for idx, frames in enumerate(batch_frames):
            save_path = os.path.join(save_dir, os.path.basename(batch_file_paths[idx]))
            out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'MP4V'), 30, (target_size[1], target_size[0]))
            for frame in frames:
                out.write(frame)
            out.release()

        yield np.array(batch_frames), batch_labels

# def train_video(train_loader):
    
# Example usage
def main():
    # Directory containing training data
    data_dir = 'data/edge_error/0_correct'

    file_paths = []
    labels = []

    for filename in os.listdir(data_dir):
        if filename == '.DS_Store':
            continue  # Skip .DS_Store file

        if filename.endswith(".mp4"):
            file_paths.append(os.path.join(data_dir, filename))
            labels.append(filename)  # Adjust as needed for your label extraction logic

    file_paths = np.array(file_paths)
    labels = np.array(labels)

    print(f"Found {len(file_paths)} video files.")

    # Split data into train and test sets
    file_paths_train, file_paths_test, labels_train, labels_test = train_test_split(file_paths, labels, test_size=0.2, random_state=42)

    batch_size = 8
    target_size = (64, 64)

    # Create new_video directory if it doesn't exist
    save_dir = 'data/new_video'
    # os.makedirs(save_dir, exist_ok=True)

    train_loader = load_video_data(file_paths_train, labels_train, target_size=target_size, batch_size=batch_size, save_dir=save_dir)
    test_loader = load_video_data(file_paths_test, labels_test, target_size=target_size, batch_size=batch_size, save_dir=save_dir)

    # train_video(train_loader)


if __name__ == "__main__":
    main()
