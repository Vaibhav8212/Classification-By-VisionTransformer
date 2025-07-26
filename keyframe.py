import av
from model_utils import predict

def extract_keyframes(video_path):
    container = av.open(video_path)
    stream = container.streams.video[0]
    stream.thread_type = 'AUTO'

    frames = []
    for frame in container.decode(stream):
        if frame.key_frame:
            img = frame.to_ndarray(format='rgb24')  # shape: (H, W, 3)
            frames.append(img)

    container.close()

    return frames if frames else None

if __name__ == '__main__':
    import time
    video_path = 'test.mov'

    start_time = time.time()
    results = predict(extract_keyframes, video_path)
    end_time = time.time()

    print(f'Results: {results}')
    print(f'Time taken: {end_time - start_time:.2f} seconds')
