import numpy as np
import cv2
import os 
import random
import glob
from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_audioclips


def resize_all_images(images_numpy, size, walk = False):
    '''
    Perform Auto Cropping
    '''

    target_w, target_h = size
    resized_images_numpy = []
    loose_w, loose_h = 0, 0
    if walk:
        loose_w, loose_h = int(target_w * 0.2), int(target_h * 0.2)        
        target_w += loose_w
        target_h += loose_h
    for image in images_numpy:
        h, w, _ = image.shape
        if abs(w - target_w) < abs(h - target_h) and w < h:
            resized_image = cv2.resize(image, (target_w, int(h * target_w / w )))
            resized_image = resized_image[resized_image.shape[0]//2-target_h//2:
                                          resized_image.shape[0]//2+target_h//2, 
                                          :, :]
        elif  abs(w - target_w) > abs(h - target_h) and h < w:
            resized_image = cv2.resize(image, (int(w * target_h / h ), target_h))
            resized_image = resized_image[:, 
                                          resized_image.shape[1]//2-target_w//2:
                                          resized_image.shape[1]//2+target_w//2, 
                                            :]
        elif h > w:
            resized_image = cv2.resize(image, (target_w, int(h * target_w / w )))
            resized_image = resized_image[resized_image.shape[0]//2-target_h//2:
                                          resized_image.shape[0]//2+target_h//2, 
                                          :, :]
        else:
            resized_image = cv2.resize(image, (int(w * target_h / h ), target_h))
            resized_image = resized_image[:, 
                                          resized_image.shape[1]//2-target_w//2:
                                          resized_image.shape[1]//2+target_w//2, 
                                            :]
        resized_images_numpy.append(resized_image)

    return resized_images_numpy, (loose_w, loose_h)


def blend(vid_path, images, seconds, fps = 30, transition_period=1, size=(512, 512), walk = False):
    

    # Read all images:

    images_numpy = [cv2.imread(image_path) for image_path in images]
    images_numpy, (loose_w, loose_h) = resize_all_images(images_numpy, size, walk)

    # print([image.shape for image in images_numpy])
    gl_index = 0
    videowriter = cv2.VideoWriter(vid_path, cv2.VideoWriter_fourcc(*'XVID'), fps, size)


    # initialize walking vectors:
    walk_start = []
    directional_vector = []
    gl_indices = [0] * len(images_numpy)
    for i in range(len(images_numpy)):
        im_start = (random.randint(0, loose_w) , random.randint(0, loose_h))
        im_end = (random.randint(0, loose_w) , random.randint(0, loose_h))
        directional_vector.append((-(im_end[0] - im_start[0])/(seconds[i] * fps + transition_period * fps * 2), -(im_end[1] - im_start[1])/(seconds[i] * fps + transition_period * fps * 2)))
        walk_start.append(im_start)


    for i, image in enumerate(images_numpy):
        
        # Add images to videowriter
        for j in range(int(seconds[i] * fps)):
            M = np.float64([
                            [1, 0, directional_vector[i][0] * gl_indices[i]],
                            [0, 1, directional_vector[i][1] * gl_indices[i]]
                        ])
            translated_image = cv2.warpAffine(image.copy(), M, (image.shape[1], image.shape[0]))
            cropped_image = translated_image[walk_start[i][1]:walk_start[i][1] + size[1],
                        walk_start[i][0]:walk_start[i][0] + size[0],
                        :]
            videowriter.write(cropped_image.astype(np.uint8))
            gl_index +=1
            gl_indices[i] += 1
        
        # Start Transition if needed
        if transition_period > 0 and i != len(images_numpy) -1:
            for j in range(transition_period * fps):
                M = np.float64([
                    [1, 0,directional_vector[i][0] * gl_indices[i]],
                    [0, 1,directional_vector[i][1] * gl_indices[i]]
                ])
                translated_image = cv2.warpAffine(image.copy(), M, (image.shape[1], image.shape[0]))
                cropped_image = translated_image[walk_start[i][1]:walk_start[i][1] + size[1],
                            walk_start[i][0]:walk_start[i][0] + size[0],
                            :]

                M = np.float64([
                    [1, 0,directional_vector[i+1][0] * gl_indices[i+1]],
                    [0, 1,directional_vector[i+1][1] * gl_indices[i+1]]
                ])
                translated_image = cv2.warpAffine(images_numpy[i + 1].copy(), M, (image.shape[1], image.shape[0]))
                cropped_image_next = translated_image[walk_start[i+1][1]:walk_start[i+1][1] + size[1],
                            walk_start[i+1][0]:walk_start[i+1][0] + size[0],
                            :]

                
                blended_image = cv2.addWeighted(cropped_image, 1 - j/(transition_period * fps),
                                                cropped_image_next, j/(transition_period * fps),
                                                0.0)

                videowriter.write(blended_image.astype(np.uint8))
                gl_index+=1
                gl_indices[i] += 1
                gl_indices[i + 1] += 1

    videowriter.release()
    return vid_path



def combine_audio_video(audio_file, video_file):
    """Combine the most recent audio and video files from their directories."""

    video_clip = VideoFileClip(video_file)
    audio_clip = AudioFileClip(audio_file)

    # Calculate the number of times to loop the audio
    loops_required = int(video_clip.duration // audio_clip.duration) + 1
    audio_clips = [audio_clip] * loops_required

    # Make the looped audio
    looped_audio_clip = concatenate_audioclips(audio_clips)

    # Set the looped audio to the video
    final_clip = video_clip.set_audio(looped_audio_clip.subclip(0, video_clip.duration))

    output_file_name = os.path.basename(video_file).replace('.mp4', '_with_audio.mp4')
    output_file_path = os.path.join(os.path.dirname(video_file), output_file_name)
    final_clip.write_videofile(output_file_path, codec="libx264", audio_codec="aac")

    return output_file_path


if __name__ == "__main__":

    blend(  "out.mp4",
            images = glob.glob("examples/*"),
            seconds=[3, 3, 3],
            transition_period=1,
            size=(512, 512),
            walk=True, 
            )

