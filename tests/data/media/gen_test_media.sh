#!/usr/bin/env bash
set -ex

SRC_VIDEO=big_buck_bunny_360p_5mb.mp4

if [[ ! -f "$SRC_VIDEO" ]]; then
    wget -c \
        "https://sample-videos.com/video123/mp4/360/big_buck_bunny_360p_5mb.mp4" \
        -O "$SRC_VIDEO"
fi


for i in $(seq 0 10); do
    image_dest_dir="video_image_folder/video$i"
    mkdir -p "$image_dest_dir"
    if [[ ! -f "$image_dest_dir/frame_00001.jpg" ]]; then
        ffmpeg -i "$SRC_VIDEO" \
            -ss "00:00:$(printf %02d $i)" \
            -t 2 \
            "$image_dest_dir/frame_%05d.jpg"
    fi

    mkdir -p video_folder
    video="video_folder/video$i.mp4"
    if [[ ! -f "$video" ]]; then
        ffmpeg -i "$SRC_VIDEO" \
            -ss "00:00:$(printf %02d $i)" \
            -t 2 \
            "$video"
    fi
done

if [[ -d gulp_output ]]; then
    rm -rf gulp_output
fi

gulp_20bn_csv_jpeg \
    --videos_per_chunk 2 \
    --num_workers 1 \
    gulp_videos.csv \
    video_image_folder \
    gulp_output

