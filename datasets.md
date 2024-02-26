# Ego4D-HCap Dataset

Ego4D-HCap is a hierarchical video captioning dataset containing captions for long-range videos across three temporal 
granularities: short-range clip captions focusing on specific human actions, medium-length segment descriptions focusing on specific human actions, and long-range video summaries depicting the overall intent and goals of the video. The dataset is built on [Ego4D](https://ego4d-data.org) videos. 

# Ego4D videos

Download the Ego4D videos using the following steps.
1. Get [License Agreement](https://ego4d-data.org/docs/start-here/#cli-download) and [download](https://github.com/facebookresearch/Ego4d/blob/main/ego4d/cli/README.md) the Ego4D videos. 
2. Use [crop_and_resize.sh](scripts/crop_and_resize.sh) to crop and chunk the videos to the smaller side of 288 pixels and chunk length of 5 minutes. \
(Note: This step is required for faster I/O. You can also evaluate the pretrained models without this step.)

# Ego4D-HCap captions
You can download the Ego4D-HCap captions from this [link](https://drive.google.com/drive/folders/14cMn3iqVw_FdH_JUjXDTZNG8e6m0FbnC?usp=share_link). The following are the descriptions of the caption annotations.


## 1. Clip Caption
```clips_train.pkl``` and ```clips_val.pkl``` are the training and validation files for short-range clip captions. Each file contains a list of tuples of (vid, start_sec, end_sec, clip_caption).
```
vid :   video_id
start_sec:  start second of the clip
end_sec:    end second of the clip
clip_caption:  ground-tuth clip caption
```

## 2. Segment Description
```segments_train.pkl``` and ```segments_val.pkl``` are the training and the validation files for medium-range segment descriptions. ```segments_train_pseudo.pkl``` is the LLM-generated pseudo-annotations for segments.\
 Each file contains a list of dictionaries containing following fields:
```
'vid' :   video_id
'start_sec' :  start second of the segment
'end_sec' :    end second of the segment
'captions_pred' :  predicted clip captions by Video ReCap for the particular segment
'captions_gt' :    ground-truth captions of the particular segment
'segment_description' :    ground-truth segment description
```


## 3. Video Summary
```videos_train.pkl``` and ```videos_val.pkl``` are the training and the validation files for long-range video summaries. ```videos_train_pseudo.pkl``` is the LLM-generated pseudo-annotations for segments.\
Each file contains a list of dictionaries containing following fields:
```
'vid' :   video_id
'start_sec' :  start second of the segment
'end_sec' :    end second of the segment
'segment_descriptions' :  ground-truth segment descriptions of the particular video
'segment_descriptions_pred' :    predicted segment descriptions by Video ReCap for the particular video
'video_summary' :    ground-truth video_summary
```

