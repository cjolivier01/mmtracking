#!/bin/bash
python \
  demo/demo_vid.py \
  configs/mot/deepsort/sort_faster-rcnn_fpn_4e_mot17-public.py \
  --input=${HOME}/Videos/zw_blackhawks-2/small_stitched.mkv \
  --output=vis_results \
  $@

