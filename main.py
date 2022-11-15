import argparse
import asone
import datetime
import numpy as np
import csv
from asone import ASOne
from loguru import logger

def main(args):
    filter_classes = args.filter_classes

    if filter_classes:
        # filter_classes = [filter_classes]
        filter_classes = ["car"]

    dt_obj = ASOne(
        # tracker=asone.BYTETRACK,
        tracker=asone.NORFAIR,
        detector=asone.YOLOX_DARKNET_PYTORCH,
        # detector=asone.YOLOV7_ONNX,
        use_cuda=args.use_cuda
        )
    # Get tracking function
    track_fn = dt_obj.track_video(args.video_path,
                                output_dir=args.output_dir,
                                save_result=args.save_result,
                                display=args.display,
                                draw_trails=args.draw_trails,
                                filter_classes=filter_classes)
    vals = []
    # Loop over track_fn to retrieve outputs of each frame 
    for bbox_details, frame_details in track_fn:
        # print(':)')
        bbox_xyxy, ids, scores, class_ids = bbox_details
        frame, frame_num, fps = frame_details
        # logger.info('Frame No. %s, IDs:%s, Classes: %s' % (frame_num, ids, class_ids))
        # logger.debug(class_ids)
        val = ('%s, %s, %s, %s' % (datetime.datetime.now(), frame_num, ids, class_ids))
        logger.info(val)
        vals.append(val)
    
    logger.debug(vals)
    with open ('output.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['datetime, frame_num, object_id, class_id'])
        for val in vals:
            logger.info('erm')
            writer.writerow([val])
        
    
    # write the vals array to the csv file

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('video_path', help='Path to input video')
    parser.add_argument('--cpu', default=True, action='store_false', dest='use_cuda',
                        help='run on cpu if not provided the program will run on gpu.')
    parser.add_argument('--no_save', default=True, action='store_false',
                        dest='save_result', help='whether or not save results')
    parser.add_argument('--no_display', default=True, action='store_false',
                        dest='display', help='whether or not display results on screen')
    parser.add_argument('--output_dir', default='data/results',  help='Path to output directory')
    parser.add_argument('--draw_trails', default=False,  help='if provided object motion trails will be drawn.')
    parser.add_argument('--filter_classes', default=None, help='Filter class name')

    args = parser.parse_args()

    main(args)
