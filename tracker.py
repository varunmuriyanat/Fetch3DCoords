import cv2
import numpy as np
import glob
from time import time, monotonic
from pathlib import Path
import depthai
from pprint import pprint
from depthai_helpers.version_check import check_depthai_version
from depthai_helpers.object_tracker_handler import show_tracklets
from depthai_helpers.config_manager import DepthConfigManager
from depthai_helpers.arg_manager import CliArgs
from collections import OrderedDict


#device = depthai.Device('', False)

#streams = device.get_available_streams()

#print("Available streams {}".format(streams))

#pprint(dir(device))

#left_homography = device.get_left_intrinsic()

#print(left_homography)

#check_depthai_version()
points_mapping = ['Nose', 'Neck', 'R-Sho', 'R-Elb', 'R-Wr', 'L-Sho', 'L-Elb', 'L-Wr', 'R-Hip', 'R-Knee', 'R-Ank', 'L-Hip', 'L-Knee', 'L-Ank', 'R-Eye', 'L-Eye', 'R-Ear', 'L-Ear']

print('Using depthai module from: ', depthai.__file__)
print('Depthai version installed: ', depthai.__version__)
global args
time_start = time()

def print_packet_info_header():
    print('[hostTimestamp streamName] devTstamp seq camSrc width height Bpp')

def print_packet_info(packet, stream_name):
    meta = packet.getMetadata()
    print("[{:.6f} {:15s}]".format(time() - time_start, stream_name), end='')
    if meta is not None:
        source = meta.getCameraName()
        if stream_name.startswith('disparity') or stream_name.startswith('depth'):
            source += '(rectif)'
        print(" {:.6f}".format(meta.getTimestamp()), meta.getSequenceNum(), source, end='')
        print('', meta.getFrameWidth(), meta.getFrameHeight(), meta.getFrameBytesPP(), end='')
    print()
    return


class AITracker:
    runThread = True

    def stopLoop(self):
        self.runThread = False


    def startLoop(self):
        cliArgs = CliArgs()
        args = vars(cliArgs.parse_args())

        args['cnn_model'] = 'openpose'
        args['mono_resolution'] = 720
        args['mono_fps'] = 30.0
        args['streams'] = ['metaout', 'previewout']
        args['cnn_camera'] = 'left_right'

        configMan = DepthConfigManager(args)

        labels = configMan.labels
        NN_json = configMan.NN_config

        # This json file is sent to DepthAI. It communicates what options you'd like to enable and what model you'd like to run.

        config = configMan.jsonConfig

        decode_nn = configMan.decode_nn
        show_nn = configMan.show_nn

        stream_names = [stream if isinstance(stream, str) else stream['name'] for stream in configMan.stream_list]
        print("CONSTRUCT DEVICE ")
        self.device = depthai.Device("", False)
        # print("DEVICE ATTRIBUTES {}".format(dir(self.device)))
        # print("DEVICE PIPELINE CREATION {}".format(config))

        p = self.device.create_pipeline(config=config)

        rotation_matrix = np.array(self.device.get_rotation())
        translation_matrix = np.array(self.device.get_translation())
        intrinsic_left = np.array(self.device.get_left_intrinsic())
        intrinsic_right = np.array(self.device.get_right_intrinsic())
        distortion_left = np.array([-5.459006, 18.099184, 0.002737, 0.000663, -18.760843, -5.511819, 18.311680, -18.972065])
        distortion_right = np.array([-4.975889, 15.759747, -0.000449, -0.000381, -15.732677, -5.032327, 15.978766, -15.938593])

        print("RECEIVED ROTATION {} -> {} -> {}".format(type(intrinsic_left), type(rotation_matrix), rotation_matrix))
        print("RECEIVED TRANSLATION {}".format(translation_matrix))
        print("MAtrix type {}, shape {}".format(type(intrinsic_left), intrinsic_left.shape))

        R1, R2, projection_left, projection_right, Q, roi1, roi2 = cv2.stereoRectify(intrinsic_left, distortion_left,
                                                      intrinsic_right, distortion_right,
                                                      (1280,720),
                                                      rotation_matrix, translation_matrix)     
        print("P LEFT {}".format(projection_left))
        print("P RIGHT {}".format(projection_right))
        print("R1 {}".format(R1))
        print("R2 {}".format(R2))
        print("Q {}".format(Q))

        #data = np.load('./stereo.npy', allow_pickle=True).item()
        #print(data)

        if p is None:
            print('Pipeline is not created.')
            exit(3)
        #print(args)
        nn2depth = self.device.get_nn_to_depth_bbox_mapping()

        print("NN2DEPTH {}".format(nn2depth))

        t_start = time()
        frame_count = {}
        frame_count_prev = {}
        nnet_prev = {}
        nnet_prev["entries_prev"] = {}
        nnet_prev["nnet_source"] = {}
        frame_count['nn'] = {}
        frame_count_prev['nn'] = {}
        horizontal_scale = 1.
        vertical_scale = 1.

        point_pairs = OrderedDict()
        # print("PP {}".format(dir(point_pairs)))

        NN_cams = {'rgb', 'left', 'right'}

        for cam in NN_cams:
            nnet_prev["entries_prev"][cam] = None
            nnet_prev["nnet_source"][cam] = None
            frame_count['nn'][cam] = 0
            frame_count_prev['nn'][cam] = 0

        stream_windows = []
        for s in stream_names:
            if s == 'previewout':
                for cam in NN_cams:
                    stream_windows.append(s + '-' + cam)
            else:
                stream_windows.append(s)

        for w in stream_windows:
            frame_count[w] = 0
            frame_count_prev[w] = 0

        while self.runThread:
            # retreive data from the device
            # data is stored in packets, there are nnet (Neural NETwork) packets which have additional functions for NNet result interpretation
            self.nnet_packets, self.data_packets = p.get_available_nnet_and_data_packets(blocking=True)

            for _, nnet_packet in enumerate(self.nnet_packets):
                #print_packet_info(nnet_packet, 'NNet')
                meta = nnet_packet.getMetadata()
                camera = meta.getCameraName()
                nnet_prev["nnet_source"][camera] = nnet_packet
                nnet_prev["entries_prev"][camera] = decode_nn(nnet_packet, config=config, NN_json=NN_json)
                seq = meta.getSequenceNum()
                keys = point_pairs.keys()
                frame_count['metaout'] += 1
                frame_count['nn'][camera] += 1
                keys = list(point_pairs.keys())
                if len(keys) > 10:
                    # print(keys)
                    point_pairs.popitem(last=False)
                if len(keys) > 0 and keys[len(keys)-1] -seq >  100:
                    continue
                if seq not in point_pairs.keys():
                    point_pairs[seq] = {}

                point_pairs[seq][camera] = nnet_prev["entries_prev"][camera][0]
                if 'left' in point_pairs[seq].keys() and 'right' in point_pairs[seq].keys():
                    print(seq)
                    print(point_pairs[seq])
                    print("=============================")
                    left_camera_points = np.array(point_pairs[seq]['left'])
                    right_camera_points = np.array(point_pairs[seq]['right'])
                    for i in range(0, 18):
                        if len(left_camera_points[i]) > 0 and len(right_camera_points[i]) > 0:
                            print((left_camera_points[i][0][:2]))
                            left_points = np.asarray(left_camera_points[i][0][:2])
                            right_points = np.asarray(right_camera_points[i][0][:2])
                            left_points[0] = left_points[0]*horizontal_scale + nn2depth['off_x']
                            left_points[1] = left_points[1]*vertical_scale + nn2depth['off_y']
                            right_points[0] = right_points[0]*horizontal_scale + nn2depth['off_x']
                            right_points[1] = right_points[1]*vertical_scale + nn2depth['off_y']

                            left_points = left_points.astype('float64')
                            right_points = right_points.astype('float64')
                            point_3d = cv2.triangulatePoints(projection_left, projection_right, left_points, right_points)
                            #print(type(projection_left[0][0]))
                            #print(type(left_points[0]))

                            print("{}  3D POINT {}".format(points_mapping[i], point_3d))

                #print("CAM {}, SEQ {} -> {}".format(camera, meta.getSequenceNum(), nnet_prev["entries_prev"][camera][0]))
                #print(nnet_prev["entries_prev"][camera][0])


            for packet in self.data_packets:
                packetData = packet.getData()

                if packetData is None:
                    print('Invalid packet data!')
                    continue

                if packet.stream_name == 'previewout':
                    #frame_bgr = packetData
                    meta = packet.getMetadata()
                    camera = meta.getCameraName()
                    data0 = packetData[0, :, :]
                    data1 = packetData[1, :, :]
                    data2 = packetData[2, :, :]
                    frame = cv2.merge([data0, data1, data2])
                    window_name = 'previewout-' + camera
                    if nnet_prev["entries_prev"][camera] is not None:
                        frame = show_nn(nnet_prev["entries_prev"][camera], frame, NN_json=NN_json, config=config)

                    cv2.imshow(window_name, frame)
                    #print(meta.getFrameWidth())
                    horizontal_scale = nn2depth['max_w']/meta.getFrameWidth()
                    vertical_scale = nn2depth['max_h']/meta.getFrameHeight()
                #elif packet.stream_name.startswith('depth') or packet.stream_name == 'disparity_color':
                    #frame = packetData
                    #print(frame)

                if packet.stream_name == 'depth':
                    frame = packetData
                    #print("Depth packet {}".format(frame))
                    if len(frame.shape) == 2:
                        if frame.dtype == np.uint8:  # grayscale
                            cv2.putText(frame, packet.stream_name, (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255))
                            cv2.putText(frame, "fps: " + str(frame_count_prev[window_name]), (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255))
                        else:  # uint16
                            if args['pointcloud'] and "depth" in stream_names and "rectified_right" in stream_names and right_rectified is not None:
                                try:
                                    from depthai_helpers.projector_3d import PointCloudVisualizer
                                except ImportError as e:
                                    raise ImportError(f"\033[1;5;31mError occured when importing PCL projector: {e} \033[0m ")
                                if pcl_converter is None:
                                    pcl_converter = PointCloudVisualizer(self.device.get_right_intrinsic(), 1280, 720)
                                right_rectified = cv2.flip(right_rectified, 1)
                                pcl_converter.rgbd_to_projection(frame, right_rectified)
                                pcl_converter.visualize_pcd()

                            frame = (65535 // frame).astype(np.uint8)
                            # colorize depth map, comment out code below to obtain grayscale
                            # frame = cv2.applyColorMap(frame, cv2.COLORMAP_HOT)
                            frame = cv2.applyColorMap(frame, cv2.COLORMAP_JET)
                            cv2.putText(frame, packet.stream_name, (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255)
                            cv2.putText(frame, "fps: " + str(frame_count_prev[window_name]), (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255)
                    else:  # bgr
                        cv2.putText(frame, packet.stream_name, (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255))
                        cv2.putText(frame, "fps: " + str(frame_count_prev[window_name]), (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255)
                    
                    if len(point_pairs.keys()) > 0:
                        k=(list(point_pairs.keys()))[-1]
                        if 'left' in point_pairs[k].keys() and 'right' in point_pairs[k].keys():
                            print("LEFT {}".format(point_pairs[k]['left']))
                            print("RIGHT {}".format(point_pairs[k]['right']))
                            print("{} ========".format(k))

                    camera = 'right'
                    if nnet_prev["entries_prev"][camera] is not None:
                        scale = {}
                        scale['hscale'] = horizontal_scale
                        scale['vscale'] = vertical_scale
                        scale['off_x'] = nn2depth['off_x']
                        scale['off_y'] = nn2depth['off_y']
                        frame = show_nn(nnet_prev["entries_prev"][camera], frame, NN_json=NN_json, config=config, nn2depth=nn2depth, scale=scale)
                    cv2.imshow('window_name', frame)



            key = cv2.waitKey(1)
            if key == ord('q'):
                break
        

if __name__ == "__main__":
    dai = AITracker()
    dai.startLoop()

