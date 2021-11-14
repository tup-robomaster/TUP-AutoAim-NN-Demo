'''
Description:A Demo of Yolox Armor Detect
Date:2021.09.10
'''
import numpy
import params
import torch
import numpy as np
import onnxruntime as rt
import params
import cv2

def resize(img,size):
    height_src, width_src = img.shape[:2]  # origin hw
    resize_factor_x = size[3] / width_src  # resize image to img_size
    resize_factor_y = size[2] / height_src  # resize image to img_size
    resize_factor = min(resize_factor_x, resize_factor_y)
    dst = cv2.resize(img, (int(width_src * resize_factor), int(height_src * resize_factor)), interpolation=cv2.INTER_LINEAR)
    dst = cv2.copyMakeBorder(dst, int(height_src * (resize_factor_y - resize_factor) * 0.5),
                            int(height_src * (resize_factor_y - resize_factor) * 0.5),
                            int(width_src * (resize_factor_x - resize_factor) * 0.5 + 1),
                            int(width_src * (resize_factor_x - resize_factor) * 0.5 + 1),
                            cv2.BORDER_CONSTANT,
                            (0,0,0))
    resize_matrix = np.array([[1 / resize_factor, 0], [0, 1 / resize_factor]])
    resize_vector = np.array([int(width_src * (resize_factor_x - resize_factor) * 0.5 + 1),
    				int(height_src * (resize_factor_y - resize_factor) * 0.5)])
    return resize_matrix, resize_vector, dst

def inv_sigmoid(x):
    y = np.log(1 / x - 1)
    return y


def is_overlapped(bbox_1,bbox_2):
    box_1 = cv2.boundingRect(bbox_1)
    box_2 = cv2.boundingRect(bbox_2)
    intersect_tl = [max(box_1[0], box_2[0]), max(box_1[1], box_2[1])]
    intersect_br = [min(box_1[0] + box_1[2], box_2[0] + box_2[3]), min(box_1[1] + box_1[3], box_2[1] + box_2[3])]
    # print(bbox_1)
    # print("--------------------------------------------------------")
    # print("tl.x: %d \t br.x: %d" % (intersect_tl[0], intersect_br[0]))
    # print("tl.y: %d \t br.y: %d" % (intersect_tl[1], intersect_br[1]))
    # print("dx : %d , dy : %d" % (intersect_tl[0] - intersect_br[0], intersect_tl[1] - intersect_br[1]))
    if (intersect_tl[0] - intersect_br[0]) < 0 or (intersect_tl[1] - intersect_br[1]) < 0:
        # print("Intersected!")
        return True
    else:
        # print("Fine")
        return False

class BBox:
    def __init__(self):
        self.pts = np.array([[0, 0], [0, 0], [0, 0], [0, 0]])
        self.conf = 0
        self.color = 0
        self.id = 0

if __name__ == '__main__':
    #Initialize
    model_dir = "yolox.onnx"
    using_cam = False
    using_video = True
    detect_color = 0 #0 for blue,1 for red,2 for gray
    TOPK_NUM = 128
    KEEP_THRES = 0.1
    video_dir = "sample01.mp4"		
    pred = []
    # Create Inference Session
    sess = rt.InferenceSession(model_dir)
    input = sess.get_inputs()[0]
    output =sess.get_outputs()[0]
    CamParam = params.CamParam()

    # print(input)
    if using_video:
        cap = cv2.VideoCapture(video_dir)
    elif using_cam:
        cap = cv2.VideoCapture(0)
    #Loop
    print("Start Looping...")
    while (True):
        cnt = 0
    #     #Get frame
        ret,src =cap.read()
        show_img = src
        target_list = []
        final_target_list = []
        #Resize the image
        resize_matrix, resize_vector, img = resize(src,input.shape)
        cv2.imshow("src", src)
        cv2.waitKey(1)
        img_pre = np.expand_dims(img, axis=0)
        img_pre = img_pre.astype(np.float32)
        img_pre = np.transpose(img_pre, (0, 3, 1, 2))
        pred = sess.run([output.name], {input.name: img_pre})
        pred_tensor = torch.as_tensor(pred[0][0])
        # print(pred_tensor.shape)
        pred_conf = pred_tensor[:, 8]
        pred_topk_index = torch.topk(pred_conf, k = TOPK_NUM, sorted=True).indices
        final_pred = torch.index_select(pred_tensor, dim = 0,index = pred_topk_index)

        #post process + NMS
        removed_index = np.zeros([TOPK_NUM])
        for i in range(TOPK_NUM):
            pred = final_pred[i]
            tmp_bbox = BBox()
            tmp_bbox.conf = pred[8]
            if tmp_bbox.conf < KEEP_THRES:
                break
            if removed_index[i] == 1:
                continue
            #Setting boundingbox
            tmp_bbox.pts[0] = np.array(np.matmul((pred[0:2] - resize_vector), resize_matrix), dtype=np.int32)
            tmp_bbox.pts[1] = np.array(np.matmul((pred[2:4] - resize_vector), resize_matrix), dtype=np.int32)
            tmp_bbox.pts[2] = np.array(np.matmul((pred[4:6] - resize_vector), resize_matrix), dtype=np.int32)
            tmp_bbox.pts[3] = np.array(np.matmul((pred[6:8] - resize_vector), resize_matrix), dtype=np.int32)
            #print(pred[0:8])
            #for i in range(4):
            		#cv2.line(src, tuple(tmp_bbox.pts[i % 4]), tuple(tmp_bbox.pts[(i + 1) % 4]), (255,0,0))

            tmp_bbox.color = int(torch.argmax(pred[9:12]))
            tmp_bbox.id = int(torch.argmax(pred[12:]))

            # print("i ", i, "\n", tmp_bbox.pts[0][0])
            for j in range (i + 1, TOPK_NUM):
                # print("j:%d"% j)
                tmp_bbox_j = BBox()
                # print(tmp_bbox.pts)
                pred_j = final_pred[j]
                tmp_bbox_j.conf = torch.sigmoid(pred_j[8])

                if tmp_bbox_j.conf < KEEP_THRES :
                    break
                if removed_index[j] == 1:
                    continue
                tmp_bbox_j.pts[0] = np.array(np.matmul(pred_j[0:2], resize_matrix) - resize_vector, dtype=np.int32)
                tmp_bbox_j.pts[1] = np.array(np.matmul(pred_j[2:4], resize_matrix) - resize_vector, dtype=np.int32)
                tmp_bbox_j.pts[2] = np.array(np.matmul(pred_j[4:6], resize_matrix) - resize_vector, dtype=np.int32)
                tmp_bbox_j.pts[3] = np.array(np.matmul(pred_j[6:8], resize_matrix) - resize_vector, dtype=np.int32)
                if is_overlapped(tmp_bbox.pts, tmp_bbox_j.pts):
                    removed_index[j] = True
                    continue
        # print("\n")
            target_list.append(tmp_bbox)
        #Choose Target
        if len(target_list) != 0:
            for target in target_list:
                if target.id == 0 or target.id == 6 or target.id == 7:
                    armor_size = params.Armor.armor_size_big
                else:
                    armor_size = params.Armor.armor_size_small
                armor_apex_2d = target.pts.astype(np.float32)
                rvec = np.zeros([3,3])
                tvec = np.zeros([1,3])
                armor_apex_3d = numpy.array([[0, - armor_size[0] / 2, armor_size[1] / 2],
                                            [0, - armor_size[0] / 2, - armor_size[1] / 2],
                                             [0, armor_size[0] / 2, - armor_size[1] / 2],
                                             [0, armor_size[0] / 2, armor_size[1] / 2]],dtype=np.float32)
                retval, rvec, tvec = cv2.solvePnP(armor_apex_3d, armor_apex_2d, CamParam.cam_mat, CamParam.dis_coeff, rvec, tvec, False, cv2.SOLVEPNP_EPNP)
                rmat = cv2.Rodrigues(rvec)[0]

                a_x = np.arctan2(rmat[2][1], rmat[2][2]) * 180 / np.pi
                a_y = np.arctan2(-rmat[2][0], np.sqrt(np.square(rmat[2][1]) + np.square(rmat[2][2]))) * 180 / np.pi
                a_z = np.arctan2(rmat[1][0], rmat[0][0]) * 180 / np.pi
                #print(a_x)
                #print(a_y)
                #print(a_z)
                print(target.conf)
                #print("\n")
                dist = int(np.sqrt(np.square(tvec[0]) + np.square(tvec[1]) + np.square(tvec[2])))

                if target.color == 1:
                    color = (0, 0, 255)
                else:
                    color = (255, 0, 0)
                # color = (0, 255, 0)

                cv2.putText(src, str(target.id), (target.pts[0][0], target.pts[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, color)
                cv2.putText(src, "dist: " + str(dist), (target.pts[0][0] + 10, target.pts[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0,255,0))
                for i in range(4):
                    cv2.line(src, tuple(target.pts[i % 4]), tuple(target.pts[(i + 1) % 4]), (0,255,0))

        cv2.imshow("pred",src)


