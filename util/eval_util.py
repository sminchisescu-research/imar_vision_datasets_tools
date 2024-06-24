import numpy as np
import json
import os
import copy

from util.ghum_util import GHUMHelper
from util.smplx_util import SMPLXHelper
from util.dataset_util import project_3d_to_2d


H36M_TO_J17 = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10, 0, 7, 9]


def compute_similarity_transform(S1, S2):
    """
    Source of the code: https://gist.github.com/mkocabas/54ea2ff3b03260e3fedf8ad22536f427
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    """
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.T
        S2 = S2.T
        transposed = True
    assert(S2.shape[1] == S1.shape[1])
    # 1. Remove mean.
    mu1 = S1.mean(axis=1, keepdims=True)
    mu2 = S2.mean(axis=1, keepdims=True)
    X1 = S1 - mu1
    X2 = S2 - mu2
    # 2. Compute variance of X1 used for scale.
    var1 = np.sum(X1**2)
    # 3. The outer product of X1 and X2.
    K = X1.dot(X2.T)
    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, Vh = np.linalg.svd(K)
    V = Vh.T
    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = np.eye(U.shape[0])
    Z[-1, -1] *= np.sign(np.linalg.det(U.dot(V.T)))
    # Construct R.
    R = V.dot(Z.dot(U.T))
    # 5. Recover scale.
    scale = np.trace(R.dot(K)) / var1
    # 6. Recover translation.
    t = mu2 - scale*(R.dot(mu1))
    # 7. Error:
    S1_hat = scale*R.dot(S1) + t
    if transposed:
        S1_hat = S1_hat.T
    return S1_hat

def compute_similarity_transform_batch(S1, S2):
    """Batched version of compute_similarity_transform."""
    S1_hat = np.zeros_like(S1)
    for i in range(S1.shape[0]):
        S1_hat[i] = compute_similarity_transform(S1[i], S2[i])
    return S1_hat

def euclidean_distance(a,b):
    re = np.sqrt(((a - b)**2).sum(axis=-1)).mean(axis=-1)
    return re

def h36m_to_lsp(j3d):
    j3d_j17 = j3d[:, H36M_TO_J17, ...]
    j3d_lsp = j3d_j17[:, :14, ...] - j3d_j17[:, 14:15, ...]
    return j3d_lsp

def center_blazeposeghum_33(arr):
    mid_hip = (arr[:, 23:24, ...] + arr[:, 24:25, ...]) / 2.0
    return arr - mid_hip

def lsp_joint_error(gt, pred, procrustes=False):
    gt = h36m_to_lsp(gt)
    pred = h36m_to_lsp(pred)
    pred_hat = compute_similarity_transform_batch(pred, gt)
    error = euclidean_distance(pred_hat, gt) if procrustes else euclidean_distance(pred, gt)
    return error

def center_vertices(posed_data):
    return (posed_data.vertices - posed_data.joints[:, 0:1, :]).numpy()


#####
def load_data(data_path):
    with open(data_path) as f:
        data = json.load(f)
    for subj_name in data:
        for action_name in data[subj_name]:
            for person_id in range(len(data[subj_name][action_name]['persons'])):
                for data_type in data[subj_name][action_name]['persons'][person_id]:
                    for key in data[subj_name][action_name]['persons'][person_id][data_type]:
                        if type(data[subj_name][action_name]['persons'][person_id][data_type][key]) is list:
                            data[subj_name][action_name]['persons'][person_id][data_type][key] = np.array(data[subj_name][action_name]['persons'][person_id][data_type][key])
            if 'cam_params' in data[subj_name][action_name]['other']:
                for key1 in data[subj_name][action_name]['other']['cam_params']:
                    for key2 in data[subj_name][action_name]['other']['cam_params'][key1]:
                        data[subj_name][action_name]['other']['cam_params'][key1][key2] = np.array(data[subj_name][action_name]['other']['cam_params'][key1][key2])
    return data


def save_data(old_data, data_path):
    data = copy.deepcopy(old_data)
    for subj_name in data:
        for action_name in data[subj_name]:
            for data_type in data[subj_name][action_name]['persons'][0]:
                for person_id in range(len(data[subj_name][action_name]['persons'])):
                    for key in data[subj_name][action_name]['persons'][person_id][data_type]:
                        if type(data[subj_name][action_name]['persons'][person_id][data_type][key]) is np.ndarray:
                            data[subj_name][action_name]['persons'][person_id][data_type][key] = data[subj_name][action_name]['persons'][person_id][data_type][key].tolist()
            if 'cam_params' in  data[subj_name][action_name]['other']:
                for key in data[subj_name][action_name]['other']['cam_params']:
                    for subkey in data[subj_name][action_name]['other']['cam_params'][key]:
                        if type(data[subj_name][action_name]['other']['cam_params'][key][subkey]) is np.ndarray:
                            data[subj_name][action_name]['other']['cam_params'][key][subkey] = data[subj_name][action_name]['other']['cam_params'][key][subkey].tolist()
    with open(data_path, 'w') as outfile:
        json.dump(data, outfile)


def aggregate_metrics(metric_fns, seq_results, has_contact_fr_id):
    agg_results = {}
    for subj_name in seq_results:
        for action_name in seq_results[subj_name]:
            for metric_fn in metric_fns:
                metric_names = [metric_fn.__name__]
                if has_contact_fr_id:
                    metric_names.append(metric_fn.__name__ + '_c')
                for metric_name in metric_names:
                    if metric_name not in agg_results:
                        agg_results[metric_name] = []
                    agg_results[metric_name].append(seq_results[subj_name][action_name][metric_name])
    for metric_fn in metric_fns:
        metric_names = [metric_fn.__name__]
        if has_contact_fr_id:
            metric_names.append(metric_fn.__name__ + '_c')
        for metric_name in metric_names:
            agg_results[metric_name] = np.mean(np.array(agg_results[metric_name])).astype(float)
    return agg_results


def get_pred_info(data_pred):
    # this is a big hack
    pred_types = []
    for subj_name in data_pred:
        for action_name in data_pred[subj_name]:
            pred = data_pred[subj_name][action_name]['persons'][0]
            for key in pred:
                if key in ['gpp', 'smplx', 'joints3d', 'blazeposeghum_33']:
                     pred_types.append(key)
            break
        break
    print('Detected prediction types are: ', pred_types)
    return pred_types

def get_gt_info(data_gt):
    if sorted(data_gt.keys()) == ['s01', 's05']:
        dataset_name = 'chi3d'
    elif sorted(data_gt.keys()) == ['s02', 's12', 's13']:
        dataset_name = 'fit3d'
    elif sorted(data_gt.keys()) == ['s04', 's05']:
        dataset_name = 'humansc3d'
    return dataset_name

def validate_pred_format(data_pred_path, data_template_path):
    try:
        MAX_FILE_SIZE_MB = 100

        if os.path.getsize(data_pred_path) / (1024*1024.0) > MAX_FILE_SIZE_MB:
            return False, 'File size too large!'

        data_pred = load_data(data_pred_path)
        data_template = load_data(data_template_path)
        pred_types = get_pred_info(data_pred)

        if len(pred_types) == 0:
            return False, 'There should be at least one detected prediction type!'

        if data_pred.keys() != data_template.keys():
            return False, 'Prediction subjects are not the same as the ground truth!'

        for subj_name in data_pred:
            if data_pred[subj_name].keys() != data_template[subj_name].keys():
                return False, 'Actions for subject %s are not the same as in ground truth' % subj_name
            for action_name in data_pred[subj_name]:
                if data_template[subj_name][action_name]['other']['video_fr_ids'] != data_pred[subj_name][action_name]['other']['video_fr_ids']:
                    return False, 'Frames in video_fr_ids are not exactly the same as in the template file!'
                persons_pred = data_pred[subj_name][action_name]['persons']
                persons_template = data_template[subj_name][action_name]['persons']
                if len(persons_pred) != len(persons_template):
                    return False, 'There should be %d persons predicted. Currently there are %d.' % (len(persons_template), len(persons_pred))
                for person_id in range(len(persons_template)):
                    pred = persons_pred[person_id]
                    template = persons_template[person_id]
                    for pred_type in pred_types:
                        if pred_type not in pred:
                            return False, 'Prediction type %s cannot be detected in each sequence!' % pred_type
                        if pred[pred_type].keys() != template[pred_type].keys():
                            return False, 'Prediction type %s in the wrong format!' % pred_type
                        for pred_subtype in pred[pred_type]:
                            if pred_subtype == 'joints3d':
                                for dim in [0, 2]:
                                    if pred[pred_type][pred_subtype].shape[dim] != template[pred_type][pred_subtype].shape[dim]:
                                        return False, 'Shape mismatch %s!' % pred_subtype
                                if pred[pred_type][pred_subtype].shape[1] not in [17, 25]:
                                    return False, 'Shape mismatch %s! Second dimension should be either 17 or 25!'
                            elif pred_subtype == 'blazeposeghum_33':
                                for dim in [0, 2]:
                                    if pred[pred_type][pred_subtype].shape[dim] != template[pred_type][pred_subtype].shape[dim]:
                                        return False, 'Shape mismatch %s!' % pred_subtype
                                if pred[pred_type][pred_subtype].shape[1] not in [33]:
                                    return False, 'Shape mismatch %s! Second dimension should be 33!'
                            else:
                                if pred[pred_type][pred_subtype].shape != template[pred_type][pred_subtype].shape:
                                    return False, 'Shape mismatch %s!' % pred_subtype

        return True, 'Prediction file is valid.'
        
    except Exception as e:
        return False, 'Error! Please validate your prediction file before submitting!'
    



def get_area(bbox):
    return  np.maximum(0, bbox[:, :, 2] - bbox[:, :, 0] + 1) * np.maximum(0, bbox[:, :, 3] - bbox[:, :, 1] + 1)

def compute_iou_distance(pred, gt):
    pred_matched = pred[:, :gt.shape[1], :]
    gt_matched = gt
    bbxes = np.stack((pred_matched, gt_matched), axis=0)
    intersection = np.concatenate([np.max(bbxes, axis=0)[:, :, :2], np.min(bbxes, axis=0)[:, :, 2:]], axis=2)
    inter_area = get_area(intersection)
    pred_area = get_area(pred_matched)
    gt_area = get_area(gt_matched)
    union_area = pred_area + gt_area - inter_area + np.finfo(float).eps
    iou = inter_area / union_area
    return iou

def merge_persons(persons):
    merged = {}
    for data_type in persons[0]:
        merged[data_type] = {}
        for sub_data_type in persons[0][data_type]:
            merged[data_type][sub_data_type] = np.concatenate([person[data_type][sub_data_type] for person in persons], axis=0)
    return merged

class EvaluationServer():
    def __init__(self, GHUM_Models_Path, SMPLX_Models_Path):
        self.ghum_helper = GHUMHelper(GHUM_Models_Path, load_renderer=False)
        self.smplx_helper = SMPLXHelper(SMPLX_Models_Path, load_renderer=False)
        
    def _vertex_error(self, gt, pred, model_type='ghum', procrustes=False):
        if model_type == 'ghum':
            gt_posed_data = self.ghum_helper.ghum_model.pose(self.ghum_helper.get_world_gpp(gt))
            pred_posed_data = self.ghum_helper.ghum_model.pose(self.ghum_helper.get_world_gpp(pred))
        elif model_type == 'smplx':
            gt_posed_data = self.smplx_helper.smplx_model(**self.smplx_helper.get_world_smplx_params(gt))
            pred_posed_data = self.smplx_helper.smplx_model(**self.smplx_helper.get_world_smplx_params(pred))
        gt_vertices = center_vertices(gt_posed_data)
        pred_vertices = center_vertices(pred_posed_data)
        pred_vertices_hat = compute_similarity_transform_batch(pred_vertices, gt_vertices)
        error = euclidean_distance(pred_vertices_hat, gt_vertices) if procrustes else euclidean_distance(pred_vertices, gt_vertices)
        return error

    def joints3d_transl_err(self, gt, pred):
        pelvis_id = 0
        return euclidean_distance(gt[:, pelvis_id:pelvis_id+1, :], pred[:, pelvis_id:pelvis_id+1, :])

    def joints3d_mpjpe(self, gt, pred):
        return lsp_joint_error(gt, pred)

    def joints3d_mpjpe_pa(self, gt, pred):
        return lsp_joint_error(gt, pred, procrustes=True)
    
    def blazeposeghum_33_transl_err(self, gt, pred):
        mid_hip_gt = (gt[:, 23:24, ...] + gt[:, 24:25, ...]) / 2.0
        mid_hip_pred = (pred[:, 23:24, ...] + pred[:, 24:25, ...]) / 2.0        
        return euclidean_distance(mid_hip_gt, mid_hip_pred)    

    def blazeposeghum_33_mpjpe(self, gt, pred):
        pred = center_blazeposeghum_33(pred)
        gt = center_blazeposeghum_33(gt)
        error = euclidean_distance(pred, gt)
        return error

    def blazeposeghum_33_mpjpe_pa(self, gt, pred):
        pred = center_blazeposeghum_33(pred)
        gt = center_blazeposeghum_33(gt)
        pred_hat = compute_similarity_transform_batch(pred, gt)
        error = euclidean_distance(pred_hat, gt)
        return error

    def ghum_mpvpe(self, gt, pred):
        return self._vertex_error(gt, pred, model_type='ghum')

    def ghum_mpvpe_pa(self, gt, pred):
        return self._vertex_error(gt, pred, model_type='ghum', procrustes=True)

    def smplx_mpvpe(self, gt, pred):
        return self._vertex_error(gt, pred, model_type='smplx')

    def smplx_mpvpe_pa(self, gt, pred):
        return self._vertex_error(gt, pred, model_type='smplx', procrustes=True)
    
    def get_metric_fns(self, preds):
        data_type_to_metric_fns = {
            'joints3d': [self.joints3d_transl_err, self.joints3d_mpjpe, self.joints3d_mpjpe_pa], 
            'blazeposeghum_33': [self.blazeposeghum_33_transl_err, self.blazeposeghum_33_mpjpe, self.blazeposeghum_33_mpjpe_pa], 
            'gpp': [self.ghum_mpvpe, self.ghum_mpvpe_pa], 
            'smplx': [self.smplx_mpvpe, self.smplx_mpvpe_pa]
        }
        metric_fns = []
        for subj_name in preds:
            for action_name in preds[subj_name]:
                pred = preds[subj_name][action_name]
                for key in pred['persons'][0]:
                    if key in data_type_to_metric_fns:
                        metric_fns += data_type_to_metric_fns[key]
                break
            break
        return metric_fns


    def get_bboxes(self, persons):
        bboxes = []
        for person in persons:
            person_bboxes = person['bbox']['bbox']
            bboxes.append(person_bboxes)
        return bboxes

    def match_persons_by_bbox(self, pred_persons, gt_persons, cam_params):
        if len(pred_persons) == 1 and len(gt_persons) == 1:
            return pred_persons

        data_types = [key for key in pred_persons[0]]
        pred_bboxes = self.get_bboxes(pred_persons)
        gt_bboxes = self.get_bboxes(gt_persons)
        
        new_pred_persons = copy.deepcopy(pred_persons)
        for data_type in data_types:
            pred = np.array(pred_bboxes).transpose((1, 0, 2))
            gt = np.array(gt_bboxes).transpose((1, 0, 2))
            iou_0 = compute_iou_distance(pred, gt).sum(axis=1)
            iou_1 = compute_iou_distance(pred[:, [1, 0], :], gt).sum(axis=1)
            change_order = iou_0 < iou_1
            for person_id in range(len(gt_persons)):
                for key in new_pred_persons[person_id][data_type]:
                    new_pred_persons[person_id][data_type][key][change_order] = pred_persons[1-person_id][data_type][key][change_order]            
      
        return new_pred_persons

    def compute_seq_metrics(self, gts, preds, metric_fns, has_contact_fr_id):
        seq_results = {}
        for subj_name in gts:
            seq_results[subj_name] = {}
            for action_name in gts[subj_name]:
                seq_results[subj_name][action_name] = {}
                gt_persons = gts[subj_name][action_name]['persons']
                pred_persons = preds[subj_name][action_name]['persons']

                if has_contact_fr_id:
                    frame_id = gts[subj_name][action_name]['other']['contact_fr_id']
                    image_id = gts[subj_name][action_name]['other']['video_fr_ids'].index(frame_id)

                cam_params = gts[subj_name][action_name]['other']['cam_params']

                pred_persons = self.match_persons_by_bbox(pred_persons, gt_persons, cam_params)

                pred = merge_persons(pred_persons)
                gt = merge_persons(gt_persons)

                for metric_fn in metric_fns:
                    metric_name = metric_fn.__name__
                    if metric_name.startswith('joints3d'):
                        gt_data = gt['joints3d']['joints3d']
                        pred_data = pred['joints3d']['joints3d']
                    elif metric_name.startswith('blazeposeghum_33'):
                        gt_data = gt['blazeposeghum_33']['blazeposeghum_33']
                        pred_data = pred['blazeposeghum_33']['blazeposeghum_33']
                    elif metric_name.startswith('ghum'):
                        gt_data = gt['gpp']
                        pred_data = pred['gpp']
                    elif metric_name.startswith('smplx'):
                        gt_data = gt['smplx']
                        pred_data = pred['smplx']
                    metric_result = metric_fn(gt_data, pred_data)
                    seq_results[subj_name][action_name][metric_name] = np.mean(metric_result)
                    if has_contact_fr_id:
                        seq_results[subj_name][action_name][metric_name + '_c'] = metric_result[image_id]
        return seq_results


    def eval_challenge(self, data_pred_path, data_gt_path, data_template_path):
        all_metric_names = ["joints3d_transl_err", "joints3d_mpjpe", "joints3d_mpjpe_pa", "blazeposeghum_33_transl_err", "blazeposeghum_33_mpjpe", "blazeposeghum_33_mpjpe_pa", "ghum_mpvpe", "ghum_mpvpe_pa", "smplx_mpvpe", "smplx_mpvpe_pa"]
        is_valid, message = validate_pred_format(data_pred_path, data_template_path)
        if is_valid:
            data_pred = load_data(data_pred_path)
            data_gt = load_data(data_gt_path)
            dataset_name = get_gt_info(data_gt)
            has_contact_fr_id = dataset_name in ['chi3d', 'humansc3d']
            metric_fns = self.get_metric_fns(data_pred)
            seq_results = self.compute_seq_metrics(data_gt, data_pred, metric_fns, has_contact_fr_id)
            agg_results = aggregate_metrics(metric_fns, seq_results, has_contact_fr_id)
            for key in all_metric_names:
                if key not in agg_results:
                    agg_results[key] = -1.0
            return agg_results
        else:
            return {"format_error": message}





