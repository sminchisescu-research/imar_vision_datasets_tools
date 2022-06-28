import trimesh
import numpy as np
import json

color_black = np.array([[0, 0, 0, 255]])
color_gray = np.array([[120, 120, 120, 255]])
color_red = [255, 0, 0, 255]

class SignatureVisualizer():
    def __init__(self, path_to_template, model_type, models_path):
        self.path_to_template = path_to_template
        if model_type == 'GHUM':
            from util.ghum_util import GHUMHelper
            ghum_helper = GHUMHelper(models_path)
            self.mesh_template = ghum_helper.get_template()
        elif model_type == 'SMPLX':
            from util.smplx_util import SMPLXHelper
            smplx_helper = SMPLXHelper(models_path)
            self.mesh_template = smplx_helper.get_template()
        self.contact_regions_template = self.load_json('contact_regions.json')
        self.meshes = [trimesh.Trimesh(vertices=self.mesh_template['vertices'], 
                                      faces=self.mesh_template['triangles']) 
                      for person_id in range(2)]
        self.rid_to_fids = self.contact_regions_template['rid_to_ghum_fids'] if model_type == 'GHUM' else self.contact_regions_template['rid_to_smplx_fids']
        self.rid_to_color_default = {rid: self.contact_regions_template['rid_to_color'][rid] 
                                     for rid in range(len(self.contact_regions_template['rid_to_color']))}
        
    def load_json(self, file_name):
        fn = '%s/%s' % (self.path_to_template, file_name)
        with open(fn) as f:
            data = json.load(f)
        return data
    
    def color_mesh_regions(self, a_trimesh, rid_to_color):        
        for rid in rid_to_color:
            a_trimesh.visual.face_colors[self.rid_to_fids[rid]] = rid_to_color[rid]
            
    def color_mesh_facets(self, a_trimesh, fids):
        a_trimesh.visual.face_colors[fids] = color_black
    
    def interaction_contact_signature(self, signature):
        for person_id in range(2):
            self.color_mesh_regions(self.meshes[person_id], self.rid_to_color_default)
            rid_to_color_in_red = {corresp[person_id]: color_red for corresp in signature['region_id']}
            self.color_mesh_regions(self.meshes[person_id], rid_to_color_in_red)
            fids = [corresp[person_id] for corresp in signature['face_id']]
            self.color_mesh_facets(self.meshes[person_id], fids)
            self.meshes[person_id].apply_translation([person_id*2, 0, 0])
        return trimesh.Scene(self.meshes)
    
    def self_contact_signature(self, signature):
        person_id = 0
        self.color_mesh_regions(self.meshes[person_id], self.rid_to_color_default)
        rid_to_color_in_red = {reg_id: color_red for reg_id in sum(signature['region_id'], [])}
        self.color_mesh_regions(self.meshes[person_id], rid_to_color_in_red)
        fids = sum(signature['face_id'], [])
        self.color_mesh_facets(self.meshes[person_id], fids)
        return self.meshes[person_id]




