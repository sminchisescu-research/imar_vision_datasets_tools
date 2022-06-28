import os
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
import ghum
from ghum import ghum_utils
from ghum import ghum_lib
from ghum import gpp_utils
from ghum.data_objects import skeleton_data as skeleton_data_lib
from ghum.io import io
import tensorflow.compat.v2 as tf
from ghum import shmodel
import numpy as np
import attr
tf.enable_v2_behavior()


class GHUMHelper:
    def __init__(self, Models_Path=None, load_renderer=True):
        self.Models_Path = Models_Path
        self.Model_Type = 'GHUM'  #@param ['GHUM', 'GHUMLITE', 'GHUM_HEAD', 'GHUM_LEFT_HAND', 'GHUM_RIGHT_HAND']
        self.Skeleton_Joints_Dof_Type = 'ROTATION_6D'  #@param ['ROTATION_6D', 'EULER_ANGLES']
        self.Shape_Type = 'NEUTRAL'  #@param ['FEMALE', 'MALE', 'NEUTRAL']
        self.Shape_Decoder_Type = 'NONLINEAR_VAE'  #@param ['NONLINEAR_VAE', 'LINEAR_PCA']
        self.Expression_Decoder_Type = 'NONLINEAR_VAE'  #@param ['NONLINEAR_VAE', 'LINEAR_PCA']
        self.load_ghum_model()
        self.load_skeleton_data()
        self.image_shape = (900, 900)
        if load_renderer:
            from ghum.visualizations import ghum_mesh_renderer as ghum_renderer
            self.mesh_rasterizer = ghum_renderer.GHUMMeshRasterizer(height=self.image_shape[0], 
                                                                width=self.image_shape[1], 
                                                                ghum_model=self.ghum_model, 
                                                                model_type=self.model_type)
        else:
            self.mesh_rasterizer = None
        
    def load_ghum_model(self):
        # Path for the GHUM(L) models path, e.g. '/tmp/ghumrepo/ghum/shmodels'
        if not self.Models_Path:
            raise ValueError(f'Expected a valid path for the GHUM models, got {Models_Path}.')

        self.model_type = ghum_utils.ModelType(self.Model_Type)
        joint_dof_type = ghum_utils.JointDofType(self.Skeleton_Joints_Dof_Type)
        shape_type = ghum_utils.ShapeType(self.Shape_Type)
        shape_decoder_type = ghum_utils.ShapeDecoderType(
            self.Shape_Decoder_Type)
        expression_decoder_type = ghum_utils.ExpressionDecoderType(
            self.Expression_Decoder_Type)

        self.ghum_parameters = ghum_utils.create_ghum_parameters(
            self.model_type, shape_type, joint_dof_type, shape_decoder_type,
            expression_decoder_type, self.Models_Path)

        self.ghum_model = ghum_lib.create_ghum_from_parameters(self.ghum_parameters)
    
    def load_skeleton_data(self):
        template_proto = io.read_template_proto(self.ghum_parameters.template_path)
        self.skeleton_data = skeleton_data_lib.create_skeleton_data(template_proto)
        
    def get_world_gpp(self, gpps):        
        default_gpp = gpp_utils.default_gpp(381, 16, 20, 1)
        batch_size = len(gpps['posing_values'])
        world_gpp = attr.evolve(
            default_gpp,
            skeleton_posing_values=tf.convert_to_tensor(gpps['posing_values']),
            body_shape_code = tf.convert_to_tensor(gpps['body_code'])
        )
        return world_gpp
    
    def get_camera_gpp(self, gpps, cam_params): 
        batch_size = len(gpps['posing_values'])
        world_gpp = self.get_world_gpp(gpps)
        world_posed_data = self.ghum_model.pose(world_gpp)
        root_rest_r_joint = tf.tile(self.skeleton_data.root_rest_t_joint[tf.newaxis, ..., :3],
                           (batch_size, 1, 1))
        root_rest_p_joint = world_posed_data.rest_joints[:, 0:1, :]
        root_rest_t_joint = tf.concat((root_rest_r_joint, tf.transpose(root_rest_p_joint, perm=[0, 2, 1])), axis=-1)
        camera_gpp = gpp_utils.update_gpp_with_global_rt(
            world_gpp, tf.cast(tf.constant(cam_params['extrinsics']['R'][tf.newaxis, ...]), tf.float32), 
            tf.cast(tf.constant(-np.matmul(cam_params['extrinsics']['R'], cam_params['extrinsics']['T'].T).T), tf.float32),
            root_rest_t_joint
        )
        return camera_gpp
    
    def get_template_params(self, batch_size=1):
        gpps = gpp_utils.default_gpp(381, 16, 20, batch_size)
        body_code = gpps.body_shape_code.numpy()
        posing_values = gpps.skeleton_posing_values.numpy()
        posing_values[:, 3] = -1
        posing_values[:, 6] = -1
        return {'body_code': body_code, 'posing_values': posing_values}
        
    def get_template(self):
        default_gpp = gpp_utils.default_gpp(381, 16, 20, 1)
        ghum_posed_data = self.ghum_model.pose(default_gpp)
        ghum_template = {'vertices': ghum_posed_data.vertices[0], 'triangles': self.ghum_model.triangles()}
        return ghum_template

    def render(self, vertices, frame, cam_params, vertices_in_world=True):
        vertices = vertices.numpy()
        if vertices_in_world:
            vertices = np.matmul(vertices - cam_params['extrinsics']['T'], np.transpose(cam_params['extrinsics']['R']))

        vertices_to_render = tf.convert_to_tensor(vertices)
        rendered_image = self.mesh_rasterizer.rasterize(
          vertices_to_render,
          intrinsics=cam_params['intrinsics_wo_distortion']['f'].tolist() + cam_params['intrinsics_wo_distortion']['c'].tolist(),
          background_image=frame,
          blending_weight=1.0)
        return rendered_image
