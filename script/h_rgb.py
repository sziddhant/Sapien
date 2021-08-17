from os import name
import numpy as np
from gym import spaces
from numpy.lib.npyio import load

import sapien.core as sapien
from sapien.core import Pose
from sapien.utils.viewer import Viewer
from sapien_env import SapienEnv
import cv2

from PIL import Image, ImageColor

# from stable_baselines3 import PPO, SAC
# from stable_baselines3.common.env_checker import check_env
from hbaselines.algorithms import RLAlgorithm

from hbaselines.fcnet.sac import FeedForwardPolicy  # for SAC


cube_pos = 0
class LiftEnv(SapienEnv):
    def __init__(self):
        self.init_qpos = [0, -0.5019634954084936207, 0.0, -2.317993877991494,
                          0.0, 2.941592653589793, 0.7853981633974483, 0, 0]
        self.table_height = 0.8
        super().__init__(control_freq=20, timestep=0.01)

        self.robot = self.get_articulation('panda')
        self.end_effector = self.robot.get_links()[8]
        self.dof = self.robot.dof
        assert self.dof == 9, 'Panda should have 9 DoF'
        self.active_joints = self.robot.get_active_joints()
        self.dishwasher = self.get_articulation('dishwasher')
        self.cam = self.get_actor('cam')
        #self.img = np.zeros([480,640,4])
        self.cube = self.get_actor('cube')
        self.cube_pose= self.cube.get_pose()
        self.img = np.zeros([120,160,3])

        # self.cube = self.get_actor('cube')

        for joint in self.active_joints[:5]:
            joint.set_drive_property(stiffness=0, damping=4.8)
        for joint in self.active_joints[5:7]:
            joint.set_drive_property(stiffness=0, damping=0.72)

        #self.observation_space = spaces.Box(low=0, high=255, shape=(480,640,4), dtype=np.uint8)
        self.observation_space = spaces.Box(low=0, high=255, shape=(120,160,3), dtype=np.uint8)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=[self.dof], dtype=np.float32)

        self.max_episode_length = 100
        self.curr_step = 0

    # ---------------------------------------------------------------------------- #
    # Simulation world
    # ---------------------------------------------------------------------------- #
    def _build_world(self):
        physical_material = self._scene.create_physical_material(1.0, 1.0, 0.0)
        self._scene.default_physical_material = physical_material
        self._scene.add_ground(0.0)
        
        # dishwasher
        loader = self._scene.create_urdf_loader()
        loader.fix_root_link = True
        dishwasher = loader.load('../assets/12536/mobility.urdf') 
        dishwasher.set_name('dishwasher')
        dishwasher.set_root_pose(Pose([0.4, 0, 0.4]))
        dishwasher.set_qpos([1])

        # cube
        builder = self._scene.create_actor_builder()
        builder.add_box_collision(half_size=[0.02, 0.02, 0.02])
        builder.add_box_visual(half_size=[0.02, 0.02, 0.02], color=[1, 0, 0])
        cube = builder.build(name='cube')
        cube.set_pose(Pose([0.4, 0, 0.3],[1,0, 0, 0]))#(Pose([0.3, 0, 0.2]))

        # robot
        loader = self._scene.create_urdf_loader()
        loader.fix_root_link = True
        robot = loader.load('../assets/robot/panda/panda.urdf') #('../assets/robot/panda/panda.urdf') 
        robot.set_name('panda')
        robot.set_root_pose(Pose([-0.5, 0, 0]))
        robot.set_qpos(self.init_qpos)

        #camera
        near, far = 0.1, 100
        width, height = 640, 480
        camera_mount_actor = self._scene.create_actor_builder().build_kinematic(name="cam")
        self.camera = self._scene.add_mounted_camera(
        name="camera",
        actor=camera_mount_actor,
        pose=sapien.Pose(),  # relative to the mounted actor
        width=width,
        height=height,
        fovx=np.deg2rad(35),
        fovy=np.deg2rad(35),
        near=near,
        far=far)
        print('Intrinsic matrix\n', self.camera.get_camera_matrix())
        # ef= robot.get_links()[8].get_pose()
        # # print(type(ef),ef)
        camera_mount_actor.set_pose(Pose([-1.6, -0.65, 0.4],[0.9848078,0, 0, 0.1736482]))

    # ---------------------------------------------------------------------------- #
    # RL
    # ---------------------------------------------------------------------------- #
    def step(self, action):
        global cube_pos
        # Use internal velocity drive
        for idx in range(7):
            self.active_joints[idx].set_drive_velocity_target(action[idx])

        # Control the gripper by torque
        qf = self.robot.compute_passive_force(True, True, False)
        qf[-2:] += action[-2:]
        self.robot.set_qf(qf)

        for i in range(self.control_freq):
            self._scene.step()
            self._scene.update_render()
            self.camera.take_picture()
        # self.cam.set_pose(self.end_effector.get_pose())
        rgba = self.camera.get_float_texture('Color')  # [H, W, 4]
        rgba_img = (rgba * 255).clip(0, 255).astype("uint8")
        rgba_pil = Image.fromarray(rgba_img)
        # rgba_pil.show()
        rgba_pil = rgba_pil.resize((160, 120))

        rgb_img = rgba_pil.convert('RGB')
        rgb_img = np.array(rgb_img)

        #Depth
#        position = self.camera.get_float_texture('Position')
#        depth = -position[..., 2]
#        depth_image = (depth * 1000.0).astype(np.uint8)
#        depth_pil = Image.fromarray(depth_image)
#        # # depth_pil.show()
#        img = np.dstack([rgb_img,depth_image])
#        # print(img.shape)

        self.img = rgb_img
        
        
        obs = self._get_obs()
        reward = self._get_reward()

        done = False
        # done = self.dishwasher.get_qpos() > 0.8
        done = self.cube.get_pose().p[2] - self.cube_pose.p[2] > 0.2 
        done = bool(done)
        # print(reward)

        if done:
            reward += 100.0
        # print(reward)
        reward -= 0.5

        # Terminate when max episode length is reached
        self.curr_step += 1
        if self.curr_step > self.max_episode_length:
            done = True

        return obs, reward, done, {}

    def reset(self):
#        print("reset:/")
        global cube_pos
        self.robot.set_qpos(self.init_qpos)
        self.dishwasher.set_qpos([0])
        self.cube.set_pose(Pose([0.4,0,0.3]))
        self.cube_pose= self.cube.get_pose()
        self._scene.step()
        self.curr_step = 0
        return self._get_obs()

    def _get_obs(self):
        # qpos = self.robot.get_qpos()
        # qvel = self.robot.get_qvel()
        # d_qpos = self.dishwasher.get_qpos()
        # # cube_pose = self.cube.get_pose()
        # d_pose = self.dishwasher.get_pose()

        # ee_pose = self.end_effector.get_pose()
        # d_to_ee = ee_pose.p - d_pose.p
        # d_to_ee_q = ee_pose.q - d_pose.q
        # # ob = np.hstack([qpos, qvel, cube_pose.p, cube_pose.q, cube_to_ee])
        # print (ob)
        # Vector = np.hstack([qpos, qvel, d_pose.p, d_pose.q, d_to_ee, d_to_ee_q])
        # print(Vector.shape)
        Img = self.img #np.stack([self.img[:,:,0],self.img[:,:,1],self.img[:,:,2]],axis=2)
        # print(Img.shape)
        # spc = {'image': Img, 'vector': Vector}
        return Img

    def _get_reward(self):
        # reaching reward
        # d_pose = self.dishwasher.get_pose()
        # ee_pose = self.end_effector.get_pose()
        # distance = np.linalg.norm(ee_pose.p - cube_pose.p)
        # reaching_reward = 1.0 - np.tanh(1.0 * distance)
        lift_reward = self.cube_pose.p[2] - self.cube.get_pose().p[2]
        opening_reward = 1.0* np.sin(self.dishwasher.get_qpos())

        return float( opening_reward)

    # ---------------------------------------------------------------------------- #
    # Visualization
    # ---------------------------------------------------------------------------- #
    def _setup_lighting(self):
        rscene = self._scene.get_renderer_scene()
        rscene.set_ambient_light([.4, .4, .4])
        rscene.add_directional_light([1, -1, -1], [0.3, 0.3, 0.3])
        rscene.add_directional_light([0, 0, -1], [1, 1, 1])

    def _setup_viewer(self):
        self._setup_lighting()
        self.viewer = Viewer(self._renderer)
        self.viewer.set_scene(self._scene)
        self.viewer.set_camera_xyz(x=1.5, y=0.0, z=2.0)
        self.viewer.set_camera_rpy(y=3.14, p=-0.5, r=0)



def main():

    env = LiftEnv()
    env.reset()


    alg = RLAlgorithm(policy=FeedForwardPolicy, env=env, total_steps=10000)
    alg.learn(total_timesteps=10000)

    # model = PPO.load("RGBD_PPO")
    # model.set_env(env)
    # model.learn(total_timesteps=10000, log_interval=4)
    # model.save("RGB_SAC")

    # model = PPO.load("Lift_PPO2l")
    # # model.set_env(env)
    dones = False
#    obs = env.reset()
    while not dones:
#        action, _states = model.predict(obs)
       action = env.action_space.sample()
       obs, rewards, dones, info = env.step(action)
       env.render()



if __name__ == '__main__':
    main()
