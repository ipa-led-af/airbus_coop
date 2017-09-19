#!/usr/bin/env python

import sys
import rospy
import tf
import numpy as np
from airbus_ssm_core import ssm_state
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from gazebo_msgs.srv import SpawnModel
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from moveit_msgs.msg import ExecuteTrajectoryActionResult 
import ast
import copy
from moveit_msgs.msg import RobotTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint


class MoveCart(ssm_state.ssmState):
	'''@SSM
	Description :  Action skill that makes a moveit move group move to a given target (cartesian) 
	User-data :
	- group : a moveit move group name, by default the end effector will try to match his 
	frame with the target frame (string)
	- target : a position and orientation of a target frame ([['pos.x','pos.y','pos.z','roll','pitch','yaw']])
			  (it could be a list of target : [[target1],[target2],[target3]])
	- frame : the name of the reference frame (string)
	- offset : an offset for the target (['pos.x','pos.y','pos.z','roll','pitch','yaw'])
	Outcome :
	- success : sucessfully move to the target
	'''
	def __init__(self):
		ssm_state.ssmState.__init__(self,outcomes=["success"],io_keys=["group","target","frame","offset","speed","tolerance"])
		self.result = 0
		self.msg = ""
					
	def acquireFrameUD(self,ud):
		target_list = ast.literal_eval(ud)
		return target_list  
		
	def createOffset(self,target,offset):
		tf_ = tf.TransformerROS()
		target_quat = self.eulerToQuaternion(target)
		offset_quat = self.eulerToQuaternion(offset)
		target_matrix = tf_.fromTranslationRotation([target[0],target[1],target[2]],target_quat)
		offset_matrix = tf_.fromTranslationRotation([offset[0],offset[1],offset[2]],offset_quat)
		new_target_matrix = np.dot(target_matrix, offset_matrix)	
		trans = tf.transformations.translation_from_matrix(new_target_matrix)
		quat = tf.transformations.quaternion_from_matrix(new_target_matrix)
		translation = [trans[0],trans[1],trans[2]]
		quaternion = [quat[0],quat[1],quat[2],quat[3]]
		new_target = translation + quaternion
		return new_target
	
	def eulerToQuaternion(self,angles):
		roll = np.deg2rad(angles[3])
		pitch = np.deg2rad(angles[4])
		yaw = np.deg2rad(angles[5])
		quaternion_tf = tf.transformations.quaternion_from_euler(roll,pitch,yaw)
		quaternion = [quaternion_tf[0],quaternion_tf[1],quaternion_tf[2],quaternion_tf[3]]
		return quaternion
		
	def poseTarget(self,target_pos,frame):
		pose_target = PoseStamped()
		pose_target.header.frame_id = frame
		#Translations 
		pose_target.pose.position.x = target_pos[0]
		pose_target.pose.position.y = target_pos[1]
		pose_target.pose.position.z = target_pos[2]
		#Rotations				
		pose_target.pose.orientation.x = target_pos[3]
		pose_target.pose.orientation.y = target_pos[4]
		pose_target.pose.orientation.z = target_pos[5]
		pose_target.pose.orientation.w = target_pos[6]
		return pose_target
	
	def result_cb(self, msg):
		self.result = msg.result.error_code.val
		self.msg = msg.status.text

	def move(self, group, plan):
		self.result = 0
		group.execute(plan, wait=False)
		while(self.result == 0):
			if(self.preempt_requested()):
				group.stop()
			rospy.sleep(0.01)
		#rospy.sleep(1)

	def execution(self,ud):
		self.result_sub = rospy.Subscriber("/execute_trajectory/result", ExecuteTrajectoryActionResult, self.result_cb, queue_size=1)
		group = moveit_commander.MoveGroupCommander(ud.group)
		group.set_max_velocity_scaling_factor(float(ud.speed))
		group.set_goal_tolerance(float(ud.tolerance))
		group.clear_pose_targets()
		waypoints = []
		
		#Planification and Execution
		for pos in self.acquireFrameUD(ud.target):
			new_pos = self.createOffset(pos,self.acquireFrameUD(ud.offset))
			waypoints.append(self.poseTarget(new_pos,ud.frame).pose)

		(plan, fraction) = group.compute_cartesian_path(waypoints, eef_step=0.01, jump_threshold=0.0, avoid_collisions=True)
		#rospy.sleep(1)

		self.move(group,plan)
		if(self.result == 1): 
			rospy.loginfo("[MoveCartesian] : %s"%self.msg)
			return "success"
		else:
			rospy.logerr("[MoveCartesian] : %s"%self.msg)
			return 'preempt'
		


class MoveArti(ssm_state.ssmState):
	'''@SSM
	Description :  Action skill that makes a moveit move group move by commanding its joints (articular) 
	User-data :
	- group : a moveit move group name containing all the joints to move (string)
	- joints : desired angle values of all joints ([['J1','J2','J3','J4','J5','J6']])
		      (it could be a list of desired values : [[values1],[values2],[values3]])
	Outcome :
	- success : sucessfully move to all joint targets
	'''
	def __init__(self):
		ssm_state.ssmState.__init__(self,outcomes=["success"],io_keys=["group","joints","speed","tolerance"])
		self.result = 0
		self.msg = ""
	
	def acquireJointsUD(self,ud):
		target_list = ast.literal_eval(ud)
		return target_list
	
	def result_cb(self, msg):
		self.result = msg.result.error_code.val
		self.msg = msg.status.text

	def move(self, group):
		self.result = 0
		plan = group.plan()
		group.execute(plan, wait=False)
		while(self.result == 0):
			if(self.preempt_requested()):
				group.stop()
			rospy.sleep(0.01)
		rospy.sleep(0.1)		
		
		
	def execution(self,ud):
		self.result_sub = rospy.Subscriber("/execute_trajectory/result", ExecuteTrajectoryActionResult, self.result_cb, queue_size=1)
		group = moveit_commander.MoveGroupCommander(ud.group)
		group.set_max_velocity_scaling_factor(float(ud.speed))
		group.set_goal_tolerance(float(ud.tolerance))
		group.clear_pose_targets()
		#Planification and Execution
		for target in self.acquireJointsUD(ud.joints):
			for i in range (0,len(target)):
				target[i] = np.deg2rad(target[i])
			group.set_joint_value_target(target)
			self.move(group)
			if(self.result != 1):
				rospy.logerr("[MoveArticular] : %s"%self.msg)
				return 'preempt'
			
		if(self.result == 1): 
			rospy.loginfo("[MoveArticular] : %s"%self.msg)
			return "success"
		else:
			rospy.logerr("[MoveArticular] : %s"%self.msg)
			return 'preempt'

	

class MoveBase(ssm_state.ssmState):
	'''@SSM
	Description :  Skill that makes a mobile base move
	User-data :
	- goal : the goal position for the mobile base
	- frame : the name of the reference frame (string)
	Outcome :
	- success : successfully move to the target
	'''
	def __init__(self):
		ssm_state.ssmState.__init__(self,outcomes=["success"],io_keys=["goal","frame"])
		
	def acquireFrameUD(self,ud):
		target_list = ast.literal_eval(ud)
		return target_list  
		
	def eulerToQuaternion(self,angles):
		roll = np.deg2rad(angles[3])
		pitch = np.deg2rad(angles[4])
		yaw = np.deg2rad(angles[5])
		quaternion_tf = tf.transformations.quaternion_from_euler(roll,pitch,yaw)
		quaternion = [quaternion_tf[0],quaternion_tf[1],quaternion_tf[2],quaternion_tf[3]]
		return quaternion
		
	def poseTarget(self,target_pos,frame):
		pose_target = PoseStamped()
		pose_target.header.frame_id = frame
		#Translations 
		pose_target.pose.position.x = target_pos[0]
		pose_target.pose.position.y = target_pos[1]
		pose_target.pose.position.z = target_pos[2]
		#Rotations				
		pose_target.pose.orientation.x = target_pos[3]
		pose_target.pose.orientation.y = target_pos[4]
		pose_target.pose.orientation.z = target_pos[5]
		pose_target.pose.orientation.w = target_pos[6]
		return pose_target
			
	def execution(self,ud):	
		print("MOVE BASE")
		client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
		client.wait_for_server()
	
		goal_target = self.acquireFrameUD(ud.goal)
		goal_translation = [goal_target[0],goal_target[1],goal_target[2]]
		goal_quaternion = self.eulerToQuaternion(goal_target)
		goal_tr = goal_translation + goal_quaternion

		poseStamp = self.poseTarget(goal_tr,ud.frame)
		goal = MoveBaseGoal(poseStamp)
				
		client.send_goal(goal)
		client.wait_for_result(rospy.Duration.from_sec(5.0))
		
		return "success"
		

		
class Pick(ssm_state.ssmState):
	'''@SSM
	Description :  Action skill that makes the tool pick an object
	User-data :
	- tool : a tool moveit move group name (string)
	- obj : the name of an object to pick (string)
	Outcome :
	- success : sucessfully picked the object
	'''
	def __init__(self):
		ssm_state.ssmState.__init__(self,outcomes=["success"],io_keys=["tool","obj"])
									  
	def execution(self,ud):
		#Getting the gripper move group
		group_gripper = moveit_commander.MoveGroupCommander(ud.tool)
		#Picking execution
		group_gripper.attach_object(ud.obj)
		rospy.sleep(1) 	
		return "success"



class Place(ssm_state.ssmState):
	'''@SSM
	Description :  Action skill that makes the tool place an object
	User-data :
	- tool : a tool moveit move group name (string)
	- obj : the name of an object to place (string)
	Outcome :
	- success : sucessfully place the object
	'''
	def __init__(self):
		ssm_state.ssmState.__init__(self,outcomes=["success"],io_keys=["tool","obj"])
									  
	def execution(self,ud):
		#Getting the gripper move group
		group_gripper = moveit_commander.MoveGroupCommander(ud.tool)
		#Placing execution
		group_gripper.detach_object(ud.obj)
		rospy.sleep(1)	 
		return "success"


class OpenGripper(ssm_state.ssmState):
	'''@SSM
	Description :  Action skill that open the gripper
	User-data :
	- tool : a tool moveit move group name (string)
	Outcome :
	- success : sucessfully picked the object
	'''
	def __init__(self):
		ssm_state.ssmState.__init__(self,outcomes=["success"],io_keys=["tool","obj"])	
		self.opened_gripper_value = [0.011,0.011]
														  
	def execution(self,ud):
		print("OPEN GRIPPER")
		#Getting the gripper move group
		group = moveit_commander.MoveGroupCommander(ud.tool)	
		print(group.get_current_joint_values())
		group.set_joint_value_target(self.opened_gripper_value)
		plan = group.plan()
		group.execute(plan)
		rospy.sleep(3) 	
		return "success"



class CloseGripper(ssm_state.ssmState):
	'''@SSM
	Description :  Action skill that close the gripper
	User-data :
	- tool : a tool moveit move group name (string)
	Outcome :
	- success : sucessfully place the object
	'''
	def __init__(self):
		ssm_state.ssmState.__init__(self,outcomes=["success"],io_keys=["tool","obj"])
		self.closed_gripper_value = [0.001,0.001]	
							  
	def execution(self,ud):
		print("CLOSE GRIPPER")
		#Getting the gripper move group
		group = moveit_commander.MoveGroupCommander(ud.tool)			
		group.set_joint_value_target(self.closed_gripper_value)
		plan = group.plan()
		group.execute(plan)
		rospy.sleep(3)	 
		return "success"


	  
class Sensor(ssm_state.ssmState):
	'''@SSM
	Description :  Skill that wait for the sensor information (position of an object)
	User-data :
	- objpos : the position of the object
	Outcome :
	- success : sucessfully got the information
	'''
	def __init__(self):
		ssm_state.ssmState.__init__(self,outcomes=["success"],io_keys=["objpos"])	
		self._msg_receive = False
		
	def callback(self,data):
		self._msg_receive = True
		self.x=data.pose.position.x
		self.y=data.pose.position.y
		self.z=data.pose.position.z
		
	def execution(self,ud):	
		
		self._msg_receive = False
		self.sensor_info = rospy.Subscriber("/obj/pose", PoseStamped, self.callback)
		while(self._msg_receive == False):
			if(self.preempt_requested()):
				return False
			rospy.sleep(0.1)
		
		ud.objpos = [self.x,self.y,self.z]
		
		return "success"



class RobotInfo(ssm_state.ssmState):
	'''@SSM
	Description :  Info skill that displays information about the used robot
	User-data : N/A
	Outcome :
	- success : sucessfully displayed information
	'''
	def __init__(self):
		ssm_state.ssmState.__init__(self,outcomes=["success"])
		
	def execution(self,ud):
		robot = moveit_commander.RobotCommander()
		#Displays informations about the robot configuration
		print("============ Robot Groups:")
		print(robot.get_group_names())
		print("============ Printing robot state")
		print(robot.get_current_state())
		print("============")
		return "success"



class GroupInfo(ssm_state.ssmState):
	'''@SSM
	Description :  Info skill that displays information about a selected moveit move group
	User-data :
	- group : a name of a robot moveit move group (string)
	Outcome :
	- success : sucessfully displayed information
	'''
	def __init__(self):
		ssm_state.ssmState.__init__(self,outcomes=["success"],io_keys=["group"])

	def execution(self,ud):
		group = moveit_commander.MoveGroupCommander(ud.group)
		#Displays informations about the group configuration	
		print("============ Reference frame: %s" % group.get_planning_frame())
		print("============ End effector: %s" % group.get_end_effector_link())
		print("============ Position of End Effector: %s" % group.get_current_pose())
		return "success"





class InitObject(ssm_state.ssmState):
	'''@SSM
	Description :  Skill that initialize an object to pick
	User-data : N/A
	Outcome :
	- success : sucessfully initialized the object
	'''
	def __init__(self):
		ssm_state.ssmState.__init__(self,outcomes=["success"])
		self.scene = moveit_commander.PlanningSceneInterface()
			
	def createPose(self,x,y,z):
		p = PoseStamped()
		p.header.frame_id = "/world"
		p.pose.position.x = x
		p.pose.position.y = y
		p.pose.position.z = z
		p.pose.orientation.x = 0
		p.pose.orientation.y = 0
		p.pose.orientation.z = 0
		p.pose.orientation.w = 1
		return p
		
	def execution(self,ud):			
		self.scene.add_box("object", self.createPose(0.55,0.3,0.96), (0.1,0.1,0.1))
		return "success"
		
#0,55 0,3 1,15

class GenEnvironment(ssm_state.ssmState):
	'''@SSM
	Description :  Skill that generate the environment
	User-data : N/A
	Outcome :
	- success : sucessfully created the environment
	'''
	def __init__(self):
		ssm_state.ssmState.__init__(self,outcomes=["success"])
		
	def createPose(self,x,y,z):
		p = PoseStamped()
		p.header.frame_id = "/world"
		p.pose.position.x = x
		p.pose.position.y = y
		p.pose.position.z = z
		p.pose.orientation.x = 0
		p.pose.orientation.y = 0
		p.pose.orientation.z = 0
		p.pose.orientation.w = 1
		return p
		
	def execution(self,ud):	
		scene = moveit_commander.PlanningSceneInterface()	
		scene.add_box("ground", self.createPose(0,0,-0.01), (10, 10, 0.02))
		scene.add_box("table1", self.createPose(0.5,0,0.25), (0.45, 0.45, 0.5))
		scene.add_box("table2", self.createPose(0,0.5,0.25), (0.45, 0.45, 0.5))
		return "success"
