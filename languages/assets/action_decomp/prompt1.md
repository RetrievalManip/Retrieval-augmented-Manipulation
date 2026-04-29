Pretend you are a robot with a parallel jaw gripper. You are required to break down a given task into a actionable subtask list. And the task is decription by the instruction and a corresponding image of workspace of the robot. Some objects related to this instruction are labeled with green numbsers in the image.


You should give detailed description about each subtask based on the instruction and observation, including:
- subtask id: It represents the stage the current subtask in the long horizon task.
- subtask goal: It presents the goal of this action.
- action type: It represents the primitiveary action that the robot should conduct. The actions should be one of following: "grasp object", "lift object", "move object", "put and release object", "reset object". Besides, when the subtask is aming to grasp the object, you need to move the endo-effecter to the top of the target object to prepare for grasping.
- gripper status: It represents the status at the end of the subtask, including "open gripper", "close gripper".
- grasped object: It represent the grasped object by the jaw gripper, at the begining of the current subtask. The object name, label id and corresponding category should be given. 
- target objects: It represents the referece coordination when moving the grasped object.  The object name, label id and corresponding category should be given.
- like the object above which the gripper should move to or the object that the gripper should grasp. 
- obstacal objects: It respresents the objects that robots need to avoid in the current subtask. It should be a list, for every object in this list, the name and corresponding id should be given.


When conduct the subtask, you are adepted at determining the spatial constraints that the robot girpper should conduct upon the subtask completion, based on the subtask description and the observation. For each substask, you are only allowed to use the predefined primitives (like `centroid` for centroid-type primitve, `grasp_N` for the grasp-type primitive or `plane_N` for the plane-type primitve) of objects as a reference system to describe the spatial constraints of each subtask. 


Besides, you are required to use a set of steps to complete this subtask based on the constraint_upon_completion. You are only allowed to replace the content in `[]` in below step's description. The placeholder `plane_N` and `plane_M` means the plane-type primitive of the object, and they can not be `centroid`. The placeholder`grasp_N` means the grasp-type primitive of the object. You are only allowed to use the object's primitve which we already known in the observation.
For each step of constraints, the step's description must be one of follows:
- Constraint Step Type 1: 
  - step description: The gripper should at the [grasp_N] of [target_objects] to grasp this object.
- Constraint Step Type 2:
  - step description: The [plane_N] of [grasped_object] should be [positive or negative] [number] cm to the centroid of [target_objects] along the [x-axis, y-axis or z-axis].
  - notice: It is worth noting that the placeholder [plane_N] can not be [centroid]!
- Constraint Step Type 3: 
  - step description: The [plane_N] of [grasped_object] should be [positive or negative] [number] cm and paralleled to the [plane_M] of [target_objects] along its positive direction.
  - notice: It is worth noting that the placeholder [plane_N] can not be [centroid]!
- Constraint Step Type 4:
  - step description: The direction of [planeN] of [grasped_object] should be align with the [positive or negative] direction of [x-axis or y-axis].
- Constraint Step Type 5:
  - step description: The bounding box of [grasped_object] should closer to the bounding box of [target_object], the closer gap should be [number] cm.
- Constraint Step Type 7:
  - step description: The [plane_N] of [grasped_object] should be [positive or negative] [number] cm and paralleled to this plane's original position along its positive direction.
- Constraint Step Type 8:
  - step description: Tile the normal direction of [plane_N] of [grasped_object] should [up or down] [numbder] degree.
- Constraint Step Type 9:
  - step description: Rotate the [grasped_object] [clockwise or counterclockwise] around the itself's z-axis by [angle] degrees.
- Constraint Step Type 10:
  - step description: The centroid of [grasped_object] should be [positive or negative] [x] cm to the centroid of [target_objects] along the [x-axis, y-axis or z-axis].
- Constraint Step Type 11:
  - step description: The gripper should at the [contact_point_N] of [target_objects], and let the contact point rotated [number] degrees along the hinge to [close or open] the lip.
- Constraint Step Type 12:
  - step description: The gripper should at the [contact_point_N] of [target_objects] to prepare the next operation.
- Constraint Step Type 13:
  - step description: The gripper should grasp [grasp_N] of [target_objects] to make it on the same position of [grasp_M] of this object.
- Constraint Step Type 14:
  - step description: The [grasp_N] of [grasped_object] should be [positive or negative] [number] cm to the centroid of [target_objects] along the [x-axis, y-axis or z-axis].

For the workspace coordinate of the robot, we define x-axis, y-axis and z-aixs as follows:
- The positive direction of x-axis is oriented horizontally on the table surface and points towards the left side of the observation image.
- The positive direction of y-axis is oriented horizontally on the table surface and points towards to the bottom side of the observation image.
- The positive directionb of z-axis will be vertically upwards, perpendicular to the table surface, pointing away from the table's surface.
- The start position of  robot arm is on the right side of this observed image.
- The workspace have a black floor.


Besides, we have some notices:
- Use constraint step type 7 to lift object.
- when you stack the objects with another object, you should align the base plane of the grasped object with centroid the target object in z-axis in the put and release action. 
You should let "The [base_plane] of [grasp_object] should be [positive] [0] cm to the centroid of [target_object] along the [z-axis]."


Based on the given instruction, the observation and also the already known object information in the observation.
Finally, you need to give your chain of thought, and structure your output in a single jason code block as follows:
```json
[
    {
        "subtask_id" : "int type, the i-th subtask of all subtasklists ",
        "subtask_goal" : "the goal of this subtask",
        "action_type": "primitiveary action type in this subtask",
        "gripper_status": "grasp or release",
        "grasped_object": {"name": "the grasped object name.", "id": "int type, the corresponding label if of this object"}, 
        "target_objects": [
            {"name": "the target object name", "id": "int type, the corresponding label if of this object", "category": "the category this object belong to"},
        ],
        "obstacle_objects": [ 
            {"name": "the obstacle object name", "id": "int type, the corresponding label if of this object", "category": "the category this object belong to"},
        ],
        "constraint_upon_completion": "description of spatial constraints",
        "constraint_steps": [
            {"step_type": "int type, the constrint step type", "step_description": "description of step of constraint",}
        ]
    },
]
```

To help you understand, I will show you an example:

Example1:
Observation: