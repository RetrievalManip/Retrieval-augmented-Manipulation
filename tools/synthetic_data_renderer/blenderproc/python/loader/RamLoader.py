"""Provides functions to load the objects inside the bop dataset."""

import os
from random import choice
from typing import List, Optional
import warnings

from mathutils import Vector

from blenderproc.python.types.MeshObjectUtility import MeshObject
from blenderproc.python.loader.ObjectLoader import load_obj

def load_ram_objs(bop_dataset_path: str, model_type: str = "", obj_ids: Optional[List[int]] = None,
                  sample_objects: bool = False, num_of_objs_to_sample: Optional[int] = None,
                  obj_instances_limit: int = -1, mm2m: Optional[bool] = None, object_model_unit: str = 'm',
                  move_origin_to_x_y_plane: bool = False) -> List[MeshObject]:
    """ Loads all or a subset of 3D models of any BOP dataset

    :param bop_dataset_path: Full path to a specific bop dataset e.g. /home/user/bop/tless.
    :param model_type: Unused compatibility parameter.
    :param obj_ids: List of object ids to load. Default: [] (load all objects from the given BOP dataset)
    :param sample_objects: Toggles object sampling from the specified dataset.
    :param num_of_objs_to_sample: Amount of objects to sample from the specified dataset. If this amount is bigger
                                  than the dataset actually contains, then all objects will be loaded.
    :param obj_instances_limit: Limits the amount of object copies when sampling. Default: -1 (no limit).
    :param mm2m: Specify whether to convert poses and models to meters (deprecated).
    :param object_model_unit: The unit the object model is in. Object model will be scaled to meters. This does not
                              affect the annotation units. Available: ['m', 'dm', 'cm', 'mm'].
    :param move_origin_to_x_y_plane: Move center of the object to the lower side of the object, this will not work
                                     when used in combination with pose estimation tasks! This is designed for the
                                     use-case where BOP objects are used as filler objects in the background.
    :return: The list of loaded mesh objects.
    """

    bop_dataset_name = 'ram'

    assert object_model_unit in ['m', 'dm', 'cm', 'mm'], (f"Invalid object model unit: `{object_model_unit}`. "
                                                          f"Supported are 'm', 'dm', 'cm', 'mm'")
    scale = {'m': 1., 'dm': 0.1, 'cm': 0.01, 'mm': 0.001}[object_model_unit]
    if mm2m is not None:
        warnings.warn("WARNING: `mm2m` is deprecated, please use `object_model_unit='mm'` instead!")
        scale = 0.001

    if obj_ids is None:
        raise ValueError("obj_ids must be provided for RAM object loading.")

    loaded_objects = []
    # if sampling is enabled
    if sample_objects:
        loaded_ids = {}
        loaded_amount = 0
        if obj_instances_limit != -1 and len(obj_ids) * obj_instances_limit < num_of_objs_to_sample:
            raise RuntimeError(f"{bop_dataset_path}'s contains {len(obj_ids)} objects, {num_of_objs_to_sample} object "
                               f"where requested to sample with an instances limit of {obj_instances_limit}. Raise "
                               f"the limit amount or decrease the requested amount of objects.")
        while loaded_amount != num_of_objs_to_sample:
            random_id = choice(obj_ids)
            if random_id not in loaded_ids:
                loaded_ids.update({random_id: 0})
            # if there is no limit or if there is one, but it is not reached for this particular object
            if obj_instances_limit == -1 or loaded_ids[random_id] < obj_instances_limit:
                cur_obj = _RamLoader.load_mesh(random_id, bop_dataset_path, bop_dataset_name, scale)
                loaded_ids[random_id] += 1
                loaded_amount += 1
                loaded_objects.append(cur_obj)
            else:
                print(f"ID {random_id} was loaded {loaded_ids[random_id]} times with limit of {obj_instances_limit}. "
                      f"Total loaded amount {loaded_amount} while {num_of_objs_to_sample} are being requested")
    else:
        for obj_id in obj_ids:
            cur_obj = _RamLoader.load_mesh(obj_id, bop_dataset_path, bop_dataset_name, scale)
            loaded_objects.append(cur_obj)
    # move the origin of the object to the world origin and on top of the X-Y plane
    # makes it easier to place them later on, this does not change the `.location`
    # This is only useful if the BOP objects are not used in a pose estimation scenario.
    if move_origin_to_x_y_plane:
        for obj in loaded_objects:
            obj.move_origin_to_bottom_mean_point()

    return loaded_objects

class _RamLoader:
    CACHED_OBJECTS = {}

    @staticmethod
    def load_mesh(obj_id: int, bop_dataset_path: str, bop_dataset_name: str, scale: float = 1) -> MeshObject:
        """ Loads BOP mesh and sets category_id.

        :param obj_id: The obj_id of the BOP Object.
        :param bop_dataset_name: The name of the used bop dataset.
        :param scale: factor to transform set pose in mm or meters.
        :return: Loaded mesh object.
        """

        model_path = os.path.join(bop_dataset_path, 'models', 'obj_' + str(obj_id).zfill(6) + '.ply')

        # if the object was not previously loaded - load it, if duplication is allowed - duplicate it
        duplicated = model_path in _RamLoader.CACHED_OBJECTS
        objs = load_obj(model_path, cached_objects=_RamLoader.CACHED_OBJECTS)
        assert (
            len(objs) == 1
        ), f"Loading object from '{model_path}' returned more than one mesh"
        cur_obj = objs[0]

        if duplicated:
            # See issue https://github.com/DLR-RM/BlenderProc/issues/590
            for i, material in enumerate(cur_obj.get_materials()):
                material_dup = material.duplicate()
                cur_obj.set_material(i, material_dup)

        # Change Material name to be backward compatible
        cur_obj.get_materials()[-1].set_name("bop_" + bop_dataset_name + "_vertex_col_material")
        cur_obj.set_scale(Vector((scale, scale, scale)))
        cur_obj.set_cp("category_id", obj_id)
        cur_obj.set_cp("model_path", model_path)
        cur_obj.set_cp("is_bop_object", True)
        cur_obj.set_cp("bop_dataset_name", bop_dataset_name)

        return cur_obj
